# train.py 完整修复版
from argparse import Namespace
from logging import Logger
import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

from tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model
from model import FPGNN
from data import MoleDataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epoch_train(model, data, loss_f, optimizer, scheduler, args):
    model.train()
    data.random_data(args.seed) 
    loss_sum = 0
    iter_step = args.batch_size
    
    for i in range(0, len(data), iter_step):
        if i + iter_step > len(data): break
        
        indices = list(range(i, i + iter_step))
        batch_features = [data[idx].features for idx in indices]
        label = [data[idx].label for idx in indices]

        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label]).to(device)
        mask = torch.Tensor([[x is not None for x in tb] for tb in label]).to(device)
        
        model.zero_grad()
        pred = model(args, batch_features)
        loss = (loss_f(pred, target) * mask).sum() / mask.sum()
        loss_sum += loss.item()
        
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR): scheduler.step()
            
    if isinstance(scheduler, ExponentialLR): scheduler.step()
    return loss_sum

def evaluate(model, data, loss_f, args):
    model.eval()
    loss_sum = 0
    iter_step = args.batch_size
    
    with torch.no_grad():
        for i in range(0, len(data), iter_step):
            if i + iter_step > len(data): break
            indices = list(range(i, i + iter_step))
            batch_features = [data[idx].features for idx in indices]
            label = [data[idx].label for idx in indices]

            target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label]).to(device)
            mask = torch.Tensor([[x is not None for x in tb] for tb in label]).to(device)
            
            pred = model(args, batch_features)
            loss = (loss_f(pred, target) * mask).sum() / mask.sum()
            loss_sum += loss.item()
            
    return loss_sum

def predict(model, data, batch_size, scaler, args):
    model.eval()
    pred = []
    data_total = len(data)
    
    for i in range(0, data_total, batch_size):
        end = min(i + batch_size, data_total)
        indices = list(range(i, end))
        batch_features = [data[idx].features for idx in indices]
        
        with torch.no_grad():
            pred_now = model(args, batch_features)
        
        pred_now = pred_now.data.cpu().numpy()
        if scaler is not None:
            ave, std = scaler[0], scaler[1]
            pred_now = np.array(pred_now).astype(float)
            pred_now = pred_now * std + ave
        
        pred.extend(pred_now.tolist())
    return pred

def compute_score(pred, label, metric_f, args, log):
    info = log.info
    task_num = args.task_num
    data_type = args.dataset_type
    
    if len(pred) == 0: return [float('nan')] * task_num
    
    result = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] is not None:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        
        if data_type == 'classification':
            if len(set(label_val_i)) < 2:
                info(f'Warning: Task {i} labels are constant. AUC is not defined.')
                result.append(float('nan'))
                continue
        
        re = metric_f(label_val_i, pred_val_i)
        result.append(re)
    return result

def fold_train(args, log):
    info, debug = log.info, log.debug
    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path, args)
    
    # 1. 预计算特征
    data.precompute_all(args)
    
    args.task_num = data.task_num()
    if args.task_num > 1: args.is_multitask = 1
    
    # 2. 划分数据集
    train_data, val_data, test_data = split_data(data, args.split_type, args.split_ratio, args.seed, log)
    
    # 【修复关键点】给 args 赋值训练集大小，供 NoamLR 使用
    args.train_data_size = len(train_data)
    
    debug(f'Train size: {args.train_data_size}  Val size: {len(val_data)}  Test size: {len(test_data)}')
    
    label_scaler = get_label_scaler(train_data) if args.dataset_type == 'regression' else None
    loss_f = get_loss(args.dataset_type)
    metric_f = get_metric(args.metric)
    
    model = FPGNN(args).to(device) if args.cuda else FPGNN(args)
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    
    # 初始化调度器，现在 args.train_data_size 已经存在了
    scheduler = NoamLR(optimizer, [args.warmup_epochs], [args.epochs], 
                       args.train_data_size // args.batch_size, 
                       [args.init_lr], [args.max_lr], [args.final_lr])

    best_score = -float('inf') if args.dataset_type == 'classification' else float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        train_loss = epoch_train(model, train_data, loss_f, optimizer, scheduler, args)
        
        val_loss = evaluate(model, val_data, loss_f, args) 
        val_pred = predict(model, val_data, args.batch_size, label_scaler, args)
        val_score = compute_score(val_pred, val_data.label(), metric_f, args, log)
        ave_val_score = np.nanmean(val_score)

        info(f'Validation {args.metric} = {ave_val_score:.6f} | loss = {val_loss:.6f}')

        if (args.dataset_type == 'classification' and ave_val_score > best_score) or \
           (args.dataset_type == 'regression' and ave_val_score < best_score):
            best_score, best_epoch = ave_val_score, epoch
            save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)

    info(f'Best Validation {args.metric} = {best_score:.6f} at Epoch {best_epoch}')
    
    # 测试评估
    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log)
    test_pred = predict(model, test_data, args.batch_size, label_scaler, args)
    test_score = compute_score(test_pred, test_data.label(), metric_f, args, log)
    info(f'Test {args.metric} = {np.nanmean(test_score):.6f}')
    
    return test_score