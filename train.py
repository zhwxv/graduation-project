# train.py 完整修改版 (集成早停机制)
from argparse import Namespace
from logging import Logger
import os
import csv
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR

# 从 tool 导入绘图工具和相关辅助函数
from tool import mkdir, get_task_name, load_data, split_data, get_label_scaler, get_loss, get_metric, save_model, NoamLR, load_model, save_training_curves
from model import FPGNN
from data import MoleDataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epoch_train(model, data, loss_f, optimizer, scheduler, args):
    model.train()
    data.random_data(args.seed) 
    loss_sum = 0
    iter_step = args.batch_size
    num_iters = 0 # 记录实际运行的 batch 数量
    
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
        num_iters += 1
        
        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR): scheduler.step()
            
    if isinstance(scheduler, ExponentialLR): scheduler.step()
    # 返回平均 Loss
    return loss_sum / num_iters if num_iters > 0 else 0

def evaluate(model, data, loss_f, args):
    model.eval()
    loss_sum = 0
    iter_step = args.batch_size
    num_iters = 0
    
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
            num_iters += 1
            
    # 返回平均 Loss
    return loss_sum / num_iters if num_iters > 0 else 0

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
    args.train_data_size = len(train_data)
    
    debug(f'Train size: {args.train_data_size}  Val size: {len(val_data)}  Test size: {len(test_data)}')
    
    label_scaler = get_label_scaler(train_data) if args.dataset_type == 'regression' else None
    loss_f = get_loss(args.dataset_type)
    metric_f = get_metric(args.metric)
    
    model = FPGNN(args).to(device) if args.cuda else FPGNN(args)
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    
    scheduler = NoamLR(optimizer, [args.warmup_epochs], [args.epochs], 
                       args.train_data_size // args.batch_size, 
                       [args.init_lr], [args.max_lr], [args.final_lr])

    # 用于可视化的数据记录
    train_losses, val_losses, val_scores = [], [], []

    best_score = -float('inf') if args.dataset_type == 'classification' else float('inf')
    best_epoch = 0
    patience_counter = 0 

    for epoch in range(args.epochs):
        info(f'Epoch {epoch}')
        train_loss = epoch_train(model, train_data, loss_f, optimizer, scheduler, args)
        
        val_loss = evaluate(model, val_data, loss_f, args) 
        val_pred = predict(model, val_data, args.batch_size, label_scaler, args)
        val_score = compute_score(val_pred, val_data.label(), metric_f, args, log)
        ave_val_score = np.nanmean(val_score)

        # 记录数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_scores.append(ave_val_score)

        # 【修改点 1】将绘图函数移至循环内部
        # 这样每跑完一个 Epoch，图片就会被覆盖更新，你可以实时通过查看图片观察收敛情况
        save_training_curves(train_losses, val_losses, val_scores, args.metric, args.save_path)

        info(f'Validation {args.metric} = {ave_val_score:.6f} | loss = {val_loss:.6f}')

        # 检查是否有性能提升
        if (args.dataset_type == 'classification' and ave_val_score > best_score) or \
           (args.dataset_type == 'regression' and ave_val_score < best_score):
            best_score, best_epoch = ave_val_score, epoch
            save_model(os.path.join(args.save_path, 'model.pt'), model, label_scaler, args)
            patience_counter = 0  
        else:
            patience_counter += 1  

        # 触发早停判断
        if hasattr(args, 'patience') and patience_counter >= args.patience:
            info(f'Early stopping at epoch {epoch}. No improvement for {args.patience} epochs.')
            break

    # 【可选】此处保留一次最终绘图，确保最后的数据被完整记录
    save_training_curves(train_losses, val_losses, val_scores, args.metric, args.save_path)
    
    info(f'Best Validation {args.metric} = {best_score:.6f} at Epoch {best_epoch}')
    
    # 测试评估
    model = load_model(os.path.join(args.save_path, 'model.pt'), args.cuda, log)
    test_pred = predict(model, test_data, args.batch_size, label_scaler, args)
    test_score = compute_score(test_pred, test_data.label(), metric_f, args, log)
    info(f'Test {args.metric} = {np.nanmean(test_score):.6f}')
    
    return test_score