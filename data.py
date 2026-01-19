# data.py 完整修复版

from rdkit import Chem
import torch
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import random
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from graph import get_two_graph, get_three_graph
from tqdm import tqdm

class MoleData:
    def __init__(self, line, args):
        self.args = args
        self.smile = line[0]
        self.mol = Chem.MolFromSmiles(self.smile)
        self.label = [float(x) if x != '' else None for x in line[1:]]
        self.features = None # 用于存放预计算的特征

    def task_num(self):
        return len(self.label)
    
    def change_label(self, label):
        self.label = label

class MoleDataSet(Dataset):
    def __init__(self, data):
        # 【修复关键点】增加自动解包逻辑，防止嵌套封装导致 shuffle 失败
        if isinstance(data, MoleDataSet):
            self.data = data.data
        else:
            self.data = data
            
        if len(self.data) > 0:
            self.args = self.data[0].args
        else:
            self.args = None
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def task_num(self):
        """返回任务数量，解决 AttributeError"""
        if len(self.data) > 0:
            return self.data[0].task_num()
        return 0

    def precompute_all(self, args):
        """预计算所有模态特征，解决 GPU 利用率低的问题"""
        print("Pre-computing molecular features (3D conformers, FP, Graphs)...")
        for one in tqdm(self.data):
            if one.mol is None: continue
            
            # 1. 指纹
            fp = self.morgan_gen.GetFingerprintAsNumPy(one.mol)
            
            # 2. 2D 图
            x_2d, _, edge_2d = get_two_graph(one.mol, args)
            
            # 3. 3D 及几何图 (包含键角和二面角)
            x_3d, _, edge_3d, pos, a_idx, a_val, d_idx, d_val = get_three_graph(one.mol, args)
            
            one.features = {
                'fp': torch.from_numpy(fp).float(),
                'g2d': {'x': x_2d, 'edge': edge_2d},
                'g3d': {
                    'x': x_3d, 'edge': edge_3d, 'pos': pos,
                    'a_idx': a_idx, 'a_val': a_val,
                    'd_idx': d_idx, 'd_val': d_val
                }
            }

    def smile(self): return [one.smile for one in self.data]
    def label(self): return [one.label for one in self.data]
    def mol(self): return [one.mol for one in self.data]
    def __len__(self): return len(self.data)
    
    def __getitem__(self, key): 
        """支持切片，但通过构造函数自动处理解包"""
        if isinstance(key, slice):
            return MoleDataSet(self.data[key])
        return self.data[key]
    
    # 新增 __setitem__ 使类更具列表特性，支持某些特殊 shuffle 操作
    def __setitem__(self, key, value):
        self.data[key] = value

    def random_data(self, seed):
        """打乱数据顺序"""
        random.seed(seed)
        # 此时 self.data 保证为 list，shuffle 将正常运行
        random.shuffle(self.data)

    def change_label(self, label):
        for i in range(len(label)):
            self.data[i].change_label(label[i])

# --- 骨架划分相关逻辑（补全 scaffold_split_balanced） ---

def generate_scaffold(mol, include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

def scaffold_to_smiles(mols, use_indices=False):
    scaffolds = defaultdict(set)
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)
    return scaffolds

def scaffold_split(data, size, seed, log):
    assert sum(size) == 1
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)
    index_sets = sorted(list(scaffold_to_indices.values()), key=lambda index_set: len(index_set), reverse=True)
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
        elif len(val) + len(index_set) <= val_size:
            val += index_set
        else:
            test += index_set
    return MoleDataSet([data[i] for i in train]), MoleDataSet([data[i] for i in val]), MoleDataSet([data[i] for i in test])

def scaffold_split_balanced(data, size, seed, log):
    assert sum(size) == 1
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)
    index_sets = list(scaffold_to_indices.values())
    big_index_sets, small_index_sets = [], []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
        elif len(val) + len(index_set) <= val_size:
            val += index_set
        else:
            test += index_set
    return MoleDataSet([data[i] for i in train]), MoleDataSet([data[i] for i in val]), MoleDataSet([data[i] for i in test])