import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

allowable_features = {
    'possible_atomic_num_list': list(range(0, 119)),
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
}

def get_MMFF_atom_poses(mol):
    """根据开题报告，使用 ETKDG 方法生成 3D 近似结构"""
    try:
        new_mol = Chem.AddHs(mol)
        # 使用开题报告指定的 ETKDG 方法
        AllChem.EmbedMolecule(new_mol, AllChem.ETKDG()) 
        AllChem.MMFFOptimizeMolecule(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        conf = new_mol.GetConformer()
        return [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] 
                for i in range(mol.GetNumAtoms())]
    except:
        # 构象生成失败时返回 2D 投影坐标作为占位
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        return [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, 0.0] 
                for i in range(mol.GetNumAtoms())]

def get_geometry_info(mol):
    """提取键角和二面角的索引及数值，并处理构象缺失错误"""
    try:
        conf = mol.GetConformer()
    except ValueError:
        # 处理 Bad Conformer Id 错误，返回空特征
        return (torch.zeros((0, 3), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float),
                torch.zeros((0, 4), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float))

    angles_idx, angles_val = [], []
    dihedrals_idx, dihedrals_val = [], []

    # 提取键角 (i-j-k)
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) < 2: continue
        for idx_i in range(len(neighbors)):
            for idx_k in range(idx_i + 1, len(neighbors)):
                i, k = neighbors[idx_i], neighbors[idx_k]
                angles_idx.append([i, j, k])
                angles_val.append(Chem.rdMolTransforms.GetAngleRad(conf, i, j, k))

    # 提取二面角 (i-j-k-l)
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for n_j in mol.GetAtomWithIdx(j).GetNeighbors():
            i = n_j.GetIdx()
            if i == k: continue
            for n_k in mol.GetAtomWithIdx(k).GetNeighbors():
                l = n_k.GetIdx()
                if l == j or l == i: continue
                dihedrals_idx.append([i, j, k, l])
                dihedrals_val.append(Chem.rdMolTransforms.GetDihedralRad(conf, i, j, k, l))

    if not angles_idx:
        return (torch.zeros((0, 3), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float),
                torch.zeros((0, 4), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float))

    return (torch.tensor(angles_idx, dtype=torch.long), 
            torch.tensor(angles_val, dtype=torch.float).view(-1, 1),
            torch.tensor(dihedrals_idx, dtype=torch.long),
            torch.tensor(dihedrals_val, dtype=torch.float).view(-1, 1))

def get_two_graph(mol, args):
    atom_features = [[allowable_features['possible_atomic_num_list'].index(a.GetAtomicNum()),
                      allowable_features['possible_chirality_list'].index(a.GetChiralTag())] 
                     for a in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_features), dtype=torch.long)
    row, col = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row.extend([i, j]); col.extend([j, i])
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return x, mol.GetNumAtoms(), edge_index

def get_three_graph(mol, args):
    x, atom_size, edge_index = get_two_graph(mol, args)
    pos = torch.tensor(get_MMFF_atom_poses(mol), dtype=torch.float)
    a_idx, a_val, d_idx, d_val = get_geometry_info(mol)
    return x, atom_size, edge_index, pos, a_idx, a_val, d_idx, d_val