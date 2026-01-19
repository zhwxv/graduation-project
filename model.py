import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

num_atom_type = 120
num_chirality_tag = 4

# --- 1. 分子指纹分支 (FPN) ---
class FPN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, args.fp_2_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout_fpn),
            nn.Linear(args.fp_2_dim, args.hidden_size)
        )
    def forward(self, fps_tensor):
        return self.fc(fps_tensor)

# --- 2. 2D 拓扑分支 (GCN) ---
class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 嵌入原子序数和手性
        self.emb_atom = nn.Embedding(num_atom_type, args.emb_dim_gnn)
        self.emb_chiral = nn.Embedding(num_chirality_tag, args.emb_dim_gnn)
        self.conv1 = GCNConv(args.emb_dim_gnn, args.hidden_gnn)
        self.conv2 = GCNConv(args.hidden_gnn, args.hidden_gnn)

    def forward(self, batch_g2d):
        device = next(self.parameters()).device
        outs = []
        for g in batch_g2d:
            x = g['x'].to(device)
            h = self.emb_atom(x[:, 0]) + self.emb_chiral(x[:, 1])
            h = F.leaky_relu(self.conv1(h, g['edge'].to(device)), 0.2)
            h = F.leaky_relu(self.conv2(h, g['edge'].to(device)), 0.2)
            outs.append(h.mean(dim=0))
        return torch.stack(outs)

# --- 3. 3D 基础等变分支 (EGNN) ---
class EGNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phi_e = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.phi_h = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU())

    def forward(self, h, edge_index, pos):
        # 仅基于距离的消息传递
        dist = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1).view(-1, 1)
        m_ij = self.phi_e(torch.cat([h[edge_index[0]], h[edge_index[1]], dist], dim=-1))
        agg_e = torch.zeros_like(h).index_add_(0, edge_index[0], m_ij)
        return h + self.phi_h(torch.cat([h, agg_e], dim=-1))

class EGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(num_atom_type, args.emb_egnn)
        self.layer = EGNNLayer(args.emb_egnn)

    def forward(self, batch_g3d):
        device = next(self.parameters()).device
        outs = []
        for g in batch_g3d:
            h = self.emb(g['x'][:, 0].to(device))
            h = self.layer(h, g['edge'].to(device), g['pos'].to(device))
            outs.append(h.mean(dim=0))
        return torch.stack(outs)

# --- 4. 3D 高阶几何分支 (GEGNN) ---
class GEGNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.phi_angle = nn.Sequential(nn.Linear(dim * 3 + 1, dim), nn.SiLU())
        self.phi_dihedral = nn.Sequential(nn.Linear(dim * 4 + 1, dim), nn.SiLU())
        self.phi_h = nn.Linear(dim * 3, dim)

    def forward(self, h, a_idx, a_val, d_idx, d_val):
        # 捕捉键角消息
        agg_a = torch.zeros_like(h)
        if a_idx.shape[0] > 0:
            m_ijk = self.phi_angle(torch.cat([h[a_idx[:,0]], h[a_idx[:,1]], h[a_idx[:,2]], a_val], dim=-1))
            agg_a.index_add_(0, a_idx[:, 1], m_ijk)
        # 捕捉二面角消息
        agg_d = torch.zeros_like(h)
        if d_idx.shape[0] > 0:
            m_ijkl = self.phi_dihedral(torch.cat([h[d_idx[:,0]], h[d_idx[:,1]], h[d_idx[:,2]], h[d_idx[:,3]], d_val], dim=-1))
            agg_d.index_add_(0, d_idx[:, 1], m_ijkl)
        return h + self.phi_h(torch.cat([h, agg_a, agg_d], dim=-1))

class GEGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(num_atom_type, args.emb_egnn)
        self.layer = GEGNNLayer(args.emb_egnn)

    def forward(self, batch_g3d):
        device = next(self.parameters()).device
        outs = []
        for g in batch_g3d:
            h = self.emb(g['x'][:, 0].to(device))
            h = self.layer(h, g['a_idx'].to(device), g['a_val'].to(device), g['d_idx'].to(device), g['d_val'].to(device))
            outs.append(h.mean(dim=0))
        return torch.stack(outs)

# --- 5. GAT 多模态融合模块 ---
class GATFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(2 * dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, features):
        batch_size = features.size(0)
        num_m = features.size(1) 
        h = self.W(features)
        h_i = h.repeat_interleave(num_m, dim=1)
        h_j = h.repeat(1, num_m, 1)
        e = self.leakyrelu(torch.matmul(torch.cat([h_i, h_j], dim=-1), self.a).view(batch_size, num_m, num_m))
        alpha = F.softmax(e, dim=-1)
        return torch.matmul(alpha, h).sum(dim=1) # 改为 sum 增强表达能力

# --- 6. 总体整合模型 ---
class FpgnnModel(nn.Module):
    def __init__(self, is_classif, args):
        super().__init__()
        self.is_classif = is_classif
        self.fpn = FPN(args)
        self.gcn = GCN(args)
        self.egnn = EGNN(args)   # 开题报告分支 3
        self.gegnn = GEGNN(args) # 开题报告分支 4
        
        # 维度对齐与层归一化
        self.align = nn.ModuleList([nn.Linear(d, args.linear_dim) for d in [args.hidden_size, args.hidden_gnn, args.emb_egnn, args.emb_egnn]])
        self.ln = nn.ModuleList([nn.LayerNorm(args.linear_dim) for _ in range(4)])
        
        self.fusion = GATFusion(args.linear_dim)
        self.predict_head = nn.Sequential(
            nn.Linear(args.linear_dim, args.linear_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(args.dropout_fpn),
            nn.Linear(args.linear_dim, args.task_num)
        )
        if self.is_classif: self.sigmoid = nn.Sigmoid()

    def forward(self, args, batch_features):
        device = next(self.parameters()).device
        fps = torch.stack([f['fp'] for f in batch_features]).to(device)
        g2ds = [f['g2d'] for f in batch_features]
        g3ds = [f['g3d'] for f in batch_features]

        # 并行提取并归一化
        f1 = self.ln[0](self.align[0](self.fpn(fps)))
        f2 = self.ln[1](self.align[1](self.gcn(g2ds)))
        f3 = self.ln[2](self.align[2](self.egnn(g3ds)))
        f4 = self.ln[3](self.align[3](self.gegnn(g3ds))) 
        
        z = self.fusion(torch.stack([f1, f2, f3, f4], dim=1))
        logits = self.predict_head(z)
        return self.sigmoid(logits) if self.is_classif and not self.training else logits

def FPGNN(args):
    return FpgnnModel(args.dataset_type == 'classification', args)