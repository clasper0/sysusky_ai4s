# iterref_gnn_full.py
# 完整实现：支持 molecule_id 映射读取 -> 分子 featurize -> GCN encoder -> IterRef 模型 -> episodic training -> candidate 预测
# 注意：在小样本场景（如本题 200 个分子、5 个靶点），此模型为论文思想的可运行版本，建议作为 ensemble 的一员进行融合。

import os
import csv
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# -----------------------------
# 基本配置（按需修改）
# -----------------------------
DATA_DIR = "data"  # 数据目录
MOL_FILE = os.path.join(DATA_DIR, "molecule.smi")      # molecule.smi 文件路径
ACT_FILE = os.path.join(DATA_DIR, "activity_train.csv")# activity 训练文件（包含 molecule_id, target_id, pIC50）
CAND_FILE = os.path.join(DATA_DIR, "candidate.csv")    # candidate 文件（mol id, smiles）
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 超参数（可调）
MAX_NODES = 60         # 每个分子最大原子数（超过则跳过或提高此值）
SUPPORT_SIZE = 8
QUERY_SIZE = 8
N_STEPS = 2000
EMB_DIM = 128
GCN_HIDDEN = [128, 128]
ITER_DEPTH = 3
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# I. 文件读取：稳健读取 molecule.smi（支持常见格式）
# -----------------------------
def read_molecule_smi(path):
    """
    读取 molecule.smi 文件，返回 DataFrame (molecule_id, smiles)
    支持常见格式：
      - 两列：molecule_id , smiles 或 smiles , molecule_id
      - 一列：只有 smiles（则生成 molecule_id: MOL_0...）
      - 带 header 的 csv/tsv（尝试识别 'molecule_id' 与 'smiles' 列）
    返回 DataFrame，列名固定为 ['molecule_id', 'smiles']（均为字符串）
    """
    import pandas as pd
    # 先尝试以逗号读两列无 header
    try:
        df = pd.read_csv(path, sep=',', header=None, dtype=str, keep_default_na=False)
        if df.shape[1] == 2:
            col0 = df.iloc[:,0].astype(str)
            col1 = df.iloc[:,1].astype(str)
            def looks_like_smiles(s):
                if s is None: return False
                s = str(s)
                return any(ch in s for ch in ['=', '#', '/', '\\', 'c', 'C', 'N', 'O', '[', ']']) or len(s) > 1
            score0 = col0.map(looks_like_smiles).sum()
            score1 = col1.map(looks_like_smiles).sum()
            if score1 >= score0:
                df.columns = ['molecule_id','smiles']
            else:
                df.columns = ['smiles','molecule_id']
                df = df[['molecule_id','smiles']]
            df['molecule_id'] = df['molecule_id'].astype(str)
            df['smiles'] = df['smiles'].astype(str)
            return df[['molecule_id','smiles']]
    except Exception:
        pass
    # 尝试带 header 的读取
    try:
        df = pd.read_csv(path, dtype=str)
        cols = [c.lower() for c in df.columns]
        if 'molecule_id' in cols and 'smiles' in cols:
            mol_col = df.columns[cols.index('molecule_id')]
            smi_col = df.columns[cols.index('smiles')]
            out = df[[mol_col, smi_col]].copy()
            out.columns = ['molecule_id','smiles']
            out['molecule_id'] = out['molecule_id'].astype(str)
            out['smiles'] = out['smiles'].astype(str)
            return out
        if len(df.columns) == 1:
            smi_col = df.columns[0]
            out = pd.DataFrame({'molecule_id': ['MOL_'+str(i) for i in range(len(df))], 'smiles': df[smi_col].astype(str)})
            return out
    except Exception:
        pass
    # 兜底按行解析
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=',')
        rows = [r for r in reader if len(r) > 0]
    if len(rows) == 0:
        raise ValueError(f"无法读取 {path} 或文件为空")
    if all(len(r) == 1 for r in rows):
        mols = []
        for i, r in enumerate(rows):
            mols.append({'molecule_id': f'MOL_{i}', 'smiles': str(r[0])})
        return pd.DataFrame(mols)
    else:
        # 取前两列
        import pandas as pd
        df = pd.DataFrame(rows)
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['molecule_id','smiles']
            df['molecule_id'] = df['molecule_id'].astype(str)
            df['smiles'] = df['smiles'].astype(str)
            return df
        else:
            raise ValueError("无法解析 molecule.smi 格式，请检查文件")
# -----------------------------
# II. 分子 featurize（SMILES -> node feats, adj, mask）
# -----------------------------
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B']  # 常见元素列表

def atom_feature_vector(atom):
    """
    构建单个原子的特征向量（numpy array）
    包含：one-hot(atom type + other), degree bucket, implicit H, aromatic, formal charge, hybridization
    """
    at = atom.GetSymbol()
    type_onehot = [0] * (len(ATOM_LIST) + 1)
    if at in ATOM_LIST:
        type_onehot[ATOM_LIST.index(at)] = 1
    else:
        type_onehot[-1] = 1
    deg = atom.GetDegree()
    deg_bucket = [0]*6
    deg_bucket[min(deg,5)] = 1
    ih = atom.GetTotalNumHs()
    aromatic = [1 if atom.GetIsAromatic() else 0]
    charge = [atom.GetFormalCharge()]
    hyb = atom.GetHybridization()
    hyb_onehot = [
        1 if hyb == Chem.rdchem.HybridizationType.SP else 0,
        1 if hyb == Chem.rdchem.HybridizationType.SP2 else 0,
        1 if hyb == Chem.rdchem.HybridizationType.SP3 else 0
    ]
    feats = np.array(type_onehot + deg_bucket + [ih] + aromatic + charge + hyb_onehot, dtype=np.float32)
    return feats

def smiles_to_graph(smiles, max_nodes=MAX_NODES):
    """
    将 smiles 转成 graph 字典 {node_feats, adj, n_nodes}
    更鲁棒：处理 None / NaN / 解析失败，返回 None 表示无法使用该分子
    """
    try:
        if smiles is None:
            return None
        if isinstance(smiles, float) and math.isnan(smiles):
            return None
        if not isinstance(smiles, str):
            smiles = str(smiles)
        smiles = smiles.strip()
        if smiles == "" or smiles.lower() == "nan":
            return None
    except Exception:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # optional kekulize (容错)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        n = mol.GetNumAtoms()
        if n > max_nodes:
            # 可调整 max_nodes，或者直接跳过
            return None
        node_feats = []
        for atom in mol.GetAtoms():
            node_feats.append(atom_feature_vector(atom))
        node_feat_dim = node_feats[0].shape[0]
        node_feats = np.vstack(node_feats)
        adj = np.zeros((n, n), dtype=np.float32)
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            adj[a, b] = 1.0
            adj[b, a] = 1.0
        pad_feats = np.zeros((max_nodes, node_feat_dim), dtype=np.float32)
        pad_feats[:n, :] = node_feats
        pad_adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        pad_adj[:n, :n] = adj
        return {'node_feats': pad_feats, 'adj': pad_adj, 'n_nodes': n}
    except Exception:
        return None

# -----------------------------
# III. 构造 tasks_data（通过 molecule_id -> smiles 映射）
# -----------------------------
def build_tasks_data(mol_df, activity_df, max_nodes=MAX_NODES):
    """
    mol_df: DataFrame with columns ['molecule_id', 'smiles']
    activity_df: DataFrame with columns ['molecule_id', 'target_id', 'pIC50']
    返回 tasks_data: dict target_id -> list of {'molecule_id', 'graph', 'y'}
    """
    if 'molecule_id' not in mol_df.columns or 'smiles' not in mol_df.columns:
        raise ValueError("mol_df 必须包含 molecule_id 和 smiles 列")
    # 构建映射
    mol_map = dict(zip(mol_df['molecule_id'].astype(str), mol_df['smiles'].astype(str)))
    tasks = defaultdict(list)
    fail_records = []
    total = len(activity_df)
    merged_count = 0
    for idx, row in activity_df.iterrows():
        mol_id = row.get('molecule_id', None)
        target_id = row.get('target_id', None)
        # 读取 label
        try:
            y = float(row.get('pIC50', np.nan))
        except Exception:
            fail_records.append((idx, mol_id, None, target_id, 'bad_label'))
            continue
        if mol_id is None:
            fail_records.append((idx, mol_id, None, target_id, 'missing_molecule_id'))
            continue
        mol_id_str = str(mol_id)
        smi = mol_map.get(mol_id_str, None)
        if smi is None or (isinstance(smi, float) and math.isnan(smi)):
            fail_records.append((idx, mol_id_str, smi, target_id, 'smiles_not_found_in_mol_file'))
            continue
        g = smiles_to_graph(smi, max_nodes=max_nodes)
        if g is None:
            fail_records.append((idx, mol_id_str, smi, target_id, 'rdkit_failed_or_too_big'))
            continue
        mask = np.zeros((max_nodes,), dtype=np.float32)
        mask[:g['n_nodes']] = 1.0
        tasks[target_id].append({'molecule_id': mol_id_str, 'graph': {'node_feats': g['node_feats'], 'adj': g['adj'], 'mask': mask}, 'y': y})
        merged_count += 1
    print(f"[build_tasks_data] activity 总行数: {total}, 成功构造: {merged_count}, 失败: {len(fail_records)}")
    if len(fail_records) > 0:
        print("[build_tasks_data] 部分失败记录示例 (idx, molecule_id, smiles_or_None, target_id, reason):")
        for rec in fail_records[:30]:
            print(rec)
    return tasks

# -----------------------------
# IV. GCN 层与 GraphEncoder（batch friendly）
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)
    def forward(self, H, A, mask):
        """
        H: (B, N, in_dim)
        A: (B, N, N)
        mask: (B, N) 1 for real nodes
        """
        B, N, _ = H.shape
        I = torch.eye(N, device=H.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = A + I
        deg = A_hat.sum(dim=-1)  # (B,N)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        A_norm = torch.zeros_like(A_hat)
        for i in range(B):
            di = torch.diag(deg_inv_sqrt[i])
            A_norm[i] = di @ A_hat[i] @ di
        H_lin = self.linear(H)  # (B,N,out)
        out = torch.bmm(A_norm, H_lin)
        out = out * mask.unsqueeze(-1)
        return F.relu(out)

class GraphEncoder(nn.Module):
    def __init__(self, node_feat_dim, hidden_dims=[128,128], emb_dim=128, readout='mean'):
        super().__init__()
        layers = []
        dims = [node_feat_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(GCNLayer(dims[i], dims[i+1]))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Linear(dims[-1], emb_dim)
        self.readout = readout
    def forward(self, node_feats, adj, mask):
        """
        node_feats: (B,N,F)
        adj: (B,N,N)
        mask: (B,N)
        返回 graph embedding: (B, emb_dim)
        """
        x = node_feats
        for layer in self.layers:
            x = layer(x, adj, mask)
        if self.readout == 'sum':
            g = (x * mask.unsqueeze(-1)).sum(dim=1)
        elif self.readout == 'mean':
            s = (x * mask.unsqueeze(-1)).sum(dim=1)
            denom = mask.sum(dim=1, keepdim=True) + 1e-8
            g = s / denom
        elif self.readout == 'max':
            x_masked = x.clone()
            x_masked[mask==0] = -1e9
            g = x_masked.max(dim=1)[0]
        else:
            raise ValueError("readout must be sum/mean/max")
        g = self.proj(g)
        g = F.relu(g)
        return g

# -----------------------------
# V. IterRef 模块（iterative refinement attention）
# -----------------------------
class IterRefModule(nn.Module):
    def __init__(self, emb_dim=128, depth=3):
        super().__init__()
        self.emb_dim = emb_dim
        self.depth = depth
        self.lstm_q = nn.LSTMCell(emb_dim, emb_dim)
        self.lstm_s = nn.LSTMCell(emb_dim, emb_dim)
    def forward(self, q_embs, s_embs):
        """
        q_embs: (Bq, D)
        s_embs: (m, D)
        返回 attention weights: (Bq, m)
        """
        device = q_embs.device
        Bq, D = q_embs.shape
        m = s_embs.shape[0]
        hq = q_embs
        cq = torch.zeros_like(hq)
        hs = s_embs.clone()
        cs = torch.zeros_like(hs)
        for _ in range(self.depth):
            q_exp = hq.unsqueeze(1).expand(-1, m, -1)  # (Bq,m,D)
            s_exp = hs.unsqueeze(0).expand(Bq, -1, -1) # (Bq,m,D)
            sim = F.cosine_similarity(q_exp, s_exp, dim=-1)  # (Bq,m)
            a = F.softmax(sim, dim=-1)
            a_exp = a.unsqueeze(-1)
            r = (a_exp * s_exp).sum(dim=1)  # (Bq,D)
            hq, cq = self.lstm_q(r, (hq, cq))
            r_mean = r.mean(dim=0, keepdim=True).expand(m, -1)
            hs, cs = self.lstm_s(r_mean, (hs, cs))
        q_exp = hq.unsqueeze(1).expand(-1, m, -1)
        s_exp = hs.unsqueeze(0).expand(Bq, -1, -1)
        sim = F.cosine_similarity(q_exp, s_exp, dim=-1)
        a_final = F.softmax(sim, dim=-1)
        return a_final

# -----------------------------
# VI. 顶层模型：GraphEncoder + IterRef + residual predictor
# -----------------------------
class IterRefGNNRegressor(nn.Module):
    def __init__(self, node_feat_dim, gcn_hidden=[128,128], emb_dim=128, iter_depth=3):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim, hidden_dims=gcn_hidden, emb_dim=emb_dim, readout='mean')
        self.iterref = IterRefModule(emb_dim=emb_dim, depth=iter_depth)
        self.resid = nn.Sequential(nn.Linear(emb_dim, emb_dim//2), nn.ReLU(), nn.Linear(emb_dim//2, 1))
    def forward(self, q_graphs, s_graphs, s_labels):
        """
        q_graphs: dict with node_feats (Bq,N,F), adj (Bq,N,N), mask (Bq,N)
        s_graphs: dict with node_feats (m,N,F), adj (m,N,N), mask (m,N)
        s_labels: tensor (m,)
        返回: y_pred (Bq,), attention weights (Bq,m)
        """
        device = s_labels.device
        s_node_feats = s_graphs['node_feats'].to(device)
        s_adj = s_graphs['adj'].to(device)
        s_mask = s_graphs['mask'].to(device)
        s_emb = self.encoder(s_node_feats, s_adj, s_mask)  # (m, D)
        q_node_feats = q_graphs['node_feats'].to(device)
        q_adj = q_graphs['adj'].to(device)
        q_mask = q_graphs['mask'].to(device)
        q_emb = self.encoder(q_node_feats, q_adj, q_mask)  # (Bq, D)
        a = self.iterref(q_emb, s_emb)  # (Bq, m)
        s_lab = s_labels.unsqueeze(0).expand(a.size(0), -1)  # (Bq, m)
        y_weighted = (a * s_lab).sum(dim=1)
        resid = self.resid(q_emb).squeeze(-1)
        y_pred = y_weighted + resid
        return y_pred, a

# -----------------------------
# VII. Episodic 采样 & 训练
# -----------------------------
def sample_episode_from_tasks(tasks_data, support_size=SUPPORT_SIZE, query_size=QUERY_SIZE):
    """
    从 tasks_data 中随机采样一个 target_id -> 返回 support & query
    tasks_data[target_id] = list of items {'molecule_id', 'graph', 'y'}
    """
    task = random.choice(list(tasks_data.keys()))
    items = tasks_data[task]
    m = len(items)
    # 若样本少，使用带替换采样
    if m >= support_size + query_size:
        idxs = random.sample(range(m), k=(support_size + query_size))
    else:
        idxs = [random.randrange(m) for _ in range(support_size + query_size)]
    s_idxs = idxs[:support_size]
    q_idxs = idxs[support_size:]
    support = [items[i] for i in s_idxs]
    query = [items[i] for i in q_idxs]
    s_node = np.stack([it['graph']['node_feats'] for it in support])
    s_adj = np.stack([it['graph']['adj'] for it in support])
    s_mask = np.stack([it['graph']['mask'] for it in support])
    s_labels = torch.tensor([it['y'] for it in support], dtype=torch.float32)
    q_node = np.stack([it['graph']['node_feats'] for it in query])
    q_adj = np.stack([it['graph']['adj'] for it in query])
    q_mask = np.stack([it['graph']['mask'] for it in query])
    q_labels = torch.tensor([it['y'] for it in query], dtype=torch.float32)
    s_graphs = {'node_feats': torch.tensor(s_node, dtype=torch.float32),
                'adj': torch.tensor(s_adj, dtype=torch.float32),
                'mask': torch.tensor(s_mask, dtype=torch.float32)}
    q_graphs = {'node_feats': torch.tensor(q_node, dtype=torch.float32),
                'adj': torch.tensor(q_adj, dtype=torch.float32),
                'mask': torch.tensor(q_mask, dtype=torch.float32)}
    return task, s_graphs, s_labels, q_graphs, q_labels

def train_iterref_gnn(tasks_data, node_feat_dim, device=DEVICE,
                      n_steps=N_STEPS, support_size=SUPPORT_SIZE, query_size=QUERY_SIZE,
                      emb_dim=EMB_DIM, gcn_hidden=GCN_HIDDEN, iter_depth=ITER_DEPTH,
                      lr=LR, weight_decay=WEIGHT_DECAY,
                      print_every=200):
    model = IterRefGNNRegressor(node_feat_dim, gcn_hidden, emb_dim, iter_depth).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    running_loss = 0.0
    for step in range(1, n_steps+1):
        task, s_graphs, s_labels, q_graphs, q_labels = sample_episode_from_tasks(tasks_data, support_size, query_size)
        s_graphs_dev = {k: v.to(device) for k,v in s_graphs.items()}
        q_graphs_dev = {k: v.to(device) for k,v in q_graphs.items()}
        s_labels_dev = s_labels.to(device)
        q_labels_dev = q_labels.to(device)
        model.train()
        preds, _ = model(q_graphs_dev, s_graphs_dev, s_labels_dev)
        loss = F.mse_loss(preds, q_labels_dev)
        optim.zero_grad(); loss.backward(); optim.step()
        running_loss += loss.item()
        if step % print_every == 0:
            avg = running_loss / print_every
            print(f"[step {step}] avg_mse_loss: {avg:.6f}")
            running_loss = 0.0
    return model

# -----------------------------
# VIII. Candidate 预测（对每个 target，用 tasks_data[target_id] 全量作为 support）
# -----------------------------
def predict_candidates(model, tasks_data, cand_df, max_nodes=MAX_NODES, device=DEVICE):
    """
    更鲁棒的 candidate 预测函数（替换原函数）
    - 会检查 support items 中 graph 字段是否包含 node_feats/adj/mask
    - 会在出现问题时打印 debug 信息（哪些 support 出错、对应的索引与 molecule_id）
    - 对每个 target 单独处理，若该 target 的 support 构建失败会跳过该 target 并在输出中填 NaN
    """
    # 1) featurize candidates
    cand_graphs = []
    cand_ids = []
    fail = 0
    for idx, row in cand_df.iterrows():
        mid = row.get('molecule_id', None)
        smi = row.get('smiles', None)
        g = smiles_to_graph(smi, max_nodes=max_nodes)
        if g is None:
            cand_graphs.append(None)
            fail += 1
        else:
            cand_graphs.append(g)
        cand_ids.append(mid)
    print(f"[predict_candidates] candidate featurize 完成, unable_to_parse = {fail}, total = {len(cand_ids)}")

    results = {'molecule_id': cand_ids}
    model.eval()

    # iterate over targets
    for target, support_items in tasks_data.items():
        try:
            if len(support_items) == 0:
                print(f"[predict_candidates] target {target} has 0 support samples, skipping.")
                results[f'pred_{target}'] = [np.nan] * len(cand_ids)
                continue

            # 验证每个 support item 是否包含 graph 且 graph 有 node_feats/adj/mask
            bad_supports = []
            for si, it in enumerate(support_items):
                g = it.get('graph', None)
                if g is None:
                    bad_supports.append((si, it.get('molecule_id', None), 'no_graph'))
                    continue
                if not all(k in g for k in ('node_feats','adj','mask')):
                    bad_supports.append((si, it.get('molecule_id', None), f'keys={list(g.keys())}'))
            if len(bad_supports) > 0:
                print(f"[predict_candidates][WARN] target {target} 有 {len(bad_supports)} 个 support items 出问题，示例：")
                for rec in bad_supports[:10]:
                    print("  ", rec)
                # 如果太多坏 support，直接跳过该 target
                if len(bad_supports) >= len(support_items):
                    print(f"[predict_candidates][ERROR] target {target} 所有 support 均不可用，跳过此 target")
                    results[f'pred_{target}'] = [np.nan] * len(cand_ids)
                    continue
                # 否则过滤掉出问题的 support
                support_items = [it for si, it in enumerate(support_items) if si not in {r[0] for r in bad_supports}]

            # 构建 numpy arrays（注意：support_items 至少有一个有效项）
            try:
                s_node = np.stack([it['graph']['node_feats'] for it in support_items])
                s_adj = np.stack([it['graph']['adj'] for it in support_items])
                s_mask = np.stack([it['graph']['mask'] for it in support_items])
                s_labels = np.array([it['y'] for it in support_items], dtype=np.float32)
            except Exception as e:
                print(f"[predict_candidates][ERROR] target {target} 在 np.stack 支持集时出错: {e}")
                results[f'pred_{target}'] = [np.nan] * len(cand_ids)
                continue

            # 打印 support shapes 供 debug
            print(f"[predict_candidates] target {target}: support shape node={s_node.shape}, adj={s_adj.shape}, mask={s_mask.shape}, labels={s_labels.shape}")

            # 转为 torch tensors 放到 device
            s_graphs_dev = {
                'node_feats': torch.tensor(s_node, dtype=torch.float32, device=device),
                'adj': torch.tensor(s_adj, dtype=torch.float32, device=device),
                'mask': torch.tensor(s_mask, dtype=torch.float32, device=device)
            }
            s_labels_dev = torch.tensor(s_labels, dtype=torch.float32, device=device)

            # 对每个 candidate 进行预测
            preds_list = []
            for idx_cand, g in enumerate(cand_graphs):
                if g is None:
                    preds_list.append(np.nan)
                    continue
                q_graphs = {
                    'node_feats': np.expand_dims(g['node_feats'], axis=0),
                    'adj': np.expand_dims(g['adj'], axis=0),
                    'mask': np.expand_dims(g['mask'], axis=0)
                }
                q_graphs_dev = {k: torch.tensor(v, dtype=torch.float32, device=device) for k,v in q_graphs.items()}
                try:
                    with torch.no_grad():
                        y_pred, a = model(q_graphs_dev, s_graphs_dev, s_labels_dev)
                    preds_list.append(float(y_pred.cpu().numpy()[0]))
                except Exception as e:
                    # 捕获单个候选预测异常，记录 NaN 并打印 debug 信息
                    print(f"[predict_candidates][WARN] target {target} candidate idx {idx_cand} predict error: {e}")
                    preds_list.append(np.nan)

            results[f'pred_{target}'] = preds_list

        except Exception as e:
            # 捕获 target 级别的异常，避免整个流程退出
            print(f"[predict_candidates][ERROR] 在处理 target {target} 时发生异常: {e}")
            # 打印部分支持集以供定位
            try:
                print("support_items sample (first 5):")
                for it in support_items[:5]:
                    print("  molecule_id:", it.get('molecule_id'), "graph_keys:", list(it.get('graph', {}).keys()))
            except Exception:
                pass
            results[f'pred_{target}'] = [np.nan] * len(cand_ids)

    out_df = pd.DataFrame(results)
    return out_df


# -----------------------------
# IX. 主流程：读取文件 -> 构建 tasks_data -> 训练 -> 预测 -> 保存
# -----------------------------
def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # 读取 molecule.smi
    if not os.path.exists(MOL_FILE):
        raise FileNotFoundError(f"找不到 molecule 文件：{MOL_FILE}")
    print("读取 molecule.smi ...")
    mol_df = read_molecule_smi(MOL_FILE)
    print(f"读取到 {len(mol_df)} 条 molecule 记录，示例：")
    print(mol_df.head())

    # 读取 activity 文件 (molecule_id, target_id, pIC50)
    if not os.path.exists(ACT_FILE):
        raise FileNotFoundError(f"找不到 activity 文件：{ACT_FILE}")
    print("读取 activity_train.csv ...")
    act_df = pd.read_csv(ACT_FILE, dtype=str)
    # 标准化列名：尝试找到 molecule_id, target_id, pIC50
    cols = [c.lower() for c in act_df.columns]
    # molecule_id 列
    mid_col = None
    for c in act_df.columns:
        if c.lower() == 'molecule_id':
            mid_col = c; break
    if mid_col is None:
        # 尝试类似列名
        for c in act_df.columns:
            if 'mol' in c.lower() and 'id' in c.lower():
                mid_col = c; break
    if mid_col is None:
        raise ValueError(f"activity 文件缺少 molecule_id 列, 当前列: {act_df.columns.tolist()}")
    # target_id 列
    target_col = None
    for c in act_df.columns:
        if c.lower() in ['target_id','assay','protein','task']:
            target_col = c; break
    if target_col is None:
        raise ValueError(f"activity 文件缺少 target_id 列, 当前列: {act_df.columns.tolist()}")
    # pIC50 列（忽略大小写）
    pcol = None
    for c in act_df.columns:
        if c.lower() == 'pic50' or 'pic50' in c.lower():
            pcol = c; break
    if pcol is None:
        for c in act_df.columns:
            if 'value' in c.lower() or 'activity' in c.lower():
                pcol = c; break
    if pcol is None:
        raise ValueError(f"activity 文件缺少 pIC50 列 (或 'value'/'activity'), 当前列: {act_df.columns.tolist()}")

    # 构建标准 activity DataFrame
    act_df2 = act_df[[mid_col, target_col, pcol]].copy()
    act_df2.columns = ['molecule_id','target_id','pIC50']
    # 打印几行检查
    print("activity_train 示例：")
    print(act_df2.head())

    # 构造 tasks_data（会做 smiles 查找并 featurize）
    print("开始构造 tasks_data（featurize 分子）——可能耗时")
    tasks_data = build_tasks_data(mol_df, act_df2, max_nodes=MAX_NODES)
    print("各 target_id 样本数：")
    for t, lst in tasks_data.items():
        print(f"  {t}: {len(lst)}")

    # 若任务数量太少/某些 target_id 样本过少，提示
    for t, lst in tasks_data.items():
        if len(lst) < 4:
            print(f"[警告] target_id {t} 样本数少于 4: {len(lst)} （可能影响训练稳定性）")

    # node feature dim
    sample_task = next(iter(tasks_data.values()))
    node_feat_dim = sample_task[0]['graph']['node_feats'].shape[1]
    print("node_feat_dim =", node_feat_dim)

    # 训练模型（episodic）
    print("开始 episodic 训练（IterRef + GCN） ...")
    model = train_iterref_gnn(tasks_data, node_feat_dim, device=DEVICE,
                              n_steps=N_STEPS, support_size=SUPPORT_SIZE, query_size=QUERY_SIZE,
                              emb_dim=EMB_DIM, gcn_hidden=GCN_HIDDEN, iter_depth=ITER_DEPTH,
                              lr=LR, weight_decay=WEIGHT_DECAY, print_every=200)
    model_path = os.path.join(RESULT_DIR, "iterref_gnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print("训练完成，模型保存为:", model_path)

    # candidate 预测（如果有 candidate.csv）
    if os.path.exists(CAND_FILE):
        print("开始对 candidate 做预测 ...")
        try:
            cand_df = pd.read_csv(CAND_FILE, dtype=str)
            # 规范 candidate 列名： molecule_id, smiles
            if 'molecule_id' not in cand_df.columns or 'smiles' not in cand_df.columns:
                # 若只有 smiles 列
                if cand_df.shape[1] == 1:
                    col = cand_df.columns[0]
                    cand_df = cand_df.rename(columns={col:'smiles'})
                    cand_df['molecule_id'] = ['CAND_'+str(i) for i in range(len(cand_df))]
                else:
                    # 尝试猜测
                    cols_low = [c.lower() for c in cand_df.columns]
                    smi_col = None
                    for c in cand_df.columns:
                        if 'smiles' in c.lower() or 'smi' in c.lower():
                            smi_col = c; break
                    if smi_col is None:
                        raise ValueError("candidate.csv 中无法识别 smiles 列，请确保包含 smiles 列")
                    cand_df = cand_df.rename(columns={smi_col:'smiles'})
                    if 'molecule_id' not in cand_df.columns:
                        cand_df['molecule_id'] = ['CAND_'+str(i) for i in range(len(cand_df))]
            out_df = predict_candidates(model, tasks_data, cand_df, max_nodes=MAX_NODES, device=DEVICE)
            out_path = os.path.join(RESULT_DIR, "candidate_preds_iterref_gnn.csv")
            out_df.to_csv(out_path, index=False)
            print("candidate 预测结果保存为:", out_path)
        except Exception as e:
            print("candidate 预测出错:", e)
    else:
        print("未找到 candidate.csv，跳过预测步骤")

if __name__ == "__main__":
    main()
