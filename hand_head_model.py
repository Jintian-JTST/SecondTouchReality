
# hand_head_model.py
# 一个很小的“AI 头部（fusion head）”模型：
# 输入：s_w, s_l, curl, Zw, Zl   （Zw/Zl 为宽/长通道给出的深度估计，单位 cm）
# 输出：Z_pred（融合后的深度，cm）以及 alpha（对 Zw 的权重，0~1）
#
# 设计思路：
# 1) 轻量化：用一个极小的感知器（MLP），参数 < 500，可在 CPU 上几毫秒内前向。
# 2) “可解释”：核心仍是双通道加权：Z = alpha*Zw + (1-alpha)*Zl；alpha 由特征
#    [s_w, s_l, curl, Zw, Zl, Zw-Zl, |Zw-Zl|] 通过一个很小的网络输出；
# 3) 可无缝替换 hand_depth_unified_min.py 里基于规则的融合，训练数据可来自
#    你自己采的 (Zw,Zl,s_w,s_l,curl,z_gt) CSV。

from __future__ import annotations
import json, math, os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

Z_MAX_CM_DEFAULT = 200.0

@dataclass
class TrainConfig:
    lr: float = 1e-2
    weight_decay: float = 1e-4   # L2 正则
    epochs: int = 400
    val_split: float = 0.2
    z_max_cm: float = Z_MAX_CM_DEFAULT
    seed: int = 42

class FusionHead(nn.Module):
    """极小的 gating MLP。输出 alpha∈(0,1) 和校正 residual。"""
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(7, hidden)         # [s_w, s_l, curl, Zw, Zl, d, ad]
        self.act = nn.ReLU()
        self.alpha = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        # 轻量的残差校正，避免系统性偏差（例如相机 FoV 差异带来的缩放/偏移）
        self.residual = nn.Linear(hidden, 1)     # 输出可正可负的小修正

    def forward(self, s_w, s_l, curl, Zw, Zl):
        d  = Zw - Zl
        ad = torch.abs(d)
        x = torch.stack([s_w, s_l, curl, Zw, Zl, d, ad], dim=-1)
        h = self.act(self.fc1(x))
        a = self.alpha(h).squeeze(-1)        # ∈(0,1)
        r = self.residual(h).squeeze(-1)     # 任意实数（一般很小）
        z = a*Zw + (1.0 - a)*Zl + r
        return z, a

def _split_train_val(N: int, val_split: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(N * val_split))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return tr_idx, val_idx

def load_csv(path: str) -> Dict[str, np.ndarray]:
    """读取 CSV（含表头）。需要列：s_w,s_l,curl,Zw,Zl,z_gt（单位 cm）。
    允许存在多余列。"""
    import csv
    cols = {k: [] for k in ['s_w','s_l','curl','Zw','Zl','z_gt']}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cols['s_w' ] .append(float(row['s_w'] ))
                cols['s_l' ] .append(float(row['s_l'] ))
                cols['curl'] .append(float(row['curl']))
                cols['Zw'  ] .append(float(row['Zw'  ]))
                cols['Zl'  ] .append(float(row['Zl'  ]))
                cols['z_gt'] .append(float(row['z_gt']))
            except KeyError as e:
                raise KeyError(f"CSV 缺少必须列: {e}. 需要列: s_w,s_l,curl,Zw,Zl,z_gt") from e
    out = {k: np.asarray(v, dtype=np.float32) for k, v in cols.items()}
    return out

def train_fusion_head(csv_path: str, cfg: TrainConfig = TrainConfig()) -> Tuple[FusionHead, Dict[str, float]]:
    data = load_csv(csv_path)
    N = len(data['z_gt'])
    tr_idx, val_idx = _split_train_val(N, cfg.val_split, cfg.seed)

    def _to_torch(idx):
        t = lambda k: torch.tensor(data[k][idx], dtype=torch.float32)
        return t('s_w'), t('s_l'), t('curl'), t('Zw'), t('Zl'), t('z_gt')

    s_w_tr, s_l_tr, curl_tr, Zw_tr, Zl_tr, z_tr = _to_torch(tr_idx)
    s_w_val, s_l_val, curl_val, Zw_val, Zl_val, z_val = _to_torch(val_idx)

    model = FusionHead(hidden=12)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        z_hat, a = model(s_w_tr, s_l_tr, curl_tr, Zw_tr, Zl_tr)
        # 限幅：物理上不应为负，且不应超过 z_max_cm
        z_hat = torch.clamp(z_hat, 0.0, cfg.z_max_cm)
        loss_mse = mse(z_hat, z_tr)
        # 轻微正则：抑制过大残差修正（靠近 0）
        loss_reg = 1e-4 * torch.mean(model.residual.weight**2)
        loss = loss_mse + loss_reg
        loss.backward()
        opt.step()

        # 验证
        model.eval()
        with torch.no_grad():
            z_hat_v, a_v = model(s_w_val, s_l_val, curl_val, Zw_val, Zl_val)
            z_hat_v = torch.clamp(z_hat_v, 0.0, cfg.z_max_cm)
            val_mse = mse(z_hat_v, z_val).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

        if (ep+1) % max(1, cfg.epochs//10) == 0:
            print(f"[ep {ep+1:4d}] train_mse={loss_mse.item():.4f}  val_mse={val_mse:.4f}  alpha~[{a.min().item():.2f},{a.max().item():.2f}] ")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 汇总指标
    model.eval()
    with torch.no_grad():
        z_hat_tr, _ = model(s_w_tr, s_l_tr, curl_tr, Zw_tr, Zl_tr)
        z_hat_tr = torch.clamp(z_hat_tr, 0.0, cfg.z_max_cm)
        z_hat_v, _ = model(s_w_val, s_l_val, curl_val, Zw_val, Zl_val)
        z_hat_v = torch.clamp(z_hat_v, 0.0, cfg.z_max_cm)
        train_rmse = math.sqrt(float(torch.mean((z_hat_tr - z_tr)**2)))
        val_rmse   = math.sqrt(float(torch.mean((z_hat_v  - z_val)**2)))

    metrics = {
        'train_rmse_cm': float(train_rmse),
        'val_rmse_cm': float(val_rmse),
        'val_mse': float(best_val),
        'n_train': int(len(tr_idx)),
        'n_val': int(len(val_idx)),
    }
    return model, metrics

def save_model(model: FusionHead, path: str, extra: Optional[Dict]=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'state_dict': {k: v.cpu().detach().tolist() for k, v in model.state_dict().items()},
        'extra': extra or {},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)

def load_model(path: str) -> FusionHead:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    model = FusionHead(hidden=12)
    state = {k: torch.tensor(v) for k, v in payload['state_dict'].items()}
    model.load_state_dict(state)
    return model

def predict(model: FusionHead, s_w, s_l, curl, Zw, Zl, z_max_cm: float = Z_MAX_CM_DEFAULT):
    """批量或标量预测。参数可为 Python float 或 numpy.ndarray。"""
    to_t = lambda x: torch.tensor(x, dtype=torch.float32)
    z, a = model(to_t(s_w), to_t(s_l), to_t(curl), to_t(Zw), to_t(Zl))
    z = torch.clamp(z, 0.0, z_max_cm)
    return z.detach().cpu().numpy(), a.detach().cpu().numpy()

# 方便集成：一个纯函数接口
_GLOBAL_MODEL: Optional[FusionHead] = None
def load_global(path: str):
    global _GLOBAL_MODEL
    _GLOBAL_MODEL = load_model(path)

def fuse_z_learned(s_w: float, s_l: float, curl: float, Zw: float, Zl: float, z_max_cm: float = Z_MAX_CM_DEFAULT) -> Tuple[float,float]:
    """返回 (Z_pred, alpha)。无需关心 Torch，在推理时这是个普通函数。"""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        # 如果未加载，默认使用一个未训练的小模型（不建议），相当于 alpha≈0.5
        _GLOBAL_MODEL = FusionHead(hidden=12)
    z, a = predict(_GLOBAL_MODEL, s_w, s_l, curl, Zw, Zl, z_max_cm=z_max_cm)
    return float(z), float(a)
