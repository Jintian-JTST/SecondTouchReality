# hand_depth_ai_smoother.py
# -*- coding: utf-8 -*-
"""
目的：用“几何双通道测距”给出每帧 Z_meas，再用【数据自适应卡尔曼 + 极小 MLP】在线平滑与预测，得到 Z_AI。
特点：
- 仍然单目、仍用 MediaPipe 21 点；不需要外部深度相机；
- MLP 每帧根据特征自适应输出 Q_t、R_t（过程/观测噪声），卡尔曼按它们融合；
- 在线无监督训练（基于创新 likelihood），学“何时信预测、何时信测量”，避免 Z 跳变且保持响应。
"""

import cv2
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 参数区
# -----------------------------
WIN_NAME = "Hand Z (Geom) + Z_AI (Neural-Kalman)"
MAX_NUM_HANDS = 1
DRAW_LANDMARKS = True

S_VIS_TH   = 0.25      # 可见性下限
AGREE_FRAC = 0.35      # 双通道相对一致阈值
EMA_ALPHA  = 0.20      # 几何测距本身做一点点 EMA（AI 再做主要平滑）
Z_MAX_CM   = 200.0

CALIB_SAMPLES = 30
DEFAULT_D_CM  = 40
DEFAULT_W_CM  = 9
DEFAULT_L_CM  = 10

# 神经-卡尔曼
BASE_Q_SCALE = 5.0     # 过程噪声标度（越大=允许更快变化）
BASE_R_SCALE = 25.0    # 观测噪声标度（越大=更不信几何测距）
LAMBDA_REG   = 1e-3    # 对 Q,R 的轻正则

# -----------------------------
# 工具
# -----------------------------
mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

def draw_text(img, txt, x, y, color=(255,255,255), scale=0.6):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def l2(p, q):
    p = np.asarray(p, np.float32); q = np.asarray(q, np.float32)
    return float(np.linalg.norm(p - q))

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def fmt1(x):
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.1f}"

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class Measurements:
    w_px: Optional[float]
    l_px: Optional[float]
    s_w: float
    s_l: float
    p0: Tuple[int,int]
    p5: Tuple[int,int]
    p9: Tuple[int,int]
    p12: Tuple[int,int]
    p17: Tuple[int,int]

@dataclass
class CalibState:
    collecting: bool = False
    q_w: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_l: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    f_w: Optional[float] = None
    f_l: Optional[float] = None

@dataclass
class RuntimeState:
    Z_meas_ema: Optional[float] = None
    fps: float = 0.0

# -----------------------------
# 模块 3：从21点提两条线 + 可见性 + 几何方向
# -----------------------------
def extract_measurements(hand_landmarks, img_w, img_h):
    """返回 (Measurements, pts_3d, s_dir, s_norm) 或 None"""
    if not hand_landmarks:
        return None
    lm = hand_landmarks[0].landmark
    pts_px = np.array([(int(l.x * img_w), int(l.y * img_h)) for l in lm], dtype=np.int32)
    pts_3d = np.array([(l.x, l.y, l.z) for l in lm], dtype=np.float32)

    p0, p5, p9, p12, p17 = pts_px[0], pts_px[5], pts_px[9], pts_px[12], pts_px[17]

    w_px = l2(p5, p17)                         # 掌宽
    l_px = l2(p0, p9) + l2(p9, p12)           # 掌长(分段)

    def s_vis(v):
        n3 = np.linalg.norm(v) + 1e-6
        n2 = np.linalg.norm(v[:2])
        return float(n2 / n3)

    v_w_3d = pts_3d[17] - pts_3d[5]
    v_l_3d = (pts_3d[9] - pts_3d[0]) + (pts_3d[12] - pts_3d[9])
    s_w = clip(s_vis(v_w_3d), 0.0, 1.0)
    s_l = clip(s_vis(v_l_3d), 0.0, 1.0)

    # 几何方向特征：手指轴与掌面法向朝向相机的分量（用于 AI 估计 Q/R）
    V  = pts_3d[12] - pts_3d[0]
    U  = pts_3d[5]  - pts_3d[0]
    Wv = pts_3d[17] - pts_3d[0]
    Vn = V / (np.linalg.norm(V) + 1e-6)
    n  = np.cross(U, Wv)
    nn = n / (np.linalg.norm(n) + 1e-6)
    s_dir  = -float(Vn[2])   # 越大越朝向相机
    s_norm = -float(nn[2])

    meas = Measurements(w_px=w_px, l_px=l_px, s_w=s_w, s_l=s_l,
                        p0=tuple(p0), p5=tuple(p5), p9=tuple(p9),
                        p12=tuple(p12), p17=tuple(p17))
    return meas, pts_3d, s_dir, s_norm

# -----------------------------
# 模块 4：标定
# -----------------------------
def start_collect(calib: CalibState):
    calib.collecting = True
    calib.q_w.clear(); calib.q_l.clear()

def finish_collect_and_compute_f(calib: CalibState, D_cm: float, W_cm: float, L_cm: float):
    if len(calib.q_w) == 0 or len(calib.q_l) == 0:
        calib.collecting = False
        return False
    w_med = float(np.median(list(calib.q_w)))
    l_med = float(np.median(list(calib.q_l)))
    if W_cm <= 0 or L_cm <= 0 or D_cm <= 0:
        calib.collecting = False
        return False
    calib.f_w = (D_cm * w_med) / W_cm
    calib.f_l = (D_cm * l_med) / L_cm
    calib.collecting = False
    return True

# -----------------------------
# 模块 5：几何两通道估距 + 融合
# -----------------------------
def estimate_Z_geom(meas: Measurements, f_w, f_l, W_cm, L_cm):
    Z_w = None; Z_l = None
    if f_w and meas.w_px and meas.w_px > 1e-3:
        Z_w = f_w * (W_cm / meas.w_px)
    if f_l and meas.l_px and meas.l_px > 1e-3:
        Z_l = f_l * (L_cm / meas.l_px)
    Z_w = None if Z_w is None else clip(Z_w, 0.0, Z_MAX_CM)
    Z_l = None if Z_l is None else clip(Z_l, 0.0, Z_MAX_CM)
    Z = None
    if Z_w is not None and Z_l is not None:
        rel = abs(Z_w - Z_l) / max(1e-6, 0.5 * (Z_w + Z_l))
        if rel < AGREE_FRAC and (meas.s_w > S_VIS_TH and meas.s_l > S_VIS_TH):
            w_w, w_l = meas.s_w, meas.s_l
            Z = (w_w * Z_w + w_l * Z_l) / (w_w + w_l + 1e-6)
        else:
            Z = Z_w if meas.s_w >= meas.s_l else Z_l
    elif Z_w is not None:
        Z = Z_w
    elif Z_l is not None:
        Z = Z_l
    return Z, Z_w, Z_l

# -----------------------------
# 模块 6：数据自适应卡尔曼（Q,R 由极小 MLP 输出）
# -----------------------------
class QRNet(nn.Module):
    def __init__(self, in_dim=10, hid=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 2)  # [q_hat, r_hat]
        )

    def forward(self, x):
        out = self.net(x)
        q_hat, r_hat = out[..., 0], out[..., 1]
        # softplus 转正，并给个基准尺度
        q = torch.nn.functional.softplus(q_hat) * BASE_Q_SCALE + 1e-6
        r = torch.nn.functional.softplus(r_hat) * BASE_R_SCALE + 1e-6
        return q, r

class NeuralKalman1D:
    """2D 状态[x=Z, v] 常速模型；Q,R 由 QRNet 逐帧自适应；在线极小步训练。"""
    def __init__(self, feat_dim=10, lr=1e-3, device="cpu"):
        self.device = device
        self.qr = QRNet(in_dim=feat_dim).to(device)
        self.opt = optim.Adam(self.qr.parameters(), lr=lr)
        # 状态与协方差
        self.x = None  # torch.tensor([Z, V])
        self.P = None  # 2x2
        self.training_on = True

    @staticmethod
    def _Q_cv(q, dt, device):
        dt = float(dt)
        a = 0.25 * dt**4 * q
        b = 0.5  * dt**3 * q
        c =        dt**2 * q
        Q = torch.tensor([[a, b],[b, c]], dtype=torch.float32, device=device)
        return Q

    def step(self, z_meas, feats_np, dt):
        """一次滤波/训练。输入：z_meas(float 或 None)、feats_np(np.array)"""
        # 构造 torch 输入
        feats = torch.tensor(feats_np, dtype=torch.float32, device=self.device).view(1, -1)
        q, r = self.qr(feats)  # 标量（batch=1）

        # 初始化
        if self.x is None:
            if z_meas is None:
                return None  # 等待首帧观测
            self.x = torch.tensor([float(z_meas), 0.0], dtype=torch.float32, device=self.device)
            self.P = torch.eye(2, dtype=torch.float32, device=self.device) * 10.0

        I = torch.eye(2, dtype=torch.float32, device=self.device)

        # 预测
        dt_f = float(max(1e-3, dt))
        F = torch.tensor([[1.0, dt_f],[0.0, 1.0]], dtype=torch.float32, device=self.device)
        x_pred = F @ self.x.detach()                               # detach: 不展开回历史
        P_pred = F @ self.P.detach() @ F.T + self._Q_cv(q, dt_f, self.device)

        # 若无观测，只做预测前推
        if z_meas is None:
            self.x, self.P = x_pred, P_pred
            return (float(x_pred[0].item()), float(x_pred[1].item()), float(q.item()), None, None)

        # 更新
        H = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=self.device)  # 观测 Z
        z_t = torch.tensor([[float(z_meas)]], dtype=torch.float32, device=self.device)
        y   = z_t - (H @ x_pred).view(1,1)                      # 创新 r_t
        S   = (H @ P_pred @ H.T) + r.view(1,1)                  # 创新方差
        K   = (P_pred @ H.T) / S                                # 卡尔曼增益 (2x1)

        x_new = x_pred + (K @ y).view(2)
        P_new = (I - K @ H) @ P_pred

        # 训练目标：负对数似然 ~ 0.5*(y^2/S + log S)，并加一点对 Q,R 的正则，避免走极端
        loss = 0.5 * (y.pow(2) / S + torch.log(S)) + LAMBDA_REG * (q + r)
        if self.training_on:
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qr.parameters(), 5.0)
            self.opt.step()

        self.x, self.P = x_new.detach(), P_new.detach()

        # 返回数值
        k_gain = float(K[0,0].item())      # 位置维的增益（越大越信测量）
        return (float(self.x[0].item()),   # Z_AI
                float(self.x[1].item()),   # V_AI
                float(q.item()),           # Q_t (过程噪声标度)
                float(r.item()),           # R_t (观测噪声标度)
                k_gain)                    # 卡尔曼增益

# -----------------------------
# Trackbar
# -----------------------------
def _noop(v): pass
def create_trackbars():
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)
    cv2.createTrackbar("D_cm", WIN_NAME, DEFAULT_D_CM, 200, _noop)
    cv2.createTrackbar("W_cm", WIN_NAME, DEFAULT_W_CM, 30, _noop)
    cv2.createTrackbar("L_cm", WIN_NAME, DEFAULT_L_CM, 35, _noop)
    cv2.createTrackbar("EMA_x100", WIN_NAME, int(EMA_ALPHA * 100), 100, _noop)

def get_D_cm(): return max(1, cv2.getTrackbarPos("D_cm", WIN_NAME))
def get_W_cm(): return max(1, cv2.getTrackbarPos("W_cm", WIN_NAME))
def get_L_cm(): return max(1, cv2.getTrackbarPos("L_cm", WIN_NAME))
def get_EMA():  return clip(cv2.getTrackbarPos("EMA_x100", WIN_NAME) / 100.0, 0.0, 1.0)

# -----------------------------
# 主循环
# -----------------------------
def main():
    create_trackbars()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    calib = CalibState()
    state = RuntimeState()

    device = "cpu"
    nk = NeuralKalman1D(feat_dim=10, lr=1e-3, device=device)

    last_t = time.time()
    prev_wpx = None
    prev_lpx = None

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            meas_pack = extract_measurements(res.multi_hand_landmarks, w, h) if res.multi_hand_landmarks else None
            meas, pts_3d, s_dir, s_norm = (meas_pack if meas_pack else (None, None, None, None))

            # 画 landmarks
            if DRAW_LANDMARKS and res.multi_hand_landmarks:
                for hlm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # 标定采样
            if calib.collecting:
                draw_text(frame, "[标定采样] 请保持手稳定在设定 D_cm 位置", 10, 26, (180,230,255))
                if meas is not None:
                    if meas.w_px and meas.w_px > 1: calib.q_w.append(meas.w_px)
                    if meas.l_px and meas.l_px > 1: calib.q_l.append(meas.l_px)
                draw_text(frame, f"进度: w {len(calib.q_w)}/{CALIB_SAMPLES} | l {len(calib.q_l)}/{CALIB_SAMPLES}", 10, 50, (180,230,255))
                if len(calib.q_w) >= CALIB_SAMPLES and len(calib.q_l) >= CALIB_SAMPLES:
                    if finish_collect_and_compute_f(calib, get_D_cm(), get_W_cm(), get_L_cm()):
                        draw_text(frame, f"[标定完成] f_w={calib.f_w:.1f}, f_l={calib.f_l:.1f}", 10, 74, (180,255,200))
                    else:
                        draw_text(frame, "[标定失败] 检查 D/W/L", 10, 74, (0,0,255))

            # 几何测距
            Z_meas = None; Zw = None; Zl = None
            if meas is not None and (calib.f_w or calib.f_l):
                Z_meas, Zw, Zl = estimate_Z_geom(meas, calib.f_w, calib.f_l, get_W_cm(), get_L_cm())
                if Z_meas is not None:
                    alpha = get_EMA()
                    state.Z_meas_ema = Z_meas if state.Z_meas_ema is None else (alpha * Z_meas + (1 - alpha) * state.Z_meas_ema)

            # 时间步长
            now = time.time()
            dt = max(1e-3, now - last_t); last_t = now

            # 特征（供 QRNet）
            # 归一化像素与可见性、朝向、以及帧间变化率
            if meas is not None:
                w_hat = (meas.w_px or 0.0) / 60.0
                l_hat = (meas.l_px or 0.0) / 100.0
                dw = 0.0 if prev_wpx is None else (meas.w_px - prev_wpx) / dt
                dl = 0.0 if prev_lpx is None else (meas.l_px - prev_lpx) / dt
                prev_wpx, prev_lpx = meas.w_px, meas.l_px
                dw_hat = abs(dw) / 200.0     # 粗略归一
                dl_hat = abs(dl) / 200.0
                feats = np.array([
                    float(meas.s_w or 0.0), float(meas.s_l or 0.0),
                    w_hat, l_hat,
                    float(s_dir or 0.0), float(s_norm or 0.0),
                    dw_hat, dl_hat,
                    1.0,                      # bias-like
                    0.0 if Z_meas is None else min(1.0, Z_meas / 100.0)  # 距离尺度线索
                ], dtype=np.float32)
            else:
                feats = np.zeros(10, dtype=np.float32)

            # 神经-卡尔曼一步（若无观测，就用预测）
            nk_out = nk.step(Z_meas, feats, dt)
            Z_ai = V_ai = Q_t = R_t = Kz = None
            if nk_out is not None:
                Z_ai, V_ai, Q_t, R_t, Kz = nk_out

            # FPS
            fps = 0.9 * state.fps + 0.1 * (1.0/dt) if state.fps > 0 else (1.0/dt)
            state.fps = fps

            # HUD
            y = 24
            draw_text(frame, f"FPS {fps:5.1f}", 10, y); y += 24
            if calib.f_w or calib.f_l:
                draw_text(frame, f"已标定  f_w={fmt1(calib.f_w)}  f_l={fmt1(calib.f_l)}", 10, y, (200,255,200)); y += 24
            else:
                draw_text(frame, f"未标定 - 调 D/W/L 后按 'c' 采样", 10, y, (80,230,255)); y += 24

            if meas is not None:
                draw_text(frame, f"w_px={meas.w_px:.1f} s_w={meas.s_w:.2f} | l_px={meas.l_px:.1f} s_l={meas.s_l:.2f}", 10, y); y += 24
            if Z_meas is not None:
                draw_text(frame, f"Z_meas(cm) fused={fmt1(Z_meas)}  Zw={fmt1(Zw)}  Zl={fmt1(Zl)}", 10, y, (255,220,180)); y += 24
                if state.Z_meas_ema is not None:
                    draw_text(frame, f"Z_meas_EMA(cm)={fmt1(state.Z_meas_ema)}", 10, y, (255,220,180)); y += 24
            if Z_ai is not None:
                draw_text(frame, f"Z_AI(cm)={fmt1(Z_ai)}  V_AI(cm/s)={fmt1(V_ai)}  K={fmt1(Kz)}", 10, y, (180,255,200)); y += 24
                draw_text(frame, f"AI Q={fmt1(Q_t)}  R={fmt1(R_t)}  Train={'ON' if nk.training_on else 'OFF'}", 10, y, (200,200,255)); y += 24

            # 位置标注：在手腕处画圆点，颜色表示“AI更信测量/预测”
            if meas is not None:
                cx, cy = meas.p0
                if Kz is None:  # 未进入更新态
                    color = (128,128,128)
                else:
                    # K 近1 => 信测量(绿)；近0 => 信预测(蓝)
                    k = clip(Kz, 0.0, 1.0)
                    color = (int(255*(1-k)), int(255*k), 255 - int(255*k))
                cv2.circle(frame, (cx, cy), 18, (0,0,0), 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 18, color, -1, cv2.LINE_AA)

            draw_text(frame, "键位: c=标定  r=重置  t=切换训练  q=退出", 10, h-12, (230,230,230))

            cv2.imshow(WIN_NAME, frame)

            # 键盘
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('c'):
                start_collect(calib)
            elif k == ord('r'):
                calib = CalibState()
                state = RuntimeState()
                nk = NeuralKalman1D(feat_dim=10, lr=1e-3, device=device)
                prev_wpx = prev_lpx = None
            elif k == ord('t'):
                nk.training_on = not nk.training_on

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
