# final_hand_orient_depth_screen.py
# -*- coding: utf-8 -*-
"""
屏幕显示版：MediaPipe手部 -> 掌宽/掌长双通道估距 + 稳健标定 + EMA -> 纵向朝向(几何+在线LR) -> HUD叠加
键位:
  c : 标定(自动采样多帧取中位数; 用 D/W/L 求 f_w, f_l)
  r : 重置标定与状态
  g : 在线学习-记为“正对相机”(正样本)
  h : 在线学习-记为“非正对”(负样本)
  q : 退出
"""

import cv2
import time
import math
from typing import Optional, Tuple

from dataclasses import dataclass, field
from collections import deque

import numpy as np
import mediapipe as mp

# -----------------------------
# 参数区
# -----------------------------
WIN_NAME = "Hand Z + Facing (Screen Only)"
MAX_NUM_HANDS = 1
DRAW_LANDMARKS = True

# 估距融合/质量门限
S_VIS_TH = 0.25       # s_w/s_l 下限（0~1）
AGREE_FRAC = 0.35     # 双通道相对差异阈值(小于则求加权平均)
EMA_ALPHA = 0.25      # EMA 初值；可实时用轨迹条调
Z_MAX_CM = 200.0      # 距离限幅

# 标定采样
CALIB_SAMPLES = 30
DEFAULT_D_CM = 40
DEFAULT_W_CM = 9
DEFAULT_L_CM = 10

# -----------------------------
# 工具
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_text(img, txt, x, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def l2(p, q):
    p = np.asarray(p, np.float32); q = np.asarray(q, np.float32)
    return float(np.linalg.norm(p - q))

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def fmt1(x):
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.1f}"

def lerp(a, b, t):  # 线性插值
    return a + (b - a) * t

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class Measurements:
    w_px: Optional[float]
    l_px: Optional[float]
    s_w: float
    s_l: float
    p0: Tuple[int, int]
    p5: Tuple[int, int]
    p9: Tuple[int, int]
    p12: Tuple[int, int]
    p17: Tuple[int, int]

# 其余 import 保持不变...

@dataclass
class CalibState:
    collecting: bool = False
    q_w: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_l: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    f_w: Optional[float] = None
    f_l: Optional[float] = None


@dataclass
class RuntimeState:
    Z_ema_cm: Optional[float] = None
    fps: float = 0.0

# -----------------------------
# 模块 3：从21点提两条线 + 可见性
# -----------------------------
def extract_measurements(hand_landmarks, img_w, img_h):
    """返回 (Measurements, pts_3d) 或 None"""
    if not hand_landmarks:
        return None

    lm = hand_landmarks[0].landmark
    pts_px = np.array([(int(l.x * img_w), int(l.y * img_h)) for l in lm], dtype=np.int32)
    pts_3d = np.array([(l.x, l.y, l.z) for l in lm], dtype=np.float32)

    p0  = pts_px[0]   # wrist
    p5  = pts_px[5]   # index_mcp
    p9  = pts_px[9]   # middle_mcp
    p12 = pts_px[12]  # middle_tip
    p17 = pts_px[17]  # pinky_mcp

    # 两条测量线
    w_px = l2(p5, p17)                      # 掌宽
    l_px = l2(p0, p9) + l2(p9, p12)        # 掌长(分段)

    # 可见性：3D向量在像平面投影比值，抑制朝外压扁
    def s_vis(v):
        n3 = np.linalg.norm(v) + 1e-6
        n2 = np.linalg.norm(v[:2])
        return float(n2 / n3)

    v_w_3d = pts_3d[17] - pts_3d[5]
    v_l_3d = (pts_3d[9] - pts_3d[0]) + (pts_3d[12] - pts_3d[9])
    s_w = clip(s_vis(v_w_3d), 0.0, 1.0)
    s_l = clip(s_vis(v_l_3d), 0.0, 1.0)

    meas = Measurements(w_px=w_px, l_px=l_px, s_w=s_w, s_l=s_l,
                        p0=tuple(p0), p5=tuple(p5), p9=tuple(p9),
                        p12=tuple(p12), p17=tuple(p17))
    return meas, pts_3d

# -----------------------------
# 模块 4：标定（中位数稳住）
# -----------------------------
def start_collect(calib: CalibState):
    calib.collecting = True
    calib.q_w.clear(); calib.q_l.clear()

def finish_collect_and_compute_f(calib: CalibState, D_cm: float, W_cm: float, L_cm: float):
    """用 D/W/L 求 f_w, f_l; 公式: Z = f * (Real / pixel) => f = Z * pixel / Real"""
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
# 模块 5：两通道估距 + 融合
# -----------------------------
def estimate_Z_cm(meas: Measurements, f_w: Optional[float], f_l: Optional[float], W_cm: float, L_cm: float):
    """返回 (Z_fused, Z_w, Z_l)，单位 cm"""
    Z_w = None; Z_l = None
    if f_w and meas.w_px and meas.w_px > 1e-3:
        Z_w = f_w * (W_cm / meas.w_px)
    if f_l and meas.l_px and meas.l_px > 1e-3:
        Z_l = f_l * (L_cm / meas.l_px)

    Z_w = None if Z_w is None else clip(Z_w, 0.0, Z_MAX_CM)
    Z_l = None if Z_l is None else clip(Z_l, 0.0, Z_MAX_CM)

    Z = None
    if Z_w is not None and Z_l is not None:
        rel = abs(Z_w - Z_l) / max(1e-6, (Z_w + Z_l) * 0.5)
        if rel < AGREE_FRAC and (meas.s_w > S_VIS_TH and meas.s_l > S_VIS_TH):
            w_w = meas.s_w; w_l = meas.s_l
            Z = (w_w * Z_w + w_l * Z_l) / (w_w + w_l + 1e-6)
        else:
            Z = Z_w if meas.s_w >= meas.s_l else Z_l
    elif Z_w is not None:
        Z = Z_w
    elif Z_l is not None:
        Z = Z_l
    return Z, Z_w, Z_l

# -----------------------------
# 模块 5.5：朝向估计（几何 + 在线逻辑回归）
# -----------------------------
class TinyOrientLR:
    """6 维特征的在线逻辑回归：x->[s_dir, s_norm, s_w, s_l, w_hat, l_hat]"""
    def __init__(self, dim=6, lr=0.05, l2=1e-3):
        self.w = np.zeros(dim, dtype=np.float32)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2
        self.n_pos = 0
        self.n_neg = 0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def predict_proba(self, x):
        x = np.asarray(x, np.float32)
        return float(self._sigmoid(float(self.w @ x + self.b)))

    def update(self, x, y):
        x = np.asarray(x, np.float32)
        p = self._sigmoid(float(self.w @ x + self.b))
        g = (p - float(y))
        self.w -= self.lr * (g * x + self.l2 * self.w)
        self.b -= self.lr * g
        if y == 1: self.n_pos += 1
        else: self.n_neg += 1

def orientation_features(pts_3d, w_px, l_px, s_w, s_l):
    P = pts_3d.astype(np.float32)
    V = P[12] - P[0]      # wrist->middle_tip
    U = P[5]  - P[0]      # wrist->index_mcp
    Wv = P[17] - P[0]     # wrist->pinky_mcp
    Vn = V / (np.linalg.norm(V) + 1e-6)
    n  = np.cross(U, Wv)
    nn = n / (np.linalg.norm(n) + 1e-6)
    s_dir  = -Vn[2]       # 越大越朝向相机
    s_norm = -nn[2]       # 掌面法向越朝向相机越大
    w_hat = float(w_px) / 60.0 if w_px else 0.0
    l_hat = float(l_px) / 100.0 if l_px else 0.0
    return np.array([s_dir, s_norm, float(s_w or 0.0), float(s_l or 0.0), w_hat, l_hat], dtype=np.float32)

def orientation_angle_deg(pts_3d):
    P = pts_3d.astype(np.float32)
    V = P[12] - P[0]
    Vn = V / (np.linalg.norm(V) + 1e-6)
    cos_t = float(np.clip(-Vn[2], -1.0, 1.0))  # 与 -Z 的夹角
    return float(np.degrees(np.arccos(cos_t))) # 0°=正对相机

# -----------------------------
# Trackbar 参数
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
    lr_orient = TinyOrientLR(dim=6, lr=0.05, l2=1e-3)

    last_t = time.time()

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
            meas, pts_3d = (meas_pack if meas_pack else (None, None))

            # 画 landmarks
            if DRAW_LANDMARKS and res.multi_hand_landmarks:
                for hlm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            # 标定采样显示 & 完成
            if calib.collecting:
                draw_text(frame, f"[标定采样] 请保持稳定手势/距离", 10, 26, (180, 230, 255))
                if meas is not None:
                    if meas.w_px and meas.w_px > 1:
                        calib.q_w.append(meas.w_px)
                    if meas.l_px and meas.l_px > 1:
                        calib.q_l.append(meas.l_px)
                draw_text(frame, f"进度: w {len(calib.q_w)}/{CALIB_SAMPLES} | l {len(calib.q_l)}/{CALIB_SAMPLES}", 10, 50, (180, 230, 255))
                if len(calib.q_w) >= CALIB_SAMPLES and len(calib.q_l) >= CALIB_SAMPLES:
                    ok_f = finish_collect_and_compute_f(calib, get_D_cm(), get_W_cm(), get_L_cm())
                    if ok_f:
                        draw_text(frame, f"[标定完成] f_w={calib.f_w:.1f}, f_l={calib.f_l:.1f}", 10, 74, (180, 255, 200))
                    else:
                        draw_text(frame, f"[标定失败] 请检查 D/W/L", 10, 74, (0, 0, 255))

            # 距离估计
            Z_cm = None; Zw = None; Zl = None
            if meas is not None and (calib.f_w or calib.f_l):
                Z_cm, Zw, Zl = estimate_Z_cm(meas, calib.f_w, calib.f_l, get_W_cm(), get_L_cm())
                if Z_cm is not None:
                    alpha = get_EMA()
                    state.Z_ema_cm = Z_cm if state.Z_ema_cm is None else (alpha * Z_cm + (1 - alpha) * state.Z_ema_cm)

            # 朝向估计
            p_face = None; theta_deg = None
            if meas is not None and pts_3d is not None:
                feats = orientation_features(pts_3d, meas.w_px, meas.l_px, meas.s_w, meas.s_l)
                p_face = lr_orient.predict_proba(feats)
                theta_deg = orientation_angle_deg(pts_3d)

            # FPS
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                state.fps = 0.9 * state.fps + 0.1 * (1.0 / dt) if state.fps > 0 else (1.0 / dt)

            # HUD
            y = 24
            draw_text(frame, f"FPS {state.fps:5.1f}", 10, y); y += 24
            if calib.f_w or calib.f_l:
                draw_text(frame, f"已标定  f_w={fmt1(calib.f_w)}  f_l={fmt1(calib.f_l)}", 10, y, (200,255,200)); y += 24
            else:
                draw_text(frame, f"未标定 - 调 D/W/L 后按 'c' 采样", 10, y, (80,230,255)); y += 24

            if meas is not None:
                draw_text(frame, f"w_px={meas.w_px:.1f} s_w={meas.s_w:.2f} | l_px={meas.l_px:.1f} s_l={meas.s_l:.2f}", 10, y); y += 24
            if Z_cm is not None:
                draw_text(frame, f"Z(cm) fused={fmt1(Z_cm)}  Zw={fmt1(Zw)}  Zl={fmt1(Zl)}", 10, y, (255,220,180)); y += 24
                if state.Z_ema_cm is not None:
                    draw_text(frame, f"Z_ema(cm)={fmt1(state.Z_ema_cm)}", 10, y, (255,220,180)); y += 24
            if p_face is not None and theta_deg is not None:
                draw_text(frame, f"Facing prob={p_face:.2f}  angle={theta_deg:.1f}° (0°=正对)", 10, y, (180,255,200)); y += 24
                draw_text(frame, "按键: c=标定  r=重置  g/h=正对/非正对样本  q=退出", 10, y, (220,220,220)); y += 24

            # 额外可视化：手腕处彩色圆点 + 判词
            if meas is not None:
                cx, cy = meas.p0
                # 颜色: 红(背对)->绿(正对)
                pf = 0.0 if p_face is None else clip(p_face, 0.0, 1.0)
                color = (int(255 * (1 - pf)), int(255 * pf), 0)
                cv2.circle(frame, (cx, cy), 18, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 18, color, thickness=-1, lineType=cv2.LINE_AA)

                # 文本标签
                label = "正对" if pf >= 0.65 else ("略偏" if pf >= 0.45 else "背对")
                draw_text(frame, label, cx + 24, cy + 6, color=(255,255,255), scale=0.7)

                # 画手指纵轴箭头（像素：wrist->middle_tip）
                cv2.arrowedLine(frame, meas.p0, meas.p12, (240, 240, 240), 2, tipLength=0.2)

            cv2.imshow(WIN_NAME, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('c'):
                start_collect(calib)
            elif k == ord('r'):
                calib = CalibState()
                state = RuntimeState()
                lr_orient = TinyOrientLR(dim=6, lr=0.05, l2=1e-3)
            elif k == ord('g'):
                if meas is not None and pts_3d is not None:
                    x = orientation_features(pts_3d, meas.w_px, meas.l_px, meas.s_w, meas.s_l)
                    lr_orient.update(x, 1)
            elif k == ord('h'):
                if meas is not None and pts_3d is not None:
                    x = orientation_features(pts_3d, meas.w_px, meas.l_px, meas.s_w, meas.s_l)
                    lr_orient.update(x, 0)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
