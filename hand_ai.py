# hand_depth_unified.py
# -*- coding: utf-8 -*-
"""
整合版（仅深度，无通信）：
- MediaPipe 21 点
- 掌宽(5↔17) 与 掌长(0↔9) 两通道
- 正确投影校正：Z = f * L_nom * s / ell_px,  s = ||(vx,vy)|| / ||(vx,vy,vz)||
- 标定：按 'c' 自动采样 CALIB_SAMPLES 帧中位数，算 f_w / f_l（允许 s != 1）
- 融合：用 s^2 当质量权重 + 简单一致性门控
- 轻量 EMA 平滑（仅为抗偶发坏点，不改变几何）
依赖：pip install opencv-python mediapipe numpy
"""

import cv2
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import mediapipe as mp

# =========================
# 参数区
# =========================
WIN = "Hand Depth (Unified, s-corrected)"
MAX_NUM_HANDS = 1
DRAW_LANDMARKS = True

# 标定
CALIB_SAMPLES = 60
DEFAULT_D_CM  = 40      # 标定距离（厘米）
DEFAULT_W_CM  = 9       # 掌宽名义长度（厘米）   -> 5↔17
DEFAULT_L_CM  = 10      # 掌长名义长度（厘米）   -> 0↔9（避开指尖）

# 融合与平滑
S_VIS_TH   = 0.80       # s 的最低可用阈值
AGREE_FRAC = 0.05       # 双通道相对差异阈值（<则求加权平均）
EMA_INIT   = 0.25       # EMA 初值（可轨迹条调）
Z_MAX_CM   = 200.0      # 限幅

# =========================
# 工具
# =========================
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_text(img, txt, x, y, color=(255,255,255), scale=0.6):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def l2(p, q):
    p = np.asarray(p, np.float32); q = np.asarray(q, np.float32)
    return float(np.linalg.norm(p - q))

def fmt1(x):
    return "—" if x is None or (isinstance(x, float) and (math.isnan(x))) else f"{x:.1f}"

def clip(v, lo, hi):
    return max(lo, min(hi, v))

# =========================
# 数据结构
# =========================
@dataclass
class Measurements:
    w_px: Optional[float]
    l_px: Optional[float]
    s_w: float
    s_l: float
    p0: Tuple[int,int]
    p5: Tuple[int,int]
    p9: Tuple[int,int]
    p17: Tuple[int,int]

@dataclass
class CalibState:
    collecting: bool = False
    q_w_px: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_l_px: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_sw:   deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_sl:   deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    f_w: Optional[float] = None   # 等效焦距(像素) for 宽
    f_l: Optional[float] = None   # 等效焦距(像素) for 长

@dataclass
class RuntimeState:
    Z_ema_cm: Optional[float] = None
    fps: float = 0.0

# =========================
# 从21点提两条线 + s 因子
# =========================
def extract_meas(hand_landmarks, img_w, img_h):
    """返回 (Measurements, pts_3d) 或 None"""
    if not hand_landmarks:
        return None
    lm = hand_landmarks[0].landmark

    pts_px = np.array([(int(l.x * img_w), int(l.y * img_h)) for l in lm], dtype=np.int32)
    pts_3d = np.array([(l.x, l.y, l.z) for l in lm], dtype=np.float32)  # MP: z<0 朝相机

    p0  = pts_px[0]   # wrist
    p5  = pts_px[5]   # index_mcp
    p9  = pts_px[9]   # middle_mcp
    p17 = pts_px[17]  # pinky_mcp

    # 像素长度
    w_px = l2(p5, p17)          # 掌宽 5↔17
    l_px = l2(p0, p9)           # 掌长 0↔9（避开指尖）

    # s 因子（投影校正）
    vw = pts_3d[17] - pts_3d[5]
    vl = pts_3d[9]  - pts_3d[0]
    def s_of(v):
        v = v.astype(np.float32)
        n3 = float(np.linalg.norm(v)) + 1e-6
        n2 = float(np.linalg.norm(v[:2]))
        return clip(n2 / n3, 0.0, 1.0)
    s_w = s_of(vw)
    s_l = s_of(vl)

    meas = Measurements(w_px=w_px, l_px=l_px, s_w=s_w, s_l=s_l,
                        p0=tuple(p0), p5=tuple(p5), p9=tuple(p9), p17=tuple(p17))
    return meas, pts_3d

# =========================
# 标定（中位数；允许 s!=1）
# =========================
def start_collect(calib: CalibState):
    calib.collecting = True
    calib.q_w_px.clear(); calib.q_l_px.clear()
    calib.q_sw.clear();   calib.q_sl.clear()

def finish_collect_and_compute_f(calib: CalibState, D_cm: float, W_cm: float, L_cm: float):
    """f = D * median(ell_px) / (L_nom * median(s))"""
    if len(calib.q_w_px) == 0 or len(calib.q_l_px) == 0:
        calib.collecting = False
        return False
    w_med = float(np.median(list(calib.q_w_px)))
    l_med = float(np.median(list(calib.q_l_px)))
    sw_med = max(1e-3, float(np.median(list(calib.q_sw))))  # 防止除0
    sl_med = max(1e-3, float(np.median(list(calib.q_sl))))
    if D_cm <= 0 or W_cm <= 0 or L_cm <= 0:
        calib.collecting = False
        return False
    calib.f_w = (D_cm * w_med) / (W_cm * sw_med)
    calib.f_l = (D_cm * l_med) / (L_cm * sl_med)
    calib.collecting = False
    return True

# =========================
# 两通道深度 + 融合
# =========================
def z_from_channel(f_pix, L_nom_cm, s, ell_px):
    """Z = f * L * s / ell_px"""
    if not f_pix or not ell_px or ell_px <= 1e-3 or s <= 1e-6:
        return None
    z = (f_pix * L_nom_cm * s) / float(ell_px)
    return clip(z, 0.0, Z_MAX_CM)

def estimate_fused_Z(meas: Measurements, calib: CalibState, W_cm: float, L_cm: float):
    Zw = z_from_channel(calib.f_w, W_cm, meas.s_w, meas.w_px) if calib.f_w else None
    Zl = z_from_channel(calib.f_l, L_cm, meas.s_l, meas.l_px) if calib.f_l else None

    Z = None
    if Zw is not None and Zl is not None:
        # 质量权重：s^3（s 小表示严重压扁 -> 降权）
        ww = (meas.s_w ** 1) if meas.s_w > S_VIS_TH else 0.0
        wl = (meas.s_l ** 1) if meas.s_l > S_VIS_TH else 0.0
        if ww == 0.0 and wl == 0.0:
            # 两条都太斜，择其一（选像素长度更“可信”的那条）
            Z = Zw if meas.s_w >= meas.s_l else Zl
        else:
            # 一致性门控：相对差异小则加权平均，否则取权重大者
            rel = abs(Zw - Zl) / max(1e-6, 0.5 * (Zw + Zl))
            if rel < AGREE_FRAC and (ww > 0 and wl > 0):
                Z = (ww * Zw + wl * Zl) / (ww + wl)
            else:
                Z = Zw if ww >= wl else Zl
    elif Zw is not None:
        Z = Zw
    elif Zl is not None:
        Z = Zl

    return Z, Zw, Zl

# =========================
# Trackbars
# =========================
def _noop(v): pass

def create_trackbars():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.createTrackbar("D_cm", WIN, DEFAULT_D_CM, 300, _noop)
    cv2.createTrackbar("W_cm", WIN, DEFAULT_W_CM, 40, _noop)
    cv2.createTrackbar("L_cm", WIN, DEFAULT_L_CM, 40, _noop)
    cv2.createTrackbar("EMA_x100", WIN, int(EMA_INIT * 100), 100, _noop)

def get_D_cm(): return max(1, cv2.getTrackbarPos("D_cm", WIN))
def get_W_cm(): return max(1, cv2.getTrackbarPos("W_cm", WIN))
def get_L_cm(): return max(1, cv2.getTrackbarPos("L_cm", WIN))
def get_EMA():  return clip(cv2.getTrackbarPos("EMA_x100", WIN) / 100.0, 0.0, 1.0)

# =========================
# 主循环
# =========================
def main():
    create_trackbars()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    calib = CalibState()
    state = RuntimeState()

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

            meas_pack = extract_meas(res.multi_hand_landmarks, w, h) if res.multi_hand_landmarks else None
            meas, pts_3d = (meas_pack if meas_pack else (None, None))

            # 画骨架
            if DRAW_LANDMARKS and res.multi_hand_landmarks:
                for hlm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # 标定采样显示 & 完成
            if calib.collecting:
                draw_text(frame, "[标定采样] 请在 D_cm 处保持手势稳定（越接近正对越好）", 10, 26, (180,230,255))
                if meas is not None:
                    if meas.w_px and meas.w_px > 1:
                        calib.q_w_px.append(meas.w_px)
                        calib.q_sw.append(meas.s_w)
                    if meas.l_px and meas.l_px > 1:
                        calib.q_l_px.append(meas.l_px)
                        calib.q_sl.append(meas.s_l)
                draw_text(frame, f"进度: w {len(calib.q_w_px)}/{CALIB_SAMPLES} | l {len(calib.q_l_px)}/{CALIB_SAMPLES}", 10, 50, (180,230,255))
                if len(calib.q_w_px) >= CALIB_SAMPLES and len(calib.q_l_px) >= CALIB_SAMPLES:
                    ok_f = finish_collect_and_compute_f(calib, get_D_cm(), get_W_cm(), get_L_cm())
                    if ok_f:
                        draw_text(frame, f"[标定完成] f_w={calib.f_w:.1f}, f_l={calib.f_l:.1f}", 10, 74, (180,255,200))
                    else:
                        draw_text(frame, "[标定失败] 检查 D/W/L 与采样质量", 10, 74, (0,0,255))

            # 估距
            Z = None; Zw = None; Zl = None
            if meas is not None and (calib.f_w or calib.f_l):
                Z, Zw, Zl = estimate_fused_Z(meas, calib, get_W_cm(), get_L_cm())
                if Z is not None:
                    alpha = get_EMA()
                    state.Z_ema_cm = Z if state.Z_ema_cm is None else (alpha * Z + (1 - alpha) * state.Z_ema_cm)

            # FPS
            now = time.time()
            dt = now - last_t; last_t = now
            if dt > 0:
                state.fps = 0.9 * state.fps + 0.1 * (1.0/dt) if state.fps > 0 else (1.0/dt)

            # HUD
            y = 24
            draw_text(frame, f"FPS {state.fps:5.1f}", 10, y); y += 24
            if calib.f_w or calib.f_l:
                draw_text(frame, f"已标定  f_w={fmt1(calib.f_w)}  f_l={fmt1(calib.f_l)}", 10, y, (200,255,200)); y += 24
            else:
                draw_text(frame, f"未标定 - 调 D/W/L 后按 'c' 采样", 10, y, (80,230,255)); y += 24

            if meas is not None:
                draw_text(frame, f"w_px={meas.w_px:.1f} s_w={meas.s_w:.2f} | l_px={meas.l_px:.1f} s_l={meas.s_l:.2f}", 10, y); y += 24
            if Z is not None:
                draw_text(frame, f"Z(cm) fused={fmt1(Z)}  Zw={fmt1(Zw)}  Zl={fmt1(Zl)}", 10, y, (255,220,180)); y += 24
                if state.Z_ema_cm is not None:
                    draw_text(frame, f"Z_ema(cm)={fmt1(state.Z_ema_cm)}", 10, y, (255,220,180)); y += 24

            # 可视反馈：在手腕处画当前 Z（颜色随 s_w/s_l）
            if meas is not None:
                cx, cy = meas.p0
                s_avg = 0.5 * (meas.s_w + meas.s_l)
                s_avg = clip(s_avg, 0.0, 1.0)
                color = (int(255*(1-s_avg)), int(200*s_avg), int(255*s_avg))
                cv2.circle(frame, (cx, cy), 18, (0,0,0), 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 18, color, -1, cv2.LINE_AA)
                if Z is not None:
                    draw_text(frame, f"{fmt1(Z)} cm", cx + 24, cy + 6, (255,255,255), scale=0.7)

            draw_text(frame, "键位: c=开始标定  r=重置  q=退出", 10, h-12, (220,220,220))
            cv2.imshow(WIN, frame)

            # 键盘
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('c'):
                start_collect(calib)
            elif k == ord('r'):
                calib = CalibState()
                state = RuntimeState()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
