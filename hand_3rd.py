#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手掌“菱形”两对角线法：用掌宽 + 掌长 两条线估计相机-手的距离 Z（米）
- 依赖：pip install opencv-python mediapipe numpy
- 核心公式（单帧）：
    Z_w = f * W_real * s_w / w_px     （掌宽通道）
    Z_l = f * L_real * s_l / l_px     （掌长通道）
    Z   = (w_w*Z_w + w_l*Z_l)/(w_w + w_l) ，其中 w_w ∝ s_w^2，w_l ∝ s_l^2
- 标定（一次）：在已知距离 D_cm 下按 'c'，自动求 f（等效焦距，像素）
    f_w = w_px(D)*D / (W_real*s_w(D))   f_l = l_px(D)*D / (L_real*s_l(D))
    f   = 0.5*(f_w + f_l)
- UI：
    轨迹条 D_cm（标定距离，厘米）、W_cm（真实掌宽，厘米）、L_cm（真实掌长，厘米）
    'c' 标定 | 'r' 清除 | 'q' 退出
- 说明：
    * 21 点仅用于量 w_px、l_px，并用 3D 方向估计可见性 s_w、s_l（抗旋转压扁）
    * 本实现将逻辑“分块”，每块都独立清晰，方便你阅读与调试
"""

from collections import deque
import time
from dataclasses import dataclass, field
import cv2
import numpy as np
import mediapipe as mp

# ===================== 模块 0｜参数区 =====================
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
MODEL_COMPLEXITY = 0        # 0 更快；1/2 更准
MAX_NUM_HANDS = 1
EMA_ALPHA_Z = 0.25          # 最终 Z 的 EMA 平滑系数
CALIB_WINDOW = 15           # 标定窗口帧数（中位数）
S_MIN = 0.15                # 可见性下限（避免 s≈0 爆噪）
PX_MIN = 10                 # 像素长度下限（px）
TAU_ERR = 0.15              # 互相重投影的相对误差阈值（<=15% 视为相容）
GOOD_S = 0.60               # 认为“姿态良好”的 s 参考值（用于置信度映射）
HYST_FRAMES = 4             # 模式切换的最少连续帧数（迟滞）
MAX_JUMP = 0.30             # 单帧最大允许相对跳变（30%）

# 关键点编号（MediaPipe Hands）
ID_WRIST = 0
ID_INDEX_MCP = 5
ID_MIDDLE_MCP = 9
ID_MIDDLE_TIP = 12
ID_PINKY_MCP = 17

# ===================== 模块 1｜通用工具 =====================

def draw_text(img, txt, x, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def l2(a, b):
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

# ===================== 模块 2｜数据结构 =====================
@dataclass
class Measurements:
    # 像素测量
    w_px: float | None = None  # 掌宽像素（5↔17）
    l_px: float | None = None  # 掌长像素（12↔9 + 9↔0）
    # 可见性（基于 3D 方向投影）
    s_w: float | None = None
    s_l: float | None = None
    # 可视化点
    p5: tuple | None = None
    p17: tuple | None = None
    p12: tuple | None = None
    p9: tuple | None = None
    p0: tuple | None = None

@dataclass
class CalibState:
    f: float | None = None
    # 采样缓冲（中位数更稳）
    w_buf: deque = field(default_factory=lambda: deque(maxlen=CALIB_WINDOW))
    l_buf: deque = field(default_factory=lambda: deque(maxlen=CALIB_WINDOW))
    sw_buf: deque = field(default_factory=lambda: deque(maxlen=CALIB_WINDOW))
    sl_buf: deque = field(default_factory=lambda: deque(maxlen=CALIB_WINDOW))

    def clear(self):
        self.f = None
        self.w_buf.clear(); self.l_buf.clear(); self.sw_buf.clear(); self.sl_buf.clear()

# 运行时状态（用于迟滞与限幅）
@dataclass
class RuntimeState:
    mode: str | None = None   # 'W' | 'L' | 'F'（F=融合）
    streak: int = 0           # 候选模式连续帧计数（用于迟滞）
    Z_prev: float | None = None

# ===================== 模块 3｜从 21 点提取两条线 + 可见性 =====================

def extract_measurements(mp_landmarks, img_w, img_h) -> Measurements | None:
    """从 MediaPipe 的 21 点提取 w_px、l_px、s_w、s_l
    - 可见性 s 使用 3D 方向向量在像平面的投影长度：s = ||proj_xy(û)||，自然落在 [0,1]
    """
    if not mp_landmarks:
        return None
    lm = mp_landmarks[0].landmark  # 仅取第一只手

    # 像素坐标数组 (N,2) 与 归一化相机坐标数组 (N,3)
    pts_px = np.array([(int(l.x * img_w), int(l.y * img_h)) for l in lm], dtype=np.int32)
    pts_3d = np.array([(l.x, l.y, l.z) for l in lm], dtype=np.float32)  # 相对坐标即可

    # 掌宽像素（5↔17）
    p5, p17 = pts_px[ID_INDEX_MCP], pts_px[ID_PINKY_MCP]
    w_px = l2(p5, p17)

    # 掌长像素（12↔9 + 9↔0）
    p12, p9, p0 = pts_px[ID_MIDDLE_TIP], pts_px[ID_MIDDLE_MCP], pts_px[ID_WRIST]
    l_px = l2(p12, p9) + l2(p9, p0)

    # 3D 方向（用归一化相机坐标求方向即可）
    U = unit(pts_3d[ID_INDEX_MCP] - pts_3d[ID_PINKY_MCP])     # 宽轴向量
    V = unit(pts_3d[ID_MIDDLE_TIP] - pts_3d[ID_WRIST])        # 长轴向量（顶→根）

    # 可见性 = 在像素平面 (x,y) 的分量长度
    s_w = float(np.clip(np.linalg.norm(U[:2]), 0.0, 1.0))
    s_l = float(np.clip(np.linalg.norm(V[:2]), 0.0, 1.0))

    # 下限防爆噪
    s_w = max(s_w, S_MIN)
    s_l = max(s_l, S_MIN)

    return Measurements(
        w_px=w_px, l_px=l_px, s_w=s_w, s_l=s_l,
        p5=tuple(p5), p17=tuple(p17), p12=tuple(p12), p9=tuple(p9), p0=tuple(p0)
    )

# ===================== 模块 4｜标定：求 f =====================

def calibrate_f(calib: CalibState, D_m: float, W_m: float, L_m: float) -> float | None:
    """用缓冲区的中位数求 f，失败返回 None"""
    if len(calib.w_buf) < max(3, CALIB_WINDOW // 2) or len(calib.l_buf) < max(3, CALIB_WINDOW // 2):
        return None
    w_med = float(np.median(calib.w_buf))
    l_med = float(np.median(calib.l_buf))
    sw_med = float(np.median(calib.sw_buf))
    sl_med = float(np.median(calib.sl_buf))

    if W_m <= 1e-6 or L_m <= 1e-6 or D_m <= 1e-6:
        return None

    f_w = (w_med * D_m) / (W_m * max(sw_med, S_MIN))
    f_l = (l_med * D_m) / (L_m * max(sl_med, S_MIN))
    f = 0.5 * (f_w + f_l)
    calib.f = float(f)
    return calib.f

# ===================== 模块 5｜估距：两通道 + 自判定/互斥 + 迟滞 =====================

def estimate_Z_robust(meas: Measurements, f: float, W_m: float, L_m: float, state: RuntimeState):
    """
    返回：Zw, Zl, Z_out, conf_w, conf_l, mode
    逻辑：
      1) 有效性门限（s 与 像素长度）
      2) 各自出候选距离 Zw/Zl
      3) 互相重投影一致性检查（相对误差 <= TAU_ERR）
      4) 置信度：c = g(s) * h(px) * I(一致)
      5) 决策：单通道/明显更优/温和融合
      6) 迟滞：连续 HYST_FRAMES 帧更优才切换主导模式
      7) 限幅：单帧最大相对变化 MAX_JUMP
    """
    def clamp01(x):
        return float(min(max(x, 0.0), 1.0))

    if f is None or meas is None:
        return None, None, None, 0.0, 0.0, state.mode

    # 1) 有效性
    valid_w = (meas.s_w >= S_MIN) and (meas.w_px is not None) and (meas.w_px >= PX_MIN)
    valid_l = (meas.s_l >= S_MIN) and (meas.l_px is not None) and (meas.l_px >= PX_MIN)

    Zw = Zl = None
    if valid_w:
        Zw = float((f * W_m * meas.s_w) / meas.w_px)
    if valid_l:
        Zl = float((f * L_m * meas.s_l) / meas.l_px)

    # 2) 互相重投影一致性
    c_w = 0.0; c_l = 0.0
    if valid_w and Zw is not None and meas.l_px is not None and meas.l_px > 1:
        lhat = (f * L_m * meas.s_l) / Zw
        e = abs(meas.l_px - lhat) / (meas.l_px + 1e-6)
        ok = (e <= TAU_ERR)
        g = clamp01((meas.s_w - S_MIN) / (GOOD_S - S_MIN))
        h = clamp01((meas.w_px - PX_MIN) / (60 - PX_MIN))
        c_w = (1.0 if ok else 0.0) * g * h

    if valid_l and Zl is not None and meas.w_px is not None and meas.w_px > 1:
        what = (f * W_m * meas.s_w) / Zl
        e = abs(meas.w_px - what) / (meas.w_px + 1e-6)
        ok = (e <= TAU_ERR)
        g = clamp01((meas.s_l - S_MIN) / (GOOD_S - S_MIN))
        h = clamp01((meas.l_px - PX_MIN) / (60 - PX_MIN))
        c_l = (1.0 if ok else 0.0) * g * h

    # 3) 模式建议
    suggested = None
    if (c_w > 0) and (c_l == 0):
        suggested = 'W'
    elif (c_l > 0) and (c_w == 0):
        suggested = 'L'
    elif (c_w == 0) and (c_l == 0):
        suggested = None
    else:
        if c_w >= 1.5 * c_l:
            suggested = 'W'
        elif c_l >= 1.5 * c_w:
            suggested = 'L'
        else:
            suggested = 'F'

    # 4) 迟滞：只有当候选模式连续 HYST_FRAMES 帧更优才切换
    if suggested is not None:
        if state.mode is None:
            state.mode = suggested
            state.streak = 0
        elif suggested != state.mode:
            state.streak += 1
            if state.streak >= HYST_FRAMES:
                state.mode = suggested
                state.streak = 0
        else:
            state.streak = 0

    # 5) 选最终 Z
    Z_out = None
    if state.mode == 'W' and Zw is not None:
        Z_out = Zw
    elif state.mode == 'L' and Zl is not None:
        Z_out = Zl
    elif state.mode == 'F' and (c_w + c_l) > 0:
        Z_out = (c_w * (Zw or 0.0) + c_l * (Zl or 0.0)) / (c_w + c_l)
    else:
        # 双失效或无建议，尝试退化：谁有用用谁
        Z_out = Zw if Zw is not None else Zl

    # 6) 限幅防抖
    if (Z_out is not None) and (state.Z_prev is not None):
        low = state.Z_prev * (1.0 - MAX_JUMP)
        high = state.Z_prev * (1.0 + MAX_JUMP)
        Z_out = float(max(min(Z_out, high), low))
    if Z_out is not None:
        state.Z_prev = Z_out

    return Zw, Zl, Z_out, c_w, c_l, state.mode

# ===================== 模块 6｜主循环 =====================

def main():
    # 摄像头与手部模型
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # HUD 窗口与轨迹条
    WIN = "HandDepth-Diamond"
    cv2.namedWindow(WIN)
    cv2.createTrackbar("D_cm", WIN, 60, 200, lambda v: None)  # 标定距离（厘米）
    cv2.createTrackbar("W_cm", WIN, 8, 30, lambda v: None)    # 真实掌宽（厘米）
    cv2.createTrackbar("L_cm", WIN, 18, 35, lambda v: None)   # 真实掌长（厘米）

    calib = CalibState()
    state = RuntimeState()
    Z_ema = None

    fps_last_t = time.time(); fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        h, w = frame.shape[:2]

        # 识别 21 点
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        meas = extract_measurements(res.multi_hand_landmarks, w, h)

        # 累积标定样本
        if meas is not None:
            calib.w_buf.append(meas.w_px)
            calib.l_buf.append(meas.l_px)
            calib.sw_buf.append(meas.s_w)
            calib.sl_buf.append(meas.s_l)

        # 估距（若已标定）
        W_m = max(cv2.getTrackbarPos("W_cm", WIN), 1) / 100.0
        L_m = max(cv2.getTrackbarPos("L_cm", WIN), 1) / 100.0
        Zw = Zl = Z = None; c_w = c_l = 0.0; mode = state.mode
        if meas is not None and calib.f is not None:
            Zw, Zl, Z, c_w, c_l, mode = estimate_Z_robust(meas, calib.f, W_m, L_m, state)
            if Z is not None:
                Z_ema = ema(Z_ema, Z, EMA_ALPHA_Z)

        # 可视化
        if meas is not None:
            # 画线：掌宽与掌长
            cv2.line(frame, meas.p5, meas.p17, (0, 200, 255), 2)
            cv2.line(frame, meas.p12, meas.p9, (0, 255, 0), 2)
            cv2.line(frame, meas.p9, meas.p0, (0, 255, 0), 2)

            y0 = 26
            draw_text(frame, f"w_px={meas.w_px:.1f}  s_w={meas.s_w:.2f}", 10, y0)
            draw_text(frame, f"l_px={meas.l_px:.1f}  s_l={meas.s_l:.2f}", 10, y0+22)

        # 显示 Z 与模式
        y1 = 26 + 22*2 + 6
        if calib.f is None:
            draw_text(frame, "Z=未标定，按 'c' 在 D_cm 处标定", 10, y1, color=(200,230,255))
        else:
            draw_text(frame, f"Zw={Zw:.3f} m" if Zw is not None else "Zw=--", 10, y1)
            draw_text(frame, f"Zl={Zl:.3f} m" if Zl is not None else "Zl=--", 10, y1+22)
            draw_text(frame, f"Z={Z:.3f} m  mode={mode or '-'}  cw={c_w:.2f} cl={c_l:.2f}" if Z is not None else "Z=--", 10, y1+44, color=(255,255,200))
            draw_text(frame, f"Z(EMA)={Z_ema:.3f} m" if Z_ema is not None else "Z(EMA)=--", 10, y1+66, color=(255,255,200))

        # FPS & HUD
        dt = max(t0 - fps_last_t, 1e-6)
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
        fps_last_t = t0
        draw_text(frame, f"FPS: {fps:.1f}", 10, h - 12)
        D_cm = max(cv2.getTrackbarPos("D_cm", WIN), 1)
        draw_text(frame, f"D_cm={D_cm}  W_cm={int(W_m*100)}  L_cm={int(L_m*100)}  ['c':calib  'r':reset  'q':quit]", 10, h - 34)

        cv2.imshow(WIN, frame)

        # 键盘控制
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            calib.clear(); state = RuntimeState(); Z_ema = None
        elif k == ord('c'):
            D_m = max(D_cm, 1) / 100.0
            _ = calibrate_f(calib, D_m, W_m, L_m)
            # 标定后清空缓冲，防止旧样本污染下一次
            calib.w_buf.clear(); calib.l_buf.clear(); calib.sw_buf.clear(); calib.sl_buf.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    main()
