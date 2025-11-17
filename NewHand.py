#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小可用（升级版）：单目手部距离（米）估计 —— MediaPipe + 单点比例标定（z_vis）+ 掌宽法（width）+ 融合
- 依赖：pip install opencv-python mediapipe numpy
- 新操作：
  * 轨迹条 D_cm：标定用的真实距离（厘米）
  * 轨迹条 W_cm：你的真实掌宽（厘米，建议拿尺子量“食指根到小指根”5↔17）
  * 'c' ：在当前手距下完成标定（同时对 z_vis 与 width 两路一起标定）
  * 'r' ：清除标定（scale 与 f_eff 均清空）
  * 'q' ：退出
- 说明：
  * 路线 Z_from_z ：使用 MediaPipe 的相对深度 z_vis = -z，经比例尺 scale 变为米制：Zz = z_vis * scale
  * 路线 Z_from_width：使用像素掌宽 w_px 与真实掌宽 W_real，等效焦距 f_eff：Z_w = f_eff * W_real / w_px
  * 融合：Z = alpha * Zz + (1-alpha) * Zw（若其中一路无效则用另一路）
  * 代表点：掌心 = {0,5,9,13,17} 平均；掌宽用 5↔17 两点像素距离
  * 只做第一只手，保证稳定与速度。
"""

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ============ 可调参数（初学者主要改这里） ============
CAM_INDEX = 0            # 摄像头索引
FRAME_W, FRAME_H = 640, 480
MODEL_COMPLEXITY = 2     # 0 更快；1/2 更准但更慢
MAX_NUM_HANDS = 1        # 初学先做 1 只手
SMOOTH_ALPHA = 0.3       # EMA 平滑系数（0-1），越大越灵敏
CALIB_WINDOW = 15        # 标定时取最近多少帧的中位数
PALM_IDS = [0, 5, 9, 13, 17]
PALM_WIDTH_PAIR = (5, 17) # 用于掌宽的两点
ALPHA_Z = 0           # 融合权重：越大越倚重 z_vis 路线

# ============ 小工具 ============

def ema(prev, x, alpha):
    if prev is None:
        return x
    return alpha * x + (1 - alpha) * prev


def draw_text(img, txt, x, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fuse(z1, z2, alpha=ALPHA_Z):
    if (z1 is not None) and (z2 is not None):
        return alpha * z1 + (1 - alpha) * z2
    return z1 if z2 is None else z2

# ============ 初始化 ============
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

# HUD 窗口与轨迹条（标定距离、掌宽，单位厘米）
WIN_NAME = "HandDepth"
cv2.namedWindow(WIN_NAME)
cv2.createTrackbar("D_cm", WIN_NAME, 50, 200, lambda v: None)  # 1~200 cm

# 状态量（两路标定参数）
scale = None   # z_vis 的比例尺（m per z_vis）
f_eff = None   # 掌宽法的等效焦距常量（以像素·米/米 = 像素）

prev_palm_f = None  # EMA 后的接近度（palm）
prev_wrist_f = None
recent_palm_f = deque(maxlen=CALIB_WINDOW)
recent_width_px = deque(maxlen=CALIB_WINDOW)

fps_last_t = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.time()
    h, w = frame.shape[:2]

    # BGR->RGB 送入 MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    palm_center_px = None
    wrist_px = None
    palm_z_vis = None
    wrist_z_vis = None
    width_px = None

    if res.multi_hand_landmarks:
        # 只取第一只手
        lm = res.multi_hand_landmarks[0].landmark

        # 像素坐标
        pts_px = np.array([(int(l.x * w), int(l.y * h)) for l in lm], dtype=np.int32)

        # 掌心 = 指定 5 点平均
        palm_pts = pts_px[PALM_IDS]
        palm_center_px = palm_pts.mean(axis=0).astype(int)
        wrist_px = pts_px[0]

        # 掌宽像素（5↔17）
        p5 = pts_px[PALM_WIDTH_PAIR[0]].astype(np.float32)
        p17 = pts_px[PALM_WIDTH_PAIR[1]].astype(np.float32)
        width_px = float(np.linalg.norm(p5 - p17))

        # 相对深度（接近度）z_vis = -z
        palm_z_vis = -float(np.mean([lm[i].z for i in PALM_IDS]))
        wrist_z_vis = -float(lm[0].z)

        # EMA 平滑
        palm_f = ema(prev_palm_f, palm_z_vis, SMOOTH_ALPHA)
        wrist_f = ema(prev_wrist_f, wrist_z_vis, SMOOTH_ALPHA)
        prev_palm_f, prev_wrist_f = palm_f, wrist_f

        # 收集最近帧供标定
        recent_palm_f.append(palm_f)
        if width_px is not None and width_px > 0:
            recent_width_px.append(width_px)

        # 计算米制距离（两路 + 融合）
        palm_Zz = wrist_Zz = None
        palm_Zw = None

        if scale is not None:
            palm_Zz = float(palm_f * scale)
            wrist_Zz = float(wrist_f * scale)

        W_cm = 8.5  # 默认掌宽
        W_m = W_cm / 100.0
        if f_eff is not None and width_px is not None and width_px > 1:
            palm_Zw = float((f_eff * W_m) / width_px)

        palm_Z_m = fuse(palm_Zz, palm_Zw, ALPHA_Z)
        wrist_Z_m = fuse(wrist_Zz, palm_Zw, ALPHA_Z)  # wrist 没有独立 width 量，复用 palm 的 Zw

        # 画点
        if palm_center_px is not None:
            cv2.circle(frame, tuple(palm_center_px), 6, (0, 255, 0), -1)
        if wrist_px is not None:
            cv2.circle(frame, tuple(wrist_px), 6, (255, 200, 0), -1)
        if width_px is not None and PALM_WIDTH_PAIR[0] is not None:
            cv2.line(frame, tuple(pts_px[PALM_WIDTH_PAIR[0]]), tuple(pts_px[PALM_WIDTH_PAIR[1]]), (0, 180, 255), 2)

        # 文本输出
        y0 = 30
        draw_text(frame, f"width_px={width_px:.1f}" if width_px else "width_px=--", 10, y0)
        draw_text(frame, f"palm z_vis={palm_f:.4f}", 10, y0 + 24)
        draw_text(frame, f"wrist z_vis={wrist_f:.4f}", 10, y0 + 48)

        y1 = y0 + 75
        if scale is None and f_eff is None:
            draw_text(frame, "Z=未标定，按 c 完成标定", 10, y1, color=(200, 230, 255))
        else:
            # 显示三路：Zz、Zw、Z（融合）
            draw_text(frame, f"palm Zz={palm_Zz:.3f} m" if palm_Zz is not None else "palm Zz=--", 10, y1, color=(200, 230, 255))
            draw_text(frame, f"palm Zw={palm_Zw:.3f} m" if palm_Zw is not None else "palm Zw=--", 10, y1 + 24, color=(200, 230, 255))
            draw_text(frame, f"palm Z ={palm_Z_m:.3f} m (fused)" if palm_Z_m is not None else "palm Z =--", 10, y1 + 48, color=(255, 255, 200))
            draw_text(frame, f"wrist Z={wrist_Z_m:.3f} m" if wrist_Z_m is not None else "wrist Z=--", 10, y1 + 72, color=(255, 255, 200))

    # FPS
    dt = t0 - fps_last_t
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
    fps_last_t = t0
    draw_text(frame, f"FPS: {fps:.1f}", 10, frame.shape[0] - 12)

    # HUD：轨迹条显示
    D_cm = max(cv2.getTrackbarPos("D_cm", WIN_NAME), 1)
    draw_text(frame, f"D_cm={D_cm} ['c'=calib  'r'=reset  'q'=quit]", 10, frame.shape[0] - 36)

    cv2.imshow(WIN_NAME, frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('r'):
        scale = None
        f_eff = None
        recent_palm_f.clear()
        recent_width_px.clear()
    elif k == ord('c'):
        # 使用最近若干帧的中位数作为当前观测
        if len(recent_palm_f) >= max(3, CALIB_WINDOW // 2):
            f_med = float(np.median(recent_palm_f))
            D_m = max(D_cm, 1) / 100.0
            scale = (D_m / f_med) if f_med > 1e-6 else None
        # 同时给 width 路线标定 f_eff
        if len(recent_width_px) >= max(3, CALIB_WINDOW // 2):
            w_med = float(np.median(recent_width_px))
            D_m = max(D_cm, 1) / 100.0
            W_m = max(cv2.getTrackbarPos("W_cm", WIN_NAME), 1) / 100.0
            f_eff = (w_med * D_m / W_m) if (w_med > 1.0 and W_m > 1e-6) else None

cap.release()
cv2.destroyAllWindows()
