# -*- coding: utf-8 -*-
# hand_udp_vectors.py
"""
目标：
  1) 只获取掌根位置（像素 + 归一化 + 深度米）；
  2) 从 MediaPipe 的 21 点计算手部骨骼方向向量（单位向量）；
  3) 通过 UDP 把这些信息发给 Unity / 其他程序；
  4) 后端（Python / Unity）用「掌根 3D 位置 + 每节骨骼长度 + 骨骼方向向量」重建 21 个点。

按键：
  q / ESC : 退出
  c       : 开始标定（采样 50 帧，然后在终端输入真实距离）
  r       : 重置标定
"""

import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time
from collections import defaultdict

# ======== 从 hand_easy 复用深度逻辑 ========
from hand_easy import (
    CalibState,
    RuntimeState,
    compute_palm_width_and_length,
    compute_curl,
    compute_side,
    compute_face_sign,
    fuse_depth,
    clamp,
)

# ======== 配置 ========
WIN_NAME = "Hand UDP Vectors"

# UDP 地址（Unity 那边要监听同样端口）
UDP_IP = "127.0.0.1"
UDP_PORT = 5065

# EMA 平滑
EMA_ALPHA = 0.35

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# 骨骼拓扑：from_idx -> to_idx
BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),      # 食指
    (0, 9), (9, 10), (10, 11), (11, 12), # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

def draw_text_lines(img, lines, org=(10, 30), dy=22, color=(0, 255, 0)):
    x, y = org
    for line in lines:
        cv2.putText(
            img, line, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 1, cv2.LINE_AA
        )
        y += dy


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 标定状态：全局一份（和 hand_easy 一样）
    calib = CalibState()
    # 每只手一个 RuntimeState，用于各自的深度平滑（避免两只手互相干扰）
    states = defaultdict(RuntimeState)

    last_t = time.time()
    fps = 0.0

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("读取摄像头失败")
                    break

                # 和 hand_easy 一样镜像翻转，保持直观
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # FPS
                now = time.time()
                dt = now - last_t
                fps = 1.0 / dt if dt > 0 else 0.0
                last_t = now

                # MediaPipe 推理
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                rgb.flags.writeable = True

                hands_out = []

                # ===== 标定采样：只看第 1 只检测到的手 =====
                if calib.sampling and res.multi_hand_landmarks:
                    lms0 = res.multi_hand_landmarks[0].landmark
                    palm_w, palm_l = compute_palm_width_and_length(lms0, w, h)
                    calib.samples_w.append(palm_w)
                    calib.samples_l.append(palm_l)

                    if len(calib.samples_w) >= 50:
                        calib.sampling = False
                        w_med = float(np.median(calib.samples_w))
                        l_med = float(np.median(calib.samples_l))

                        print("=" * 60)
                        print("标定采样结束，请保持刚才姿态不动。")
                        print(f"中位数掌宽: {w_med:.2f} px")
                        print(f"中位数掌长: {l_med:.2f} px")
                        d_real = float(input("请输入此时掌根到摄像头的真实距离(米): ").strip())

                        calib.w_ref_open = w_med
                        calib.l_ref_open = l_med
                        calib.k_w = d_real * w_med
                        calib.k_l = d_real * l_med
                        calib.samples_w.clear()
                        calib.samples_l.clear()
                        print(f"标定完成: k_w={calib.k_w:.4f}, k_l={calib.k_l:.4f}")
                        print("=" * 60)

                # ===== 处理每只手 =====
                if res.multi_hand_landmarks:
                    for hi, hand_lms in enumerate(res.multi_hand_landmarks):
                        lms = hand_lms.landmark

                        # 可视化骨架
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                        # ------------- 1) 掌根像素 / 标定深度 -------------
                        wrist_lm = lms[0]
                        wrist_nx = float(wrist_lm.x)
                        wrist_ny = float(wrist_lm.y)
                        wrist_nz = float(wrist_lm.z)

                        wrist_px = int(wrist_nx * w)
                        wrist_py = int(wrist_ny * h)

                        # 深度估计（完全沿用 hand_easy 的逻辑）
                        palm_width, palm_length = compute_palm_width_and_length(lms, w, h)
                        curl = compute_curl(lms, w, h)
                        side = compute_side(palm_width, palm_length, calib)
                        face_sign = compute_face_sign(lms)  # [-1,1]

                        palm_front = 0.5 * (face_sign + 1.0)
                        palm_front = 1.0 - clamp(palm_front, 0.0, 1.0)

                        Zw = None
                        Zl = None
                        if calib.k_w is not None and palm_width > 1e-3:
                            Zw = calib.k_w / palm_width
                        if calib.k_l is not None and palm_length > 1e-3:
                            Zl = calib.k_l / palm_length

                        Z_raw, w_w, w_l = fuse_depth(Zw, Zl, curl, side, palm_front)

                        wrist_depth_m = None
                        if Z_raw is not None:
                            st = states[hi]
                            st.z_hist.append(Z_raw)
                            Z_med = float(np.median(st.z_hist))
                            if st.z_ema is None:
                                st.z_ema = Z_med
                            else:
                                st.z_ema = EMA_ALPHA * Z_med + (1.0 - EMA_ALPHA) * st.z_ema
                            wrist_depth_m = st.z_ema

                        # ------------- 2) 计算 21 点 -> 20 条骨骼方向向量 -------------
                        # 用归一化坐标 (x, y, z) 当作相机坐标系里的“形状”，只取方向
                        coords = [(float(p.x), float(p.y), float(p.z)) for p in lms]

                        bones_out = []
                        for bi, (a, b) in enumerate(BONE_PAIRS):
                            ax, ay, az = coords[a]
                            bx, by, bz = coords[b]
                            dx = bx - ax
                            dy = by - ay
                            dz = bz - az
                            length = (dx * dx + dy * dy + dz * dz) ** 0.5
                            if length < 1e-6:
                                dirx = diry = dirz = 0.0
                            else:
                                dirx = dx / length
                                diry = dy / length
                                dirz = dz / length

                            bones_out.append({
                                "id": bi,
                                "from": a,
                                "to": b,
                                "dir": [dirx, diry, dirz]
                            })

                        # ------------- 3) 打包 hand JSON -------------
                        hand_dict = {
                            "hand_index": int(hi),
                            "wrist": {
                                "pixel": {"x": wrist_px, "y": wrist_py},
                                "normalized": {
                                    "x": wrist_nx,
                                    "y": wrist_ny,
                                    "z": wrist_nz,
                                },
                                "depth_m": None if wrist_depth_m is None else float(wrist_depth_m),
                            },
                            "bones": bones_out,
                        }
                        hands_out.append(hand_dict)

                        # 屏幕 HUD（在掌根旁边画一下深度，方便你对比 hand_easy）
                        if wrist_depth_m is not None:
                            text = f"Z: {wrist_depth_m:.3f} m"
                            cv2.putText(
                                frame, text,
                                (wrist_px + 10, wrist_py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1, cv2.LINE_AA
                            )

                # ===== 4) UDP 发送 =====
                payload = {
                    "timestamp": time.time(),
                    "fps": float(fps),
                    "hands": hands_out,
                }
                try:
                    data = json.dumps(payload).encode("utf-8")
                    sock.sendto(data, (UDP_IP, UDP_PORT))
                except Exception:
                    pass

                # 全局 HUD
                hud = [f"FPS: {fps:5.1f}"]
                if calib.sampling:
                    hud.append(f"Calib sampling... {len(calib.samples_w)}/50")
                elif calib.k_w is None or calib.k_l is None:
                    hud.append("Calib: NOT SET (press 'c')")
                else:
                    hud.append("Calib: OK (press 'r' to reset)")
                draw_text_lines(frame, hud, org=(10, 30), dy=22)

                cv2.imshow(WIN_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                elif key == ord("c") and not calib.sampling:
                    calib.sampling = True
                    calib.samples_w.clear()
                    calib.samples_l.clear()
                    print("=" * 60)
                    print("开始标定：")
                    print("请将单手掌 **完全张开、正对摄像头**，保持不动。")
                    print("会自动采样约 50 帧，结束后在终端输入真实距离(米)。")
                    print("=" * 60)
                elif key == ord("r"):
                    calib.k_w = calib.k_l = None
                    calib.w_ref_open = calib.l_ref_open = None
                    calib.sampling = False
                    calib.samples_w.clear()
                    calib.samples_l.clear()
                    # 重置所有手的运行状态
                    for st in states.values():
                        st.z_hist.clear()
                        st.z_ema = None
                    print("标定已重置。")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("退出。")


if __name__ == "__main__":
    main()
