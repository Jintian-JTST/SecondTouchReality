# -*- coding: utf-8 -*-
# hand_two_hands_z_udp.py
"""
从 hand_easy 复用距离估计逻辑：
- 仍然用 hand_easy 里的 CalibState / RuntimeState / 各种 compute_* / fuse_depth；
- Z 深度的最终显示值 Z_disp，原样作为 wrist_z_m 通过 UDP 发给 Unity；
- 支持最多 2 只手，但 Z_disp 只对第 1 只手（hand_index=0）做滤波，与 hand_easy 行为一致。

按键：
  q / Esc  退出
  c  标定（正对摄像头、手张开，采样 50 帧后在终端输入真实距离）
  r  重置标定
"""

import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time

# 从 hand_easy 导入你原来的所有逻辑
from hand_easy import (
    CalibState,
    RuntimeState,
    compute_palm_width_and_length,
    compute_curl,
    compute_side,
    compute_face_sign,
    compute_palm_center_px,
    fuse_depth,
    clamp,
)

WIN_NAME = "TwoHandsDepthUDP"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---- UDP 配置 ----
UDP_IP = "127.0.0.1"
UDP_PORT = 5065


def draw_text_lines(img, lines, org=(10, 30), dy=22, color=(0, 255, 0)):
    x, y = org
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
        y += dy


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    calib = CalibState()
    state = RuntimeState()   # 注意：只一个，全局 state，和 hand_easy 一样
    last_t = time.time()
    EMA_ALPHA = 0.35

    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,           # 和原来两手脚本一样
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("读取摄像头失败")
                    break

                # 和 hand_easy 一样，把图像左右翻转
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # FPS
                now = time.time()
                dt = now - last_t
                if dt > 0:
                    fps = 1.0 / dt
                else:
                    fps = 0.0
                last_t = now

                # 送给 MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                rgb.flags.writeable = True

                hands_out = []

                # ===== 标定采样（完全照 hand_easy 的写法，只看第 1 只手）=====
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
                        print("标定采样结束。请保持刚才的姿态/距离不动。")
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

                # ===== 每只手处理 & 打包 =====
                if res.multi_hand_landmarks:
                    for hi, hand_lms in enumerate(res.multi_hand_landmarks):
                        lms = hand_lms.landmark

                        # 画手
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

                        # 掌心中心，用来贴 HUD（不发到 Unity）
                        palm_cx, palm_cy = compute_palm_center_px(lms, w, h)

                        # 计算你原来的全部特征
                        palm_width, palm_length = compute_palm_width_and_length(lms, w, h)
                        curl = compute_curl(lms, w, h)
                        side = compute_side(palm_width, palm_length, calib)

                        face_sign = compute_face_sign(lms)        # [-1,1]
                        palm_front = 0.5 * (face_sign + 1.0)     # 映射到 [0,1]
                        palm_front = 1.0 - clamp(palm_front, 0.0, 1.0)

                        Zw = None
                        Zl = None
                        if calib.k_w is not None and palm_width > 1e-3:
                            Zw = calib.k_w / palm_width
                        if calib.k_l is not None and palm_length > 1e-3:
                            Zl = calib.k_l / palm_length

                        Z_final_raw, w_w, w_l = fuse_depth(Zw, Zl, curl, side, palm_front)

                        # ===== 关键：Z_disp 逻辑，完全照 hand_easy =====
                        Z_disp = None
                        if hi == 0 and Z_final_raw is not None:
                            # 只对第 1 只手做滤波，使用同一个 state
                            state.z_hist.append(Z_final_raw)
                            Z_med = float(np.median(state.z_hist))

                            if state.z_ema is None:
                                state.z_ema = Z_med
                            else:
                                state.z_ema = EMA_ALPHA * Z_med + (1.0 - EMA_ALPHA) * state.z_ema

                            Z_disp = state.z_ema

                        # ===== landmarks 打包（和你之前 rec 的结构保持一致）=====
                        lm_list = []
                        for idx, lm in enumerate(lms):
                            nx = float(lm.x)
                            ny = float(lm.y)
                            nz = float(lm.z)

                            px = int(nx * w)
                            py = int(ny * h)

                            z_rel = nz
                            z_vis = abs(z_rel)

                            lm_list.append(
                                {
                                    "id": int(idx),
                                    "normalized": {"x": nx, "y": ny, "z": nz},
                                    "pixel": {"x": px, "y": py},
                                    "z_rel": float(z_rel),
                                    "z_vis": float(z_vis),
                                }
                            )

                        hand_dict = {
                            "hand_index": int(hi),
                            # 这里就是 hand_easy 里的 Z_disp，一模一样
                            "wrist_z_m": None if Z_disp is None else float(Z_disp),
                            "landmarks": lm_list,
                        }
                        hands_out.append(hand_dict)

                        # 在画面上给第 1 只手画 HUD，方便你对比数值
                        if hi == 0:
                            hud_lines = []
                            hud_lines.append(
                                f"Z_disp: {Z_disp if Z_disp is not None else -1:6.2f}"
                            )
                            if calib.k_w is None or calib.k_l is None:
                                hud_lines.append("Calib: NOT SET (c=calibrate)")
                            else:
                                hud_lines.append(
                                    f"Calib: OK  k_w={calib.k_w:.1f} k_l={calib.k_l:.1f}"
                                )
                            hud_lines.append(f"curl={curl:4.2f}, side={side:4.2f}")
                            hud_lines.append(
                                f"Zw={Zw if Zw is not None else -1:5.2f}, "
                                f"Zl={Zl if Zl is not None else -1:5.2f}"
                            )
                            draw_text_lines(
                                frame,
                                hud_lines,
                                org=(palm_cx + 10, palm_cy - 40),
                                dy=18,
                                color=(255, 255, 255),
                            )

                # ===== 发送 UDP =====
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

                # 画全局 HUD
                lines = [f"FPS: {fps:5.1f}"]
                if calib.sampling:
                    lines.append(f"Sampling... {len(calib.samples_w)} frames")
                draw_text_lines(frame, lines, org=(10, 30), dy=22, color=(0, 255, 0))

                cv2.imshow(WIN_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord("q"):
                    break
                elif key == ord("c") and not calib.sampling:
                    calib.sampling = True
                    calib.samples_w.clear()
                    calib.samples_l.clear()
                    print("=" * 60)
                    print("开始标定：")
                    print("请将一只手掌 **完全张开、正对摄像头**，保持不动。")
                    print("会自动采样约 50 帧，结束后在终端输入真实距离(米)。")
                    print("=" * 60)
                elif key == ord("r"):
                    calib.k_w = None
                    calib.k_l = None
                    calib.w_ref_open = None
                    calib.l_ref_open = None
                    calib.sampling = False
                    calib.samples_w.clear()
                    calib.samples_l.clear()
                    state.z_hist.clear()
                    state.z_ema = None
                    print("标定已重置。")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("退出。")


if __name__ == "__main__":
    main()
