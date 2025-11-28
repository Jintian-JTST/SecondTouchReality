"""
目标：
  1) 支持多只手（MediaPipe 里 max_num_hands 控制，上限 2 或更多都行）；
  2) 每只手输出：
       - 掌根位置（像素 + 归一化 + 深度米）；
       - 20 条骨骼方向向量（单位向量）；
       - 是否捏合 thumb/index (pinch: true/false)；
  3) UDP 发送 JSON：
       {
         "timestamp": ...,
         "fps": ...,
         "num_hands": <int>,
         "hands": [ {hand0...}, {hand1...}, ... ]
       }

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

from tools.hand_easy import (CalibState,RuntimeState,compute_palm_width_and_length,compute_curl,compute_side,compute_face_sign,fuse_depth,clamp)
# hand_udp.py 顶部
on_payload = None  # 外部可以把一个函数丢进来，每帧拿到 JSON

WIN_NAME = "Hand UDP Vectors (Multi-hand + Pinch)"

UDP_IP = "127.0.0.1"
UDP_PORT = 5065
ARDUINO_UDP_PORT = 5066      # 给 Python 舵机脚本
EMA_ALPHA = 0.6
PINCH_RATIO_THRESH = 0.35

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
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


def compute_pinch_thumb_index(landmarks, img_w, img_h, palm_width_px):
    if palm_width_px < 1e-3:
        return False

    lm_thumb = landmarks[4]
    lm_index = landmarks[8]

    dx = (lm_thumb.x - lm_index.x) * img_w
    dy = (lm_thumb.y - lm_index.y) * img_h
    dist_px = float((dx * dx + dy * dy) ** 0.5)

    ratio = dist_px / palm_width_px 
    return ratio < PINCH_RATIO_THRESH


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    calib = CalibState()
    states = defaultdict(RuntimeState)

    last_t = time.time()
    fps = 0.0
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
                    print("Failed to read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                now = time.time()
                dt = now - last_t
                fps = 1.0 / dt if dt > 0 else 0.0
                last_t = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                rgb.flags.writeable = True

                hands_out = []

                if calib.sampling and res.multi_hand_landmarks:
                    lms0 = res.multi_hand_landmarks[0].landmark
                    palm_w0, palm_l0 = compute_palm_width_and_length(lms0, w, h)
                    calib.samples_w.append(palm_w0)
                    calib.samples_l.append(palm_l0)

                    if len(calib.samples_w) >= 50:
                        calib.sampling = False
                        w_med = float(np.median(calib.samples_w))
                        l_med = float(np.median(calib.samples_l))

                        print("=" * 60)
                        print("Sampling complete.")
                        #print(f": {w_med:.2f} px")
                        #print(f"中位数掌长: {l_med:.2f} px")
                        d_real = float(input("Please enter the real distance (meters): ").strip())

                        calib.w_ref_open = w_med
                        calib.l_ref_open = l_med
                        calib.k_w = d_real * w_med
                        calib.k_l = d_real * l_med
                        calib.samples_w.clear()
                        calib.samples_l.clear()
                        #print(f"Calibration complete: k_w={calib.k_w:.4f}, k_l={calib.k_l:.4f}")
                        #print("=" * 60)

                if res.multi_hand_landmarks:
                    for hi, (hand_lms, handed) in enumerate(
                        zip(res.multi_hand_landmarks, res.multi_handedness)
                    ):
                        lms = hand_lms.landmark

                        handed_cls = handed.classification[0]
                        label = handed_cls.label
                        score = float(handed_cls.score)
                        is_left = (label == "Left")


                        wrist_lm = lms[0]
                        wrist_nx = float(wrist_lm.x)
                        wrist_ny = float(wrist_lm.y)
                        wrist_nz = float(wrist_lm.z)

                        wrist_px = int(wrist_nx * w)
                        wrist_py = int(wrist_ny * h)

                        palm_width, palm_length = compute_palm_width_and_length(lms, w, h)
                        curl = compute_curl(lms, w, h)
                        side = compute_side(palm_width, palm_length, calib)
                        face_sign = compute_face_sign(lms)

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

                        is_pinch = compute_pinch_thumb_index(
                            lms, w, h, palm_width
                        )
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
                                "dir": [dirx, diry, dirz],
                            })

                        hand_dict = {
                            "hand_index": int(hi),
                            "pinch": bool(is_pinch),
                            "is_left": bool(is_left),
                            "hand_label": label,
                            "hand_score": score,
                            "wrist": {
                                "pixel": {"x": wrist_px, "y": wrist_py},
                                "normalized": {
                                    "x": wrist_nx,
                                    "y": wrist_ny,
                                    "z": wrist_nz,
                                },
                                "depth_m": None if wrist_depth_m is None else float(wrist_depth_m),
                                "depth_cm": None if wrist_depth_m is None else round(float(wrist_depth_m) * 100.0, 2),
                            },
                            "bones": bones_out,
                        }

                        hands_out.append(hand_dict)

                        txt = ""
                        if wrist_depth_m is not None:
                            txt += f"Z: {wrist_depth_m * 100:.2f}cm  "
                        txt += f"pinch: {is_pinch}"
                        cv2.putText(
                            frame, txt,
                            (wrist_px + 10, wrist_py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA
                        )

                payload = {
                    "timestamp": time.time(),
                    "fps": float(fps),
                    "num_hands": len(hands_out),
                    "hands": hands_out,
                }


                if on_payload is not None:
                    try:
                        on_payload(payload)
                    except Exception as e:
                        print("on_payload error:", e)

                try:
                    data = json.dumps(payload).encode("utf-8")
                    sock.sendto(data, (UDP_IP, UDP_PORT))
                    sock.sendto(data, (UDP_IP, ARDUINO_UDP_PORT))
                except Exception:
                    pass

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
                    #print("=" * 60)
                    #print("开始标定：")
                    print("Please open your palm completely and face the camera.")
                    print("It will automatically sample about 50 frames.")
                    print("After that, enter the real distance (meters) in the terminal.")
                    print("=" * 60)
                elif key == ord("r"):
                    calib.k_w = calib.k_l = None
                    calib.w_ref_open = calib.l_ref_open = None
                    calib.sampling = False
                    calib.samples_w.clear()
                    calib.samples_l.clear()
                    for st in states.values():
                        st.z_hist.clear()
                        st.z_ema = None
                    print("Calibration reset.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("Exiting...")


if __name__ == "__main__":
    main()
