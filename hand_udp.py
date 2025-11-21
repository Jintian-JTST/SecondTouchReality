# -*- coding: utf-8 -*-
"""
hand_easy.py （重写版）

功能：
- 使用 MediaPipe 检测单只手的 21 个关键点；
- 计算：
    * 掌宽 / 掌长；
    * 卷曲度 curl ∈ [0,1]；
    * 侧度 side ∈ [0,1]；
    * 掌心/手背朝向 palm_front ∈ [0,1]；
- 通过一次标定获得掌宽/掌长两条通道的距离 Zw / Zl；
- 根据 curl + side + palm_front 融合成最终距离 Z_final；
- 对 Z_final 做中值 + EMA 滤波，得到稳定的掌根距离 Z_disp（米）；
- 额外计算 21 点骨骼上的 20 条方向向量（单位向量）；
- 通过 UDP 把掌根位置 + 深度 + 骨骼方向向量发给 Unity。

按键：
  q  / ESC  退出
  c         标定（正对摄像头、手张开，采样若干帧后会在终端里让你输入真实距离）
  r         重置标定
"""

import cv2
import time
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from collections import deque
import socket
import json

WIN_NAME = "HandDepth"

# UDP 相关配置（Unity 端监听相同 IP/端口）
UDP_IP = "127.0.0.1"
UDP_PORT = 5065

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# EMA 平滑系数
EMA_ALPHA = 0.35

# 骨骼拓扑：from_idx -> to_idx
BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小拇指
]


@dataclass
class CalibState:
    k_w: float | None = None
    k_l: float | None = None
    w_ref_open: float | None = None
    l_ref_open: float | None = None
    sampling: bool = False
    samples_w: list = field(default_factory=list)
    samples_l: list = field(default_factory=list)


@dataclass
class RuntimeState:
    z_hist: deque = field(default_factory=lambda: deque(maxlen=7))
    z_ema: float | None = None


def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))


def l2(p1, p2):
    """二维欧氏距离"""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def lm2px(lm, w, h):
    """把单个 landmark 从归一化坐标转为像素坐标 (x, y)"""
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def compute_face_sign(landmarks):
    """
    判断手掌是掌心朝相机还是手背朝相机。
    返回一个 face_sign ∈ [-1, +1]：
        正负只代表方向，具体哪边是掌心靠试验/经验定死。
    landmarks: result.multi_hand_landmarks[0].landmark
    """
    p0 = np.array([landmarks[0].x,  landmarks[0].y,  landmarks[0].z],  dtype=np.float32)
    p5 = np.array([landmarks[5].x,  landmarks[5].y,  landmarks[5].z],  dtype=np.float32)
    p17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z], dtype=np.float32)

    u = p5 - p0
    v = p17 - p0
    n = np.cross(u, v)  # 手掌底那条“扇形”的法向量

    camera_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    s = float(np.dot(n, camera_dir))

    denom = (np.linalg.norm(n) * np.linalg.norm(camera_dir) + 1e-6)
    cos_theta = s / denom
    cos_theta = max(-1.0, min(1.0, cos_theta))

    return cos_theta


def compute_palm_width_and_length(landmarks, img_w, img_h):
    """
    掌宽：食指 MCP(5) 到 小指 MCP(17) 的直线距离
    掌长：中指三节 + 中指根到掌根的总长度：0-9, 9-10, 10-11, 11-12
    """
    pts = {}
    idxs = [0, 5, 9, 10, 11, 12, 17]
    for i in idxs:
        pts[i] = lm2px(landmarks[i], img_w, img_h)

    palm_width = l2(pts[5], pts[17])

    seg1 = l2(pts[0], pts[9])
    seg2 = l2(pts[9], pts[10])
    seg3 = l2(pts[10], pts[11])
    seg4 = l2(pts[11], pts[12])
    palm_length = seg1 + seg2 + seg3 + seg4

    return palm_width, palm_length


def compute_curl(landmarks, img_w, img_h):
    """
    卷曲度 curl ∈ [0,1]：
    用 “掌根到中指指尖的直线距离 / 中指路径长度” 这个比值来表示伸直程度，
    然后反过来映射到 [0,1] 作为卷曲度。
    """
    pts = {}
    idxs = [0, 9, 10, 11, 12]
    for i in idxs:
        pts[i] = lm2px(landmarks[i], img_w, img_h)

    path_len = (
        l2(pts[0], pts[9])
        + l2(pts[9], pts[10])
        + l2(pts[10], pts[11])
        + l2(pts[11], pts[12])
    )
    straight = l2(pts[0], pts[12])

    if path_len < 1e-6:
        return 0.0

    ratio = straight / path_len  # 伸直时 ~1，弯曲时 < 1

    curl_raw = (1.0 - ratio) / 0.5
    curl = clamp(curl_raw, 0.0, 1.0)
    return curl


def compute_side(palm_width_px, palm_length_px, calib: CalibState, gain=1.5):
    """
    用“掌宽/掌长”的比例计算侧度，跟远近解耦。
    """
    if (calib.w_ref_open is None or calib.l_ref_open is None or
        calib.w_ref_open < 1e-3 or calib.l_ref_open < 1e-3 or
        palm_length_px < 1e-3):
        return 0.0

    ar_ref = calib.w_ref_open / calib.l_ref_open
    ar_cur = palm_width_px / palm_length_px

    ratio = ar_cur / ar_ref   # 理想情况：1.0
    ratio = max(min(ratio, 1.0), 0.0)

    side_raw = 1.0 - ratio      # ratio=1 -> 0, ratio=0.5 -> 0.5
    side = side_raw * gain      # 放大一点，让侧向更敏感

    return clamp(side, 0.0, 1.0)


def fuse_depth(Zw, Zl, curl, side, palm_front,
               beta=1.0,      # 卷曲对掌宽的加权强度
               k_curl=0.15,   # 正对+掌心+完全握拳时最多减 15%
               side_gain=1.8, # 把 side 放大一点，让侧向更“敏感”
               alpha_side=1.5 # side 对掌长权重的强化系数
               ):

    if Zw is None and Zl is None:
        return None, 0.0, 0.0
    if Zw is None:
        return Zl, 0.0, 1.0
    if Zl is None:
        return Zw, 1.0, 0.0

    curl = clamp(float(curl), 0.0, 1.0)
    side = clamp(float(side), 0.0, 1.0)
    palm_front = clamp(float(palm_front), 0.0, 1.0)

    side_eff = clamp(side * side_gain, 0.0, 1.0)

    g_front = 1.0 - side_eff   # 正面对程度
    g_side  = side_eff         # 侧向程度（已经被加强了）

    w_w_raw = g_front * (1.0 + beta * curl)
    w_l_raw = g_front * (1.0 - curl) + alpha_side * g_side

    w_sum = max(w_w_raw + w_l_raw, 1e-6)
    w_w = w_w_raw / w_sum
    w_l = w_l_raw / w_sum

    Z_mix = w_w * Zw + w_l * Zl

    corr = 1.0 - k_curl * g_front * curl * palm_front
    Z_final = Z_mix * corr

    return Z_final, w_w, w_l


def compute_palm_center_px(landmarks, img_w, img_h):
    idxs = [0, 5, 17]  # wrist, index_mcp, pinky_mcp
    pts = [lm2px(landmarks[i], img_w, img_h) for i in idxs]
    c = np.mean(pts, axis=0)
    return int(c[0]), int(c[1])


def draw_hud(img,
             curl, side, Zw, Zl, Z_final_raw, Z_disp,
             w_w, w_l, calib: CalibState, palm_front,
             anchor=None):
    h, w, _ = img.shape

    # 文本起始位置：如果给了锚点，就贴在手附近；否则左上角
    if anchor is None:
        x0, y0 = 10, 30
    else:
        ax, ay = anchor
        x0 = int(ax + 10)                # 稍微往右一点
        y0 = int(max(20, ay - 40))       # 放在手的上方一点，别挡住手

    y = y0
    dy = 22

    def put(line: str):
        nonlocal y
        cv2.putText(
            img,
            line,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += dy

    # 可以根据需要裁剪信息，这里保留核心调试量
    put(f"Z_disp: {Z_disp if Z_disp is not None else -1:6.2f}")
    if calib.k_w is None or calib.k_l is None:
        put("Calib: NOT SET (c = calibrate)")
    else:
        put("Calib: OK (r = reset, c = recal)")

    if calib.sampling:
        put(f"Sampling... {len(calib.samples_w)} frames")


def compute_bone_directions(landmarks):
    """
    根据 21 个 MediaPipe 关键点的归一化坐标，计算每条骨骼的单位方向向量。
    返回列表 bones，其中每个元素为:
      {
        "id": int,      # 从 0 开始的骨头索引
        "from": int,    # 起点关节 id
        "to": int,      # 终点关节 id
        "dir": [dx, dy, dz]  # 单位向量
      }
    """
    coords = [
        (float(lm.x), float(lm.y), float(lm.z))
        for lm in landmarks
    ]
    bones = []
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
        bones.append({
            "id": bi,
            "from": a,
            "to": b,
            "dir": [dirx, diry, dirz],
        })
    return bones


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    calib = CalibState()
    state = RuntimeState()
    last_t = time.time()

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - last_t
            fps = 1.0 / dt if dt > 0 else 0.0
            last_t = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            palm_width = 0.0
            palm_length = 0.0
            curl = 0.0
            side = 0.0
            Zw = None
            Zl = None
            Z_final_raw = None
            Z_disp = state.z_ema
            w_w = 0.0
            w_l = 0.0
            palm_front = 0.0

            # UDP 要发的内容占位
            hands_out = []

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                lms = hand_lms.landmark

                # 画手
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                palm_cx, palm_cy = compute_palm_center_px(lms, w, h)

                palm_width, palm_length = compute_palm_width_and_length(lms, w, h)
                curl = compute_curl(lms, w, h)
                side = compute_side(palm_width, palm_length, calib)

                face_sign = compute_face_sign(lms)           # [-1,1]
                palm_front = 0.5 * (face_sign + 1.0)        # 映射到 [0,1]
                palm_front = 1.0 - clamp(palm_front, 0.0, 1.0)

                if calib.k_w is not None and palm_width > 1e-3:
                    Zw = calib.k_w / palm_width
                if calib.k_l is not None and palm_length > 1e-3:
                    Zl = calib.k_l / palm_length

                Z_final_raw, w_w, w_l = fuse_depth(Zw, Zl, curl, side, palm_front)

                if Z_final_raw is not None:
                    state.z_hist.append(Z_final_raw)
                    Z_med = float(np.median(state.z_hist))

                    if state.z_ema is None:
                        state.z_ema = Z_med
                    else:
                        state.z_ema = EMA_ALPHA * Z_med + (1.0 - EMA_ALPHA) * state.z_ema

                    Z_disp = state.z_ema
                else:
                    Z_disp = None

                # 标定采样逻辑
                if calib.sampling:
                    if palm_width > 1e-3 and palm_length > 1e-3:
                        calib.samples_w.append(palm_width)
                        calib.samples_l.append(palm_length)

                        if len(calib.samples_w) >= 50:
                            calib.sampling = False
                            w_med = float(np.median(calib.samples_w))
                            l_med = float(np.median(calib.samples_l))
                            print("=" * 60)
                            print("标定采样结束。请保持刚才的姿态/距离不动。")
                            print(f"中位数掌宽:  {w_med:.2f} px")
                            print(f"中位数掌长:  {l_med:.2f} px")
                            d_real = float(input("请输入此时掌根到摄像头的真实距离(米): "))
                            calib.w_ref_open = w_med
                            calib.l_ref_open = l_med
                            calib.k_w = d_real * w_med
                            calib.k_l = d_real * l_med
                            calib.samples_w.clear()
                            calib.samples_l.clear()
                            print("标定完成：k_w=%.4f, k_l=%.4f" % (calib.k_w, calib.k_l))
                            print("=" * 60)

                # ========== 计算骨骼方向向量并打包成 UDP JSON ==========
                wrist_lm = lms[0]
                wrist_nx = float(wrist_lm.x)
                wrist_ny = float(wrist_lm.y)
                wrist_nz = float(wrist_lm.z)
                wrist_px = int(wrist_nx * w)
                wrist_py = int(wrist_ny * h)

                bones_out = compute_bone_directions(lms)

                hand_dict = {
                    "hand_index": 0,
                    "wrist": {
                        "pixel": {"x": wrist_px, "y": wrist_py},
                        "normalized": {
                            "x": wrist_nx,
                            "y": wrist_ny,
                            "z": wrist_nz,
                        },
                        "depth_m": None if Z_disp is None else float(Z_disp),
                    },
                    "bones": bones_out,
                }
                hands_out.append(hand_dict)

                # 在掌心附近画 HUD
                draw_hud(frame, curl, side, Zw, Zl, Z_final_raw, Z_disp,
                         w_w, w_l, calib, palm_front,
                         anchor=(palm_cx, palm_cy))
            else:
                # 没检测到手时，还是画一个基础 HUD
                draw_hud(frame, curl, side, Zw, Zl, Z_final_raw, Z_disp,
                         w_w, w_l, calib, palm_front,
                         anchor=None)

            # 发送 UDP
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

            cv2.imshow(WIN_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                calib = CalibState()
                state = RuntimeState()
                print("标定已重置。")
            elif key == ord('c'):
                # 开始新一轮标定
                calib.sampling = True
                calib.samples_w.clear()
                calib.samples_l.clear()
                print("=" * 60)
                print("开始标定：")
                print("请将一只手掌 **完全张开、正对摄像头**，保持不动。")
                print("会自动采样大约 50 帧，结束后在终端里输入真实距离。")
                print("=" * 60)

    cap.release()
    cv2.destroyAllWindows()
    sock.close()


if __name__ == "__main__":
    main()
