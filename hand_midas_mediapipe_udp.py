# file: hand_midas_mediapipe_udp.py
# 推荐环境 (CPU 也可跑):
#   pip install opencv-python mediapipe numpy
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install timm
#
# 功能摘要：
# 1) MediaPipe 每帧检测手部关键点；
# 2) MiDaS 小模型在“后台线程”做单目深度估计（异步 + 可降频），并与 MediaPipe z 融合；
# 3) 按 'c' 进入稳健标定：自动采样多帧 fused(wrist)，输入真实距离(米) -> 得到 scale；
# 4) 输出 JSON 通过 UDP 发给 Unity，包含 landmarks 以及每只手的 wrist_z_m / palm_center_z_m；
# 5) 屏幕 HUD 显示 FPS、scale、第一只手的 wrist 距离(米)。

import os
# 限制底层线程，避免 CPU 线程争用（要在导入 numpy/torch 之前设置更稳）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import time
import json
import socket
import numpy as np
import threading
import queue

import mediapipe as mp

# ---- 尝试加载 torch / timm（若失败将自动退化为“纯 MediaPipe”模式） ----
USE_MIDAS = True
try:
    import torch
    import torch.nn.functional as F
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(1)  # 减少 CPU 线程争用
except Exception as e:
    print("[WARN] 未能加载 torch，切换到纯 MediaPipe 模式：", repr(e))
    USE_MIDAS = False

# ---------------- MiDaS 加载（小模型） ----------------
def load_midas(device):
    # 使用 MiDaS_small （输入侧 256，速度更快）
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = transforms.small_transform
    return midas, midas_transform

# 将 midas_transform 的输出规整为 [B,C,H,W] 的 torch.Tensor
def to_bchw_on_device(transformed, device):
    if isinstance(transformed, (tuple, list)):
        tensor = transformed[0]
    elif isinstance(transformed, dict):
        tensor = transformed.get("image", next(iter(transformed.values())))
    else:
        tensor = transformed

    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    if tensor.ndim == 3:      # (C,H,W)
        tensor = tensor.unsqueeze(0)  # -> (1,C,H,W)
    elif tensor.ndim == 4:    # (B,C,H,W)
        pass
    else:
        raise RuntimeError(f"[MiDaS] Unexpected ndim: {tensor.ndim}, shape={tuple(tensor.shape)}")

    return tensor.to(device)

# 前景：MiDaS 后台线程，异步计算 depth_map（HxW, numpy float32 in [0,1]）
class MidasWorker:
    def __init__(self, expected_size_hw):
        self.enabled = USE_MIDAS
        self.device = None
        self.model = None
        self.transform = None
        self.queue = queue.Queue(maxsize=1)    # 仅保留最新帧，丢弃旧帧，避免延迟累积
        self.shared = {"depth": None, "ts": 0.0}
        self.stop_flag = False
        self.thread = None
        self.expected_size = expected_size_hw  # (H, W) 用于上采样至摄像头分辨率
        self.frame_skip = 3                    # 可调：后台只处理每第 N 次送来的帧

    def start(self):
        if not self.enabled:
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model, self.transform = load_midas(self.device)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def submit(self, frame_rgb):
        """主线程调用：提交一帧 RGB（H,W,3 uint8），只保留最新。"""
        if not self.enabled:
            return
        # 清空旧帧，放入新帧（非阻塞）
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queue.put_nowait(frame_rgb)
        except queue.Full:
            pass

    def latest_depth(self):
        """主线程读取：返回最新的 depth_map（HxW np.float32 [0,1]）或 None。"""
        return self.shared["depth"]

    def _run(self):
        cnt = 0
        H, W = self.expected_size
        while not self.stop_flag:
            try:
                frame = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cnt += 1
            # 降频：只处理每 N 帧
            if (cnt % self.frame_skip) != 0:
                continue

            # 变换与前向
            try:
                # transform 期望 RGB；transform 内部会 resize 到模型输入尺寸
                transformed = self.transform(frame)
                input_tensor = to_bchw_on_device(transformed, self.device)

                with torch.no_grad():
                    pred = self.model(input_tensor)
                    # 兼容各种输出形状
                    if pred.ndim == 4 and pred.shape[1] == 1:
                        pred = pred[:, 0, :, :]    # (B,H,W)
                    elif pred.ndim == 4:
                        pred = pred.mean(dim=1)    # (B,H,W)
                    pred = pred[0]                 # (H_out, W_out)

                    # 上采样到摄像头分辨率 (H,W)
                    pred = F.interpolate(
                        pred.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()

                depth = pred.detach().cpu().numpy().astype(np.float32)
                # 归一化为 [0,1]，只要相对深度即可
                dmin, dmax = depth.min(), depth.max()
                depth = (depth - dmin) / (dmax - dmin + 1e-9)

                self.shared["depth"] = depth
                self.shared["ts"] = time.time()

            except Exception as e:
                # 出错则禁用 MiDaS，回退纯 MediaPipe
                print("[WARN] MiDaS 线程出错，切换纯 MediaPipe：", repr(e))
                self.enabled = False
                self.shared["depth"] = None
                break

# ---------------- UDP ----------------
UDP_IP = "127.0.0.1"
UDP_PORT = 5065
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------------- MediaPipe Hands ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ---------------- 摄像头 ----------------
cap = cv2.VideoCapture(0)
# 可按需降低分辨率以提速（640x480 足够演示）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ---------------- MiDaS 后台线程启动 ----------------
midas_worker = MidasWorker(expected_size_hw=(H, W))
midas_worker.start()

# ---------------- 标定（scale） ----------------
# 交互逻辑：按 'c' 开始采集 CALIB_N 帧 wrist 的 fused 值；自动取中位数，然后提示输入真实距离(米)，算出 scale
scale = None
CALIB_N = 12
calib_collecting = False
calib_vals = []

print("按 'c' 启动多帧标定，按 'q' 退出。标定将采集多帧 wrist 的 fused 值，然后提示你输入真实距离(米)。")

# ---------------- 平滑设置 ----------------
# 对每个关键点的 fused 做中值平滑（抑制抖动）
from collections import deque
K_SMOOTH = 7
fused_history = {}  # key: (hand_idx, lm_id) -> deque

def smooth_fused(hand_idx, lm_id, val):
    key = (hand_idx, lm_id)
    dq = fused_history.get(key)
    if dq is None:
        dq = deque(maxlen=K_SMOOTH)
        fused_history[key] = dq
    dq.append(val)
    return float(np.median(np.array(dq)))

# ---------------- 小工具 ----------------
PALM_CENTER_IDS = [0, 5, 9, 13, 17]  # WRIST + MCPs

def sample_depth_at(px, py, depth_map):
    if depth_map is None:
        return None
    h, w = depth_map.shape
    x = int(np.clip(px, 0, w - 1))
    y = int(np.clip(py, 0, h - 1))
    # 3x3 中值
    x0 = max(0, x - 1); x1 = min(w - 1, x + 1)
    y0 = max(0, y - 1); y1 = min(h - 1, y + 1)
    patch = depth_map[y0:y1 + 1, x0:x1 + 1]
    return float(np.median(patch))

# 融合：优先用 MiDaS（相对值）+ MediaPipe 的 -z，缺哪项就用另一项
W_MIDAS = 0.6
W_MPZ   = 0.4

def fused_depth(mp_z_norm, midas_rel):
    if (midas_rel is not None) and (mp_z_norm is not None):
        return W_MPZ * mp_z_norm + W_MIDAS * midas_rel
    elif midas_rel is not None:
        return midas_rel
    else:
        return mp_z_norm if mp_z_norm is not None else 0.0

# ---------------- 主循环 ----------------
p_time = time.time()
frame_count = 0

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_count += 1

        # 镜像，方便人像交互
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 提交给 MiDaS 后台（异步），主线程不等待
        if USE_MIDAS and midas_worker.enabled:
            midas_worker.submit(frame_rgb)

        # 取最新深度图（可能为 None）
        depth_map = midas_worker.latest_depth() if midas_worker.enabled else None

        # MediaPipe 每帧跑（快）
        res = hands.process(frame_rgb)

        data = {"timestamp": time.time(), "hands": []}

        # 处理手部关键点
        if res.multi_hand_landmarks:
            for hi, hand_lmks in enumerate(res.multi_hand_landmarks):
                lm_list = []
                palm_px_sum = 0.0
                palm_py_sum = 0.0
                palm_count = 0

                # 遍历 21 个关键点
                for i, lm in enumerate(hand_lmks.landmark):
                    px = int(lm.x * W)
                    py = int(lm.y * H)
                    mp_z = -lm.z  # MediaPipe z（朝向相机为正的相对值）
                    md = sample_depth_at(px, py, depth_map)  # MiDaS 相对深度（0..1）
                    fused_rel = fused_depth(mp_z, md)
                    fused_s = smooth_fused(hi, i, fused_rel)

                    if i in PALM_CENTER_IDS:
                        palm_px_sum += px; palm_py_sum += py; palm_count += 1

                    lm_list.append({
                        "id": i,
                        "px": px, "py": py,
                        "fused": fused_s,  # 相对深度（已平滑）
                    })

                    # 画点（调试）
                    cv2.circle(frame_bgr, (px, py), 2, (0, 255, 0), -1)

                # 计算掌心近似点（WRIST+四个MCP的平均）
                if palm_count > 0:
                    palm_cx = int(palm_px_sum / palm_count)
                    palm_cy = int(palm_py_sum / palm_count)
                else:
                    palm_cx, palm_cy = lm_list[0]["px"], lm_list[0]["py"]

                # 采样掌心深度（相对）
                palm_md = sample_depth_at(palm_cx, palm_cy, depth_map)
                # 用 WRIST 的 mp_z 作为掌根参考
                wrist_fused = next((l["fused"] for l in lm_list if l["id"] == 0), lm_list[0]["fused"])
                # 掌心 fused（与 wrist 融合保持一致，这里直接采样 MiDaS 并与 wrist 的 mp_z 融合也可）
                palm_fused = palm_md if palm_md is not None else wrist_fused

                # 写入 hand 结构
                hand_obj = {
                    "hand_index": hi,
                    "landmarks": lm_list,
                    "palm_center_pxpy": [palm_cx, palm_cy],
                }

                # 若已标定：写米制距离
                if scale is not None:
                    for l in hand_obj["landmarks"]:
                        l["z_m"] = l["fused"] * scale
                    hand_obj["wrist_z_m"] = wrist_fused * scale
                    hand_obj["palm_center_z_m"] = palm_fused * scale

                data["hands"].append(hand_obj)

                # 在画面上显示 wrist/palm 距离
                if scale is not None:
                    cv2.putText(
                        frame_bgr,
                        f"H{hi} wrist:{hand_obj['wrist_z_m']:.2f}m palm:{hand_obj['palm_center_z_m']:.2f}m",
                        (10, 60 + 20 * hi),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )

                # 画掌心点
                cv2.circle(frame_bgr, (palm_cx, palm_cy), 4, (0, 255, 255), -1)

        # ---- 标定逻辑 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # 开始/重置采样
            calib_collecting = True
            calib_vals = []
            print(f"[Calib] 开始采集 {CALIB_N} 帧 wrist fused 值，请保持掌根在目标位置...")
        elif key == ord('q'):
            break

        # 收集标定样本
        if calib_collecting and res.multi_hand_landmarks:
            # 取第一只手的 wrist
            lm0 = data["hands"][0]["landmarks"][0] if data["hands"] else None
            if lm0 is not None:
                calib_vals.append(lm0["fused"])
                cv2.putText(frame_bgr, f"[Calib] {len(calib_vals)}/{CALIB_N}", (10, H - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            if len(calib_vals) >= CALIB_N:
                calib_collecting = False
                fused_med = float(np.median(np.array(calib_vals)))
                print(f"[Calib] 样本中位数 fused = {fused_med:.6f}")
                try:
                    s_in = input("[Calib] 输入掌根(WRIST)到相机的真实距离(米)：").strip()
                    real_d = float(s_in)
                    if fused_med <= 1e-9:
                        print("[Calib] 融合值异常，标定失败。")
                    else:
                        scale = real_d / fused_med
                        print(f"[Calib] 标定完成：scale = {scale:.12f}，验证: fused_med*scale ≈ {fused_med*scale:.4f} m")
                except Exception as e:
                    print("[Calib] 输入无效，标定取消：", repr(e))

        # ---- UDP 发送（仅在已标定时发送米制距离）----
        if scale is not None and data["hands"]:
            try:
                sock.sendto(json.dumps(data).encode("utf-8"), (UDP_IP, UDP_PORT))
            except Exception:
                pass

        # ---- HUD / FPS ----
        now = time.time()
        fps = 1.0 / (now - p_time + 1e-9)
        p_time = now
        cv2.putText(frame_bgr, f"FPS:{int(fps)} scale:{'%.6g'%scale if scale is not None else 'None'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Depth Fusion (CPU friendly)", frame_bgr)

finally:
    midas_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
