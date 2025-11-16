# file: hand_midas_mediapipe_udp.py
# CPU 也能跑的配置：
#   pip install opencv-python mediapipe numpy
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install timm

import os
# 减少 CPU 线程争用（需在导入数值库之前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import time
import json
import socket
import numpy as np
import threading
import queue
from collections import deque

import mediapipe as mp

# ---- 可选：尝试加载 torch / timm（失败则退化为纯 MediaPipe） ----
USE_MIDAS = True
try:
    import torch
    import torch.nn.functional as F
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(1)
except Exception as e:
    print("[WARN] 未能加载 torch，切换到纯 MediaPipe 模式：", repr(e))
    USE_MIDAS = False

# ---------------- MiDaS 加载（小模型） ----------------
def load_midas(device):
    # 使用 MiDaS_small：速度更快
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = transforms.small_transform
    return midas, midas_transform

def to_bchw_on_device(transformed, device):
    """把 midas_transform 的输出规整为 [B,C,H,W] 的 tensor，并移动到 device。"""
    if isinstance(transformed, (tuple, list)):
        tensor = transformed[0]
    elif isinstance(transformed, dict):
        tensor = transformed.get("image", next(iter(transformed.values())))
    else:
        tensor = transformed
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    elif tensor.ndim != 4:
        raise RuntimeError(f"[MiDaS] Unexpected ndim: {tensor.ndim}, shape={tuple(tensor.shape)}")
    return tensor.to(device)

class MidasWorker:
    """
    后台线程：异步计算 MiDaS raw 深度（不做逐帧 min-max 归一化）。
    仅保留最新帧；可设置 frame_skip 降频；输出 raw 相对深度（跨帧一致）。
    """
    def __init__(self, expected_size_hw):
        self.enabled = USE_MIDAS
        self.device = None
        self.model = None
        self.transform = None
        self.queue = queue.Queue(maxsize=1)
        self.shared = {"depth": None, "ts": 0.0}
        self.stop_flag = False
        self.thread = None
        self.expected_size = expected_size_hw  # (H, W)
        self.frame_skip = 3
        self._ema = None
        self.ema_alpha = 0.2  # 轻微 EMA 平滑

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
        """提交一帧 RGB（H,W,3），仅保留最新。"""
        if not self.enabled:
            return
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
            self.queue.put_nowait(frame_rgb)
        except queue.Full:
            pass

    def latest_depth(self):
        """返回最新 raw depth（HxW np.float32）或 None。"""
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
            if (cnt % self.frame_skip) != 0:
                continue

            try:
                transformed = self.transform(frame)
                input_tensor = to_bchw_on_device(transformed, self.device)

                with torch.no_grad():
                    pred = self.model(input_tensor)
                    if pred.ndim == 4 and pred.shape[1] == 1:
                        pred = pred[:, 0, :, :]    # (B,H,W)
                    elif pred.ndim == 4:
                        pred = pred.mean(dim=1)    # (B,H,W)
                    pred = pred[0]                 # (H_out, W_out)

                    pred = F.interpolate(
                        pred.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze()

                depth_raw = pred.detach().cpu().numpy().astype(np.float32)
                # 不做 per-frame min-max；可选 EMA 平滑
                if self.ema_alpha > 0.0:
                    if self._ema is None:
                        self._ema = depth_raw
                    else:
                        self._ema = (1.0 - self.ema_alpha) * self._ema + self.ema_alpha * depth_raw
                    out = self._ema
                else:
                    out = depth_raw

                self.shared["depth"] = out
                self.shared["ts"] = time.time()

            except Exception as e:
                print("[WARN] MiDaS 线程出错，切换纯 MediaPipe：", repr(e))
                self.enabled = False
                self.shared["depth"] = None
                break

# ---------------- 逆深度多点标定器 ----------------
class InvDepthCalibrator:
    """
    拟合：1/Z = p * f + q   ->   Z = 1 / (p*f + q)
    f 为融合后的“接近度”特征（越近越大）：来自 MiDaS raw & MediaPipe z 的加权。
    """
    def __init__(self):
        self.samples = []   # list[(f, Z)]
        self.p = None
        self.q = None

    def reset(self):
        self.samples.clear()
        self.p = None
        self.q = None

    def add_sample(self, f, Z):
        if f is None or Z is None or Z <= 1e-6:
            return False
        self.samples.append((float(f), float(Z)))
        return True

    def undo(self):
        if self.samples:
            self.samples.pop()

    def ready(self):
        return (self.p is not None) and (self.q is not None)

    def fit(self):
        """最小二乘：令 y=1/Z，则 y = p*f + q，至少 2 个点（建议 ≥3）。"""
        if len(self.samples) < 2:
            return False
        X = []
        y = []
        for f, Z in self.samples:
            X.append([f, 1.0])
            y.append(1.0 / max(Z, 1e-6))
        X = np.asarray(X, dtype=np.float64)   # (N,2)
        y = np.asarray(y, dtype=np.float64)   # (N,)
        XtX = X.T @ X
        Xty = X.T @ y
        try:
            theta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.p, self.q = float(theta[0]), float(theta[1])
        return True

    def map(self, f):
        """将特征 f 映射为距离 Z（米），带数值保护。"""
        if not self.ready():
            return None
        denom = self.p * float(f) + self.q
        if denom < 1e-6:
            denom = 1e-6
        return float(1.0 / denom)

    def __repr__(self) -> str:
        if not self.ready():
            return "InvDepthCalibrator(unfitted)"
        return f"InvDepthCalibrator(1/Z = {self.p:.6g} * f + {self.q:.6g} -> Z = 1/(p*f+q))"

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
# 降分辨率提速（可按需调整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ---------------- MiDaS 后台线程 ----------------
midas_worker = MidasWorker(expected_size_hw=(H, W))
midas_worker.start()

# ---------------- 标定器 & 提示 ----------------
calib = InvDepthCalibrator()
print("标定：将掌根(WRIST)停在一个已知距离处，按 'c' 采样并输入真实距离(米)；")
print("可在不同距离多次按 'c'（建议 ≥3），按 'f' 拟合；'u' 撤销；'x' 清空；'q' 退出。")

# ---------------- 平滑与融合 ----------------
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

PALM_CENTER_IDS = [0, 5, 9, 13, 17]  # WRIST + MCPs

def sample_depth_at(px, py, depth_map):
    if depth_map is None:
        return None
    h, w = depth_map.shape
    x = int(np.clip(px, 0, w - 1))
    y = int(np.clip(py, 0, h - 1))
    # 稳健：7x7 中值
    r = 3
    x0 = max(0, x - r); x1 = min(w - 1, x + r)
    y0 = max(0, y - r); y1 = min(h - 1, y + r)
    patch = depth_map[y0:y1 + 1, x0:x1 + 1]
    return float(np.median(patch))

# 融合权重（更偏重 MiDaS raw）
W_MIDAS = 0.8
W_MPZ   = 0.2

def fused_depth(mp_z_norm, midas_raw):
    if (midas_raw is not None) and (mp_z_norm is not None):
        return W_MPZ * mp_z_norm + W_MIDAS * midas_raw
    elif midas_raw is not None:
        return midas_raw
    else:
        return mp_z_norm if mp_z_norm is not None else 0.0

# ---------------- 主循环 ----------------
p_time = time.time()

try:
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # 镜像更自然
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 提交给 MiDaS 后台（异步）
        if USE_MIDAS and midas_worker.enabled:
            midas_worker.submit(frame_rgb)

        depth_map = midas_worker.latest_depth() if midas_worker.enabled else None

        # 处理手部关键点
        res = hands.process(frame_rgb)
        data = {"timestamp": time.time(), "hands": []}

        if res.multi_hand_landmarks:
            for hi, hand_lmks in enumerate(res.multi_hand_landmarks):
                lm_list = []
                palm_px_sum = 0.0
                palm_py_sum = 0.0
                palm_count = 0

                for i, lm in enumerate(hand_lmks.landmark):
                    px = int(lm.x * W)
                    py = int(lm.y * H)
                    mp_z = -lm.z  # MediaPipe 的相对深度（接近度取正）
                    md = sample_depth_at(px, py, depth_map)  # MiDaS raw
                    fused_rel = fused_depth(mp_z, md)
                    fused_s = smooth_fused(hi, i, fused_rel)

                    if i in PALM_CENTER_IDS:
                        palm_px_sum += px
                        palm_py_sum += py
                        palm_count += 1

                    lm_list.append({
                        "id": i,
                        "px": px, "py": py,
                        "fused": fused_s,
                    })

                    cv2.circle(frame_bgr, (px, py), 2, (0, 255, 0), -1)

                # 掌心像素位置
                if palm_count > 0:
                    palm_cx = int(palm_px_sum / palm_count)
                    palm_cy = int(palm_py_sum / palm_count)
                else:
                    palm_cx, palm_cy = lm_list[0]["px"], lm_list[0]["py"]

                # 掌心/掌根接近度
                palm_md = sample_depth_at(palm_cx, palm_cy, depth_map)
                wrist_fused = next((l["fused"] for l in lm_list if l["id"] == 0), lm_list[0]["fused"])
                palm_fused = palm_md if palm_md is not None else wrist_fused

                # 手对象：始终包含 fused；z_m 未标定时为 None
                hand_obj = {
                    "hand_index": hi,
                    "landmarks": lm_list,
                    "palm_center_pxpy": [palm_cx, palm_cy],
                    "wrist_fused": float(wrist_fused),
                    "palm_center_fused": float(palm_fused),
                    "wrist_z_m": None,
                    "palm_center_z_m": None,
                }

                # 若已拟合：映射为米，并写入 landmarks 的 z_m
                if calib.ready():
                    for l in hand_obj["landmarks"]:
                        l["z_m"] = calib.map(l["fused"])
                    hand_obj["wrist_z_m"] = calib.map(wrist_fused)
                    hand_obj["palm_center_z_m"] = calib.map(palm_fused)

                data["hands"].append(hand_obj)

                # HUD：未标定显示 fused，已标定显示米
                cv2.circle(frame_bgr, (palm_cx, palm_cy), 4, (0, 255, 255), -1)
                if calib.ready():
                    text = f"H{hi} wrist:{hand_obj['wrist_z_m']:.2f}m  palm:{hand_obj['palm_center_z_m']:.2f}m"
                else:
                    text = f"H{hi} wrist_fused:{wrist_fused:.3f}"
                cv2.putText(
                    frame_bgr, text,
                    (10, 60 + 20 * hi),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

        # ---- 键盘事件：多点标定 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if res.multi_hand_landmarks and len(res.multi_hand_landmarks) > 0 and data["hands"]:
                wrist_fused = data["hands"][0]["landmarks"][0]["fused"]
                try:
                    s_in = input("[Calib] 输入当前掌根(WRIST)到相机的真实距离(米)：").strip()
                    real_d = float(s_in)
                    if real_d <= 0:
                        print("[Calib] 距离必须为正。")
                    else:
                        calib.add_sample(wrist_fused, real_d)
                        print(f"[Calib] 已添加样本 #{len(calib.samples)}: f={wrist_fused:.6f}, Z={real_d:.3f} m")
                except Exception as e:
                    print("[Calib] 输入无效：", repr(e))
            else:
                print("[Calib] 画面中没有检测到手，无法采样。")

        elif key == ord('f'):
            ok = calib.fit()
            if ok:
                print(f"[Calib] 拟合完成：1/Z = {calib.p:.6g} * f + {calib.q:.6g} -> Z = 1/(p*f+q)")
                if len(calib.samples) >= 2:
                    errs = []
                    for f_val, Z in calib.samples:
                        Z_hat = calib.map(f_val)
                        errs.append(abs(Z_hat - Z))
                    mae = np.mean(errs)
                    print(f"[Calib] 样本 MAE ≈ {mae:.3f} m  （样本数 {len(calib.samples)}）")
            else:
                print("[Calib] 样本不足（至少 2 个）。请在不同距离多采几次 'c' 再按 'f'。")

        elif key == ord('u'):
            calib.undo()
            print(f"[Calib] 撤销，剩余样本数：{len(calib.samples)}")

        elif key == ord('x'):
            calib.reset()
            print("[Calib] 已重置标定。")

        elif key == ord('q'):
            break

        # ---- UDP 发送（已拟合才发 z_m）----
        if data["hands"]:
            try:
                sock.sendto(json.dumps(data).encode("utf-8"), (UDP_IP, UDP_PORT))
            except Exception:
                pass

        # ---- HUD / FPS ----
        now = time.time()
        fps = 1.0 / (now - p_time + 1e-9)
        p_time = now
        hud_right = f"p={calib.p:.3g}, q={calib.q:.3g}" if calib.ready() else "未标定"
        cv2.putText(frame_bgr, f"FPS:{int(fps)}  {hud_right}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Depth Fusion (inverse-depth calibration, CPU-friendly)", frame_bgr)

finally:
    midas_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
