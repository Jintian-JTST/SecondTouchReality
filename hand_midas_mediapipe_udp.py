# file: hand_midas_mediapipe_udp.py
# pip install opencv-python mediapipe torch torchvision timm
import cv2, time, json, socket, numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F

# ---------- MiDaS loader (use small model for speed) ----------
# this function tries to load MiDaS small via torch.hub
def load_midas(device):
    model_type = "MiDaS_small"  # faster, acceptable quality
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return midas, transform

# ---------- UDP config ----------
UDP_IP = "127.0.0.1"
UDP_PORT = 5065
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------- MediaPipe init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
midas_net, midas_transform = load_midas(device)

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------- calibration (scale) ----------
scale = None
print("按 'c' 做标定（把手或手腕放到已知距离并输入距离，单位米）。按 'q' 退出。")

# ---------- simple temporal filter (per-keypoint) ----------
alpha = 0.6
prev_kp = {}  # dict hand_id -> list of (x,y,z)

def sample_depth_at(px, py, depth_map):
    # px,py integer pixel coords; depth_map as numpy float32, same size as frame
    h, w = depth_map.shape
    x = int(np.clip(px, 0, w-1))
    y = int(np.clip(py, 0, h-1))
    # median over small 3x3 window
    x0 = max(0, x-1); x1 = min(w-1, x+1)
    y0 = max(0, y-1); y1 = min(h-1, y+1)
    patch = depth_map[y0:y1+1, x0:x1+1]
    return float(np.median(patch))

# ---------- main loop ----------
p_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # compute MiDaS depth for current frame
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_pil = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)  # already RGB
    # transform and run
    input_tensor = midas_transform(input_img).to(device)  # returns CHW tensor
    with torch.no_grad():
        prediction = midas_net(input_tensor.unsqueeze(0))
        prediction = F.interpolate(prediction.unsqueeze(1),
                                   size=(H, W),
                                   mode="bilinear",
                                   align_corners=False).squeeze()
        depth_map = prediction.cpu().numpy()
    # normalize depth_map to positive values (MiDaS outputs relative values)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-9)

    data = {"timestamp": time.time(), "hands": []}

    if res.multi_hand_landmarks:
        for hi, hand_landmarks in enumerate(res.multi_hand_landmarks):
            lmks = []
            for i, lm in enumerate(hand_landmarks.landmark):
                px = int(lm.x * W)
                py = int(lm.y * H)
                mp_z = -lm.z  # MediaPipe z (positive toward camera)
                # MiDaS depth sampling (relative)
                midas_d = sample_depth_at(px, py, depth_map)
                # fuse: weighted combination (weights tuneable)
                # Since both are relative, we'll fuse and later apply global scale
                fused_rel = 0.4 * mp_z + 0.6 * midas_d
                lmks.append({"id": i, "px": px, "py": py, "mp_z": mp_z, "midas_d": midas_d, "fused": fused_rel})
            data["hands"].append({"hand_index": hi, "landmarks": lmks})

            # draw for debug
            for l in lmks:
                cv2.circle(frame, (l["px"], l["py"]), 2, (0,255,0), -1)

    # calibration key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # use wrist of first detected hand as calibration point if exists
        if data["hands"]:
            wrist = data["hands"][0]["landmarks"][0]  # id 0 is wrist
            print("Detected wrist fused value:", wrist["fused"])
            s_in = input("输入 wrist 到相机的真实距离 (米)，回车：")
            try:
                real_d = float(s_in.strip())
                if wrist["fused"] <= 1e-6:
                    print("警告：采样 fused 非法")
                else:
                    scale = real_d / wrist["fused"]
                    print("标定完成：scale =", scale)
            except:
                print("标定失败，输入非法")
        else:
            print("没有检测到手，无法标定")

    # apply scale if available and send UDP
    if scale is not None and data["hands"]:
        # convert fused to meters and also back-project to camera coords if you have intrinsics
        for h in data["hands"]:
            for lm in h["landmarks"]:
                Z = lm["fused"] * scale  # meters
                lm["z_m"] = Z
                # optionally compute X,Y if you know fx,fy,cx,cy
        try:
            sock.sendto(json.dumps(data).encode('utf-8'), (UDP_IP, UDP_PORT))
        except Exception:
            pass

    # fps debug
    now = time.time()
    fps = 1.0 / (now - p_time + 1e-9)
    p_time = now
    cv2.putText(frame, f"FPS:{int(fps)} scale:{scale}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    cv2.imshow("Hand MiDaS Fusion", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
