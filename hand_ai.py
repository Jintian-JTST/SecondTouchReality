# hand_depth_unified_min.py
import cv2, time, math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np, mediapipe as mp

# —— 基础窗口/开关 ——
WIN = "Hand Depth (Unified Minimal)"  # 窗口标题
MAX_NUM_HANDS = 1                     # 同时跟踪的手数（>1 会更慢）
DRAW_LANDMARKS = False                # 是否在画面上绘制 21 点骨架

# —— 标定（按 c 时会连续采样） ——
CALIB_SAMPLES = 30    # 标定采样帧数；更大更稳（噪声更小），但标定更慢
D_CM  = 40            # 标定时手到相机的真实距离（厘米）
W_CM  = 8.5             # 名义掌宽（厘米，对应 5↔17）
L_CM  = 17            # 名义掌长（厘米，当前实现用于“长通道”）

# —— 通道融合/质量权重 ——
S_VIS_TH   = 0.10     # s 的可见性阈值：低于该值的通道直接弃用（越大越严格）
AGREE_FRAC = 0.05     # 两通道相对差异阈值：小于它则加权平均；否则择权重大者
EMA_ALPHA  = 0.25     # 一阶低通(EMA)系数：大=更跟手、稳得少；小=更稳、滞后多
WEIGHT_EXP = 4.0      # 权重幂次：权重 = s^WEIGHT_EXP；幂越大，s 小的通道权重掉得越狠
WIDTH_BIAS = 1.3      # 掌宽通道的先验偏置：>1 偏向 Zw（更“刚”），<1 偏向 Zl
Z_MAX_CM   = 200.0    # 深度硬限幅（厘米），防止异常值炸表
# —— 卷曲感知权重（握拳降权） ——
CURL_K = 1.2        # 卷曲惩罚强度；越大，握拳时 Zl 权重掉得越狠
CURL_BOOST_W = 0.6  # 握拳时对掌宽通道的动态加权系数
# —— 侧向↔正面门控 + 卷曲权重 ——
S_W_LO = 0.60   # s_w 低于此值视为“侧向”
S_W_HI = 0.80   # s_w 高于此值视为“正面”
CURL_K = 2.7    # 正面时对掌长的卷曲惩罚强度（exp(-K*curl)）
CURL_BOOST_W = 0.8  # 正面时对掌宽的卷曲增益（1 + BOOST*curl）



# —— 时域滤波（“中值 → 限速 → EMA” 链） ——
MED_WIN = 7           # 中值滤波窗口（帧数，奇数更好）；大一点更抗尖峰
MAX_SLEW_CM_S = 30.0  # 限速器最大跟随速度（厘米/秒）；小更稳但更拖尾，大更跟手



mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def smoothstep(x, lo, hi):
    t = clip((x - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)  # C1 连续，过渡干净

def draw_text(img, txt, x, y, color=(255,255,255), scale=0.6):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def l2(p, q):
    p = np.asarray(p, np.float32); q = np.asarray(q, np.float32)
    return float(np.linalg.norm(p - q))

def fmt1(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"

def clip(v, lo, hi): return max(lo, min(hi, v))

@dataclass
class Measurements:
    w_px: Optional[float]; l_px: Optional[float]
    s_w: float; s_l: float
    p0: Tuple[int,int]; p5: Tuple[int,int]; p9: Tuple[int,int]; p17: Tuple[int,int]
    curl: float

@dataclass
class CalibState:
    collecting: bool = False
    q_w_px: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_l_px: deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_sw:   deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    q_sl:   deque = field(default_factory=lambda: deque(maxlen=CALIB_SAMPLES))
    f_w: Optional[float] = None
    f_l: Optional[float] = None

@dataclass
class RuntimeState:
    Z_ema_cm: Optional[float] = None
    fps: float = 0.0
    z_hist: deque = field(default_factory=lambda: deque(maxlen=MED_WIN))
    Z_slew_cm: Optional[float] = None

def extract_meas(hand_landmarks, img_w, img_h):
    if not hand_landmarks:
        return None
    lm = hand_landmarks[0].landmark

    pts_px  = np.array([(int(l.x * img_w), int(l.y * img_h)) for l in lm], dtype=np.int32)
    pts_3d  = np.array([(l.x, l.y, l.z) for l in lm], dtype=np.float32)  # z<0 朝相机

    p0, p5, p9, p10, p11, p12, p17 = pts_px[0], pts_px[5], pts_px[9], pts_px[10], pts_px[11], pts_px[12], pts_px[17]

    w_px = l2(p5, p17)
    l_px = (
        l2(p0,  p9)  +
        l2(p9,  p10) +
        l2(p10, p11) +
        l2(p11, p12)
    )

    def s_of(v):
        n3 = float(np.linalg.norm(v)) + 1e-6
        n2 = float(np.linalg.norm(v[:2]))
        return clip(n2 / n3, 0.0, 1.0)

    vw   = pts_3d[17] - pts_3d[5]
    s_w  = s_of(vw)

    v0 = pts_3d[9]  - pts_3d[0]
    v1 = pts_3d[10] - pts_3d[9]
    v2 = pts_3d[11] - pts_3d[10]
    v3 = pts_3d[12] - pts_3d[11]
    segs = [v0, v1, v2, v3]
    sum_xy = 0.0
    sum_3d = 0.0
    for v in segs:
        n3 = float(np.linalg.norm(v)) + 1e-6
        n2 = float(np.linalg.norm(v[:2]))
        sum_xy += n2
        sum_3d += n3

    chord3d = float(np.linalg.norm(pts_3d[12] - pts_3d[9]))
    chain3d = (float(np.linalg.norm(pts_3d[10] - pts_3d[9])) +
               float(np.linalg.norm(pts_3d[11] - pts_3d[10])) +
               float(np.linalg.norm(pts_3d[12] - pts_3d[11])) + 1e-6)
    straightness = clip(chord3d / chain3d, 0.0, 1.0)
    curl = 1.0 - straightness

    s_l = clip(sum_xy / (sum_3d + 1e-6)-curl*0.1, 0.0, 1.0)

    return Measurements(w_px, l_px, s_w, s_l, tuple(p0), tuple(p5), tuple(p9), tuple(p17), curl)

def start_collect(calib: CalibState):
    calib.collecting = True
    calib.q_w_px.clear(); calib.q_l_px.clear(); calib.q_sw.clear(); calib.q_sl.clear()

def finish_collect_and_compute_f(calib: CalibState):
    if len(calib.q_w_px)==0 or len(calib.q_l_px)==0: calib.collecting=False; return False
    w_med = float(np.median(list(calib.q_w_px))); l_med = float(np.median(list(calib.q_l_px)))
    sw_med = max(1e-3, float(np.median(list(calib.q_sw)))); sl_med = max(1e-3, float(np.median(list(calib.q_sl))))
    calib.f_w = (D_CM * w_med) / (W_CM * sw_med)
    calib.f_l = (D_CM * l_med) / (L_CM * sl_med)
    calib.collecting = False
    return True

def z_from_channel(f_pix, L_nom_cm, s, ell_px):
    if not f_pix or not ell_px or ell_px <= 1e-3 or s <= 1e-6: return None
    return clip((f_pix * L_nom_cm * s) / float(ell_px), 0.0, Z_MAX_CM)

def estimate_fused_Z(meas: Measurements, calib: CalibState):
    Zw = z_from_channel(calib.f_w, W_CM, meas.s_w, meas.w_px) if calib.f_w else None
    Zl = z_from_channel(calib.f_l, L_CM, meas.s_l, meas.l_px) if calib.f_l else None

    if Zw is None and Zl is None: return None, None, None
    if Zl is None: return Zw, Zw, None
    if Zw is None: return Zl, None, Zl

    g = smoothstep(meas.s_w, S_W_LO, S_W_HI)

    ww = (meas.s_w ** WEIGHT_EXP) if meas.s_w > S_VIS_TH else 0.0
    ww *= WIDTH_BIAS * (1.0 + CURL_BOOST_W * g * meas.curl)

    wl = (meas.s_l ** WEIGHT_EXP) if meas.s_l > S_VIS_TH else 0.0
    wl *= math.exp(-CURL_K * g * meas.curl)

    if ww == 0.0 and wl == 0.0:
        Z = Zw if meas.s_w >= meas.s_l else Zl
        return Z, Zw, Zl

    rel = abs(Zw - Zl) / max(1e-6, 0.5 * (Zw + Zl))
    if rel < AGREE_FRAC and (ww > 0 and wl > 0):
        Z = (ww * Zw + wl * Zl) / (ww + wl)
        return Z, Zw, Zl

    if abs(ww - wl) < 0.1 * (ww + wl + 1e-6):
        Z = Zw
    else:
        Z = Zw if ww >= wl else Zl
    return Z, Zw, Zl


def main():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1280, 720)
    cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    calib = CalibState(); state = RuntimeState(); last_t = time.time()

    with mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_NUM_HANDS,
                        model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            h, w = frame.shape[:2]

            now = time.time()
            dt = max(1e-3, now - last_t)
            last_t = now
            state.fps = (0.9*state.fps + 0.1*(1.0/dt)) if state.fps>0 else (1.0/dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            meas = extract_meas(res.multi_hand_landmarks, w, h) if res.multi_hand_landmarks else None

            if DRAW_LANDMARKS and res.multi_hand_landmarks:
                for hlm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS,
                                           mp_styles.get_default_hand_landmarks_style(),
                                           mp_styles.get_default_hand_connections_style())

            if calib.collecting:
                draw_text(frame, "[Calib] Keep steady at D", 10, 24, (180,230,255))
                if meas is not None:
                    if meas.w_px and meas.w_px>1: calib.q_w_px.append(meas.w_px); calib.q_sw.append(meas.s_w)
                    if meas.l_px and meas.l_px>1: calib.q_l_px.append(meas.l_px); calib.q_sl.append(meas.s_l)
                draw_text(frame, f"w {len(calib.q_w_px)}/{CALIB_SAMPLES} | l {len(calib.q_l_px)}/{CALIB_SAMPLES}", 10, 48, (180,230,255))
                if len(calib.q_w_px)>=CALIB_SAMPLES and len(calib.q_l_px)>=CALIB_SAMPLES:
                    okf = finish_collect_and_compute_f(calib)
                    draw_text(frame, f"[Done] f_w={fmt1(calib.f_w)} f_l={fmt1(calib.f_l)}" if okf else "[Fail]", 10, 72, (180,255,200) if okf else (0,0,255))

            Z=Zw=Zl=None
            if meas is not None and (calib.f_w or calib.f_l):
                Z, Zw, Zl = estimate_fused_Z(meas, calib)
                if Z is not None:
                    state.z_hist.append(Z)
                    Z_med = float(np.median(state.z_hist))

                    prev = state.Z_slew_cm if state.Z_slew_cm is not None else Z_med
                    max_step = MAX_SLEW_CM_S * dt
                    step = clip(Z_med - prev, -max_step, max_step)
                    Z_slew = prev + step
                    state.Z_slew_cm = Z_slew

                    state.Z_ema_cm = Z_slew if state.Z_ema_cm is None else (EMA_ALPHA*Z_slew + (1-EMA_ALPHA)*state.Z_ema_cm)


            if dt>0: state.fps = 0.9*state.fps + 0.1*(1.0/dt) if state.fps>0 else (1.0/dt)

            y=24
            draw_text(frame, f"FPS {state.fps:5.1f}", 10, y); y+=24
            if calib.f_w or calib.f_l: draw_text(frame, f"f_w={fmt1(calib.f_w)} f_l={fmt1(calib.f_l)}", 10, y, (200,255,200)); y+=24
            else: draw_text(frame, "Press 'c' to calibrate", 10, y, (80,230,255)); y+=24
            if meas is not None: draw_text(frame, f"s_w={meas.s_w:.2f} s_l={meas.s_l:.2f}", 10, y); y+=24
            if Z is not None:
                draw_text(frame, f"Z={fmt1(Z)}cm  Zw={fmt1(Zw)}  Zl={fmt1(Zl)}", 10, y, (255,220,180)); y+=24
                if state.Z_ema_cm is not None: draw_text(frame, f"Z_ema={fmt1(state.Z_ema_cm)}cm", 10, y, (255,220,180)); y+=24

            if meas is not None:
                cx,cy = meas.p0
                s_avg = clip(0.5*(meas.s_w+meas.s_l),0.0,1.0)
                color = (int(255*(1-s_avg)), int(200*s_avg), int(255*s_avg))
                cv2.circle(frame,(cx,cy),16,(0,0,0),3,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),16,color,-1,cv2.LINE_AA)
                if Z is not None: draw_text(frame, f"{fmt1(Z)} cm", cx+20, cy+6, (255,255,255), 0.6)

            draw_text(frame, "Keys: c=calib  r=reset  q=quit", 10, h-10, (220,220,220))
            cv2.imshow(WIN, frame)

            k = cv2.waitKey(1) & 0xFF
            if k==ord('q'): break
            elif k==ord('c'): start_collect(calib)
            elif k==ord('r'): calib = CalibState(); state = RuntimeState()

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
