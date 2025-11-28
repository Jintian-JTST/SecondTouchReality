# combined_server.py
"""
一个进程做四件事：
- hand_udp: 摄像头 + Mediapipe + UDP 发给 Unity
- 回调里：根据 payload 提取 pinch，发串口给 Arduino
- text_infer_server: 文本 AI server，Unity 来连
"""

import threading
import time

import serial
from serial.serialutil import SerialException

import tools.hand_udp
from tools.text_infer_server import TextInferHandler, ThreadedTCPServer, HOST, PORT
from tools.arduino_udp_receive import send_to_arduino, extract_pinch_data
SERIAL_PORT = 'COM9'
BAUD_RATE = 9600
ser = None
last_pinch_state = None


def on_hand_payload(payload):
    global last_pinch_state, ser
    current_pinch_states = extract_pinch_data(payload)
    current_pinch = any(current_pinch_states) if current_pinch_states else False

    if current_pinch != last_pinch_state:
        cmd = "1" if current_pinch else "0"

        if ser is not None:
            ok = send_to_arduino(ser, cmd)
            if ok:
                time.sleep(0.05)

        last_pinch_state = current_pinch
        timestamp = payload.get("timestamp", time.time())
        num_hands = len(current_pinch_states)
        pinch_count = sum(1 for s in current_pinch_states if s)
        state_text = "PINCH" if current_pinch else "RELEASE"
        print(f"[{timestamp:.3f}] Hands: {num_hands}, Pinching: {pinch_count}, State: {state_text}")


def start_hand_thread():
    tools.hand_udp.on_payload = on_hand_payload
    t = threading.Thread(target=tools.hand_udp.main, daemon=True)
    t.start()
    print("[combined] Hand tracking thread started.")
    return t


def main():
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[combined] Serial port {SERIAL_PORT} opened at {BAUD_RATE} baud")
        time.sleep(2)
    except SerialException as e:
        print(f"[combined] Failed to open serial port: {e}")
        ser = None
    start_hand_thread()
    with ThreadedTCPServer((HOST, PORT), TextInferHandler) as server:
        print(f"[combined] Text server listening on {HOST}:{PORT}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("[combined] Shutting down...")

    if ser is not None and ser.is_open:
        ser.close()
        print("[combined] Serial port closed")


if __name__ == "__main__":
    main()
