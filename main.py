# 把 hand_udp 的摄像头 + UDP 发送
# 和 text_infer_server 的 TCP 文本服务
# 合并到一个进程里

import threading
from hand_udp import main as hand_main
from text_infer_server import TextInferHandler, ThreadedTCPServer, HOST, PORT

def start_hand_thread():
    t = threading.Thread(target=hand_main, daemon=True)
    t.start()
    print("[combined] Hand tracking thread started.")
    return t

def main():
    start_hand_thread()
    with ThreadedTCPServer((HOST, PORT), TextInferHandler) as server:
        print(f"[combined] Text server listening on {HOST}:{PORT}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("[combined] Shutting down...")

if __name__ == "__main__":
    main()
