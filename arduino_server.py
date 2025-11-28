"""
非常简单的 Arduino 控制服务器：

- 独占 Arduino 串口（比如 COM5，9600 波特率）
- 本地开一个 TCP 端口（默认 127.0.0.1:8000）
- Unity 或其他程序连上来，发 '0' 或 '1'（可以带换行）
- 本脚本把命令直接写到串口，Arduino 收到后控制舵机

依赖：
    pip install pyserial
"""

import socket
import sys
import threading
import time

import serial

# =====  =====
SERIAL_PORT = "COM5"  
BAUDRATE = 9600

TCP_HOST = "127.0.0.1"
TCP_PORT = 8000
# =================================


def open_serial():
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            print(f"[arduino] Serial opened on {SERIAL_PORT} ({BAUDRATE} baud)")
            return ser
        except Exception as e:
            print(f"[arduino] Failed to open serial {SERIAL_PORT}: {e}")
            print("[arduino] Retry in 3 seconds...")
            time.sleep(3)


def handle_client(conn, addr, ser):
    print(f"[arduino] Client connected from {addr}")
    try:
        with conn:
            buf = ""
            while True:
                data = conn.recv(1024)
                if not data:
                    break

                buf += data.decode("utf-8", errors="ignore")

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    cmd = line[0]  # 拿第一个字符
                    if cmd in ("0", "1"):
                        try:
                            ser.write(cmd.encode("utf-8"))
                            ser.flush()
                            print(f"[arduino] Sent to serial: {cmd!r}")
                        except Exception as e:
                            print(f"[arduino] Serial write error: {e}")
                    else:
                        print(f"[arduino] Unknown cmd from {addr}: {line!r}")

    finally:
        print(f"[arduino] Client disconnected: {addr}")


def main():
    ser = open_serial()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((TCP_HOST, TCP_PORT))
    sock.listen(5)

    print(f"[arduino] TCP server listening on {TCP_HOST}:{TCP_PORT}")
    print("[arduino] Waiting for Unity to connect...")

    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr, ser), daemon=True)
            t.start()

    except KeyboardInterrupt:
        print("\n[arduino] KeyboardInterrupt, shutting down...")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
