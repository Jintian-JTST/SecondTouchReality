#arduino_udp_receive.py
import socket
import serial
import json
import time
from serial.serialutil import SerialException

SERIAL_PORT = 'COM9'
BAUD_RATE = 9600
UDP_IP = "127.0.0.1"
UDP_PORT = 5065
MAX_UDP_PACKET_SIZE = 8192  # 限制数据包大小

def send_to_arduino(ser, data):
    """发送单个字符到Arduino"""
    if not ser.is_open:
        try:
            ser.open()
        except SerialException:
            print("Cannot open serial port")
            return False
    try:
        # 发送单个字符，不需要换行符
        ser.write(data.encode('utf-8'))
        print(f"Sent to Arduino: '{data}'")
        return True
    except SerialException as e:
        print(f"Serial write error: {e}")
        return False

def extract_pinch_data(full_data):
    """从手势数据中提取捏合状态"""
    pinch_states = []
    
    # 更安全的数据提取
    if isinstance(full_data, dict):
        hands = full_data.get("hands", [])
        for hand in hands:
            if isinstance(hand, dict):
                is_pinching = hand.get("pinch", False)
                pinch_states.append(bool(is_pinching))
    
    return pinch_states

def safe_json_parse(data):
    """安全的JSON解析"""
    try:
        return json.loads(data.decode('utf-8', errors='ignore'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"JSON parse error: {e}")
        return None

def main():
    # 初始化串口
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial port {SERIAL_PORT} opened at {BAUD_RATE} baud")
        time.sleep(2)  # 等待Arduino复位
    except SerialException as e:
        print(f"Failed to open serial port: {e}")
        return

    # 初始化UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 设置socket选项，增加缓冲区大小
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except OSError as e:
        print(f"Failed to bind to {UDP_IP}:{UDP_PORT}: {e}")
        if ser and ser.is_open:
            ser.close()
        return
    
    sock.settimeout(0.1)
    
    print(f"Listening for hand data on {UDP_IP}:{UDP_PORT}...")
    print("Press Ctrl+C to stop")

    last_pinch_state = None
    
    try:
        while True:
            try:
                # 接收手势数据，限制数据包大小
                data, addr = sock.recvfrom(MAX_UDP_PACKET_SIZE)
                
                # 检查数据包大小
                if len(data) > MAX_UDP_PACKET_SIZE:
                    print(f"Warning: Large packet received ({len(data)} bytes)")
                    continue
                
                # 安全解析JSON
                hand_data = safe_json_parse(data)
                if hand_data is None:
                    continue
                
                # 提取捏合状态
                current_pinch_states = extract_pinch_data(hand_data)
                
                # 只要有任何一只手在捏合，就认为是捏合状态
                current_pinch = any(current_pinch_states) if current_pinch_states else False
                
                # 只有当状态变化时才发送
                if current_pinch != last_pinch_state:
                    # 发送单个字符命令
                    if current_pinch:
                        command = "1"  # 捏合
                    else:
                        command = "0"  # 放开
                    
                    # 发送到Arduino
                    if send_to_arduino(ser, command):
                        # 添加小延迟确保命令被处理
                        time.sleep(0.05)
                    
                    # 更新状态记录
                    last_pinch_state = current_pinch
                    
                    # 在控制台显示状态
                    timestamp = hand_data.get("timestamp", time.time())
                    num_hands = len(current_pinch_states)
                    pinch_count = sum(1 for state in current_pinch_states if state)
                    state_text = "PINCH" if current_pinch else "RELEASE"
                    print(f"[{timestamp:.3f}] Hands: {num_hands}, Pinching: {pinch_count}, State: {state_text}")
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error processing data: {e}")
                # 继续运行，不退出
                continue
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        if sock:
            sock.close()
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed")
        print("Program exited")

if __name__ == "__main__":
    main()