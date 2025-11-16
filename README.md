# SecondTouchReality
This is the second edition for OneTouchReality. Namely after the project in Adventure X 2025, and aiming for Cambridge EduX Hackathon 2025.

# Mediapipe–Unity–Arduino Hand Control Demo (README)

> A cross-platform mini system: Python uses MediaPipe to detect hand landmarks and estimate depth → sends the data via UDP to Unity → Unity objects trigger actions and send commands back to Python via TCP → Python relays them to Arduino over serial.

---

## Overview

* **Hand landmarks + Z-depth approximation:**
  Detects both hands using MediaPipe, outputs normalized, pixel, and relative/visual/depth-in-meters (`z_m`) coordinates per frame. Press **c** to perform a real-world distance calibration. The script shows FPS and scale on HUD, and sends all data as JSON via UDP (`127.0.0.1:5065`).
* **TCP→Serial bridge:**
  A Python process listens on `127.0.0.1:8000` and writes incoming TCP text commands to a serial port (`COM5 @ 9600` by default). Commands are expected as `finger,angle\n`, which the bridge converts to `finger angle\n` for Arduino.
* **Unity demo:**

  * *Drag component:* Physically drags `Rigidbody` objects with the mouse, adjustable speed and max velocity.
  * *Collision sender:* When another object enters a trigger, the object changes color and sends a TCP message (e.g., a random angle) to the Python bridge.

---

## Project Structure

```
/python
  hand_two_hands_z_udp.py         # Hand tracking + Z-depth + UDP sender 
  tcp_to_serial_bridge.py         # TCP→Serial bridge (from provided .servo_five.py) 
/unity
  DraggableObjectController.cs    # Mouse drag physics controller 
  CollisionAndColorChanger.cs     # Trigger color change + TCP sender 
/arduino
  (user sketch)                   # Parses text commands and drives servos
```

---

## Requirements

### Python

* Python 3.9+
* Packages:

  ```bash
  pip install opencv-python mediapipe pyserial
  ```

  (Uses standard libs like `socket`, `json`, `time`, `sys`.)

### Unity

* Any Unity version with .NET 4.x scripting runtime (both example scripts rely on `Rigidbody`, `Collider`, `TcpClient`).

### Arduino

* Arduino IDE + board driver
* Default serial port: `COM5 @ 9600`.

---

## Quick Start

### 1 Start the TCP→Serial bridge

1. Open `tcp_to_serial_bridge.py` and adjust:

   ```python
   SERIAL_PORT = 'COM5'
   BAUD_RATE = 9600
   SERVER_HOST = "127.0.0.1"
   SERVER_PORT = 8000
   ```

   Run the script; it opens the serial port and listens for TCP clients.
2. The bridge expects messages like `finger,angle\n` and writes them as `finger angle\n`.
   Example: `index,120\n` → `index 120\n`.

> The handler only forwards lines containing a comma split into two fields.

### 2 Run the hand-tracking script

1. Run `hand_two_hands_z_udp.py`. It opens your webcam and shows both hands; the HUD displays the wrist (`id=0`) depth values and FPS.
2. Press **c** to calibrate: place your wrist at a known distance (in meters), enter that number in the console. The script computes `scale = real_d / z_vis`, then reports distances in meters (`z_m = z_vis * scale`).
3. Each frame’s data is sent as JSON via UDP to `127.0.0.1:5065`.

### 3 Open Unity and connect

* **Dragging demo:**
  Attach `DraggableObjectController` to a `Rigidbody` object. Adjust `dragSpeed`, `maxVelocity`, etc. The script uses raycasts and screen-to-world conversions for smooth physics dragging.
* **Trigger sender:**
  Attach `CollisionAndColorChanger` to a trigger object. Set `pythonServerIp` and `pythonServerPort` to the bridge. When another object enters, it turns red and sends a random angle string via TCP; on exit, it reverts color.

> Note: The provided Unity script currently sends **only one number (angle)**, while the bridge expects `finger,angle`. Either modify Unity to send `"finger,angle\n"`, or update the bridge to interpret single-number messages.

---

## Data Protocols

### UDP (Python → Unity)

* Target: `127.0.0.1:5065`.
* Example JSON frame:

  ```json
  {
    "timestamp": 1731780000.123,
    "fps": 30.2,
    "hands": [
      {
        "hand_index": 0,
        "landmarks": [
          {
            "id": 0,
            "normalized": {"x": 0.51, "y": 0.42, "z": -0.12},
            "pixel": {"x": 326, "y": 202},
            "z_rel": -0.12,
            "z_vis": 0.12,
            "z_m": 0.38
          }
        ]
      }
    ]
  }
  ```

  **Fields:**

  * `normalized`: MediaPipe’s normalized coordinates (`z` negative toward camera).
  * `z_vis`: Flipped sign (`-z_rel`) for “larger = closer.”
  * `z_m`: Real-world distance after calibration.
  * Wrist landmark is `id=0`.

### TCP (Unity → Python → Arduino)

* Recommended message: `finger,angle\n` (e.g., `index,120\n`). The bridge writes it to serial as `index 120\n`.
* The included Unity script sends only a single integer angle (0–180). Adapt it for full multi-servo use.

---

## Calibration (Z to Meters)

1. Run the Python hand-tracking script.
2. Press **c**, hold your wrist at a known distance (e.g., `0.40` m).
3. Enter that distance. The script calculates `scale = D / z_vis` and displays it on HUD; all following frames use this to compute `z_m`.

> If `z_vis <= 0`, calibration fails; reposition and retry.

---

## Unity Components

### DraggableObjectController

* Requires a `Rigidbody`. Uses raycasts and `ScreenToWorldPoint` to set a target position and moves the object with velocity capped at `maxVelocity`.
* Gravity is disabled while dragging; re-enable it manually after release if needed.

### CollisionAndColorChanger

* Connects to the Python bridge (`localhost:8000` by default) on `Start()`.
* `OnTriggerEnter`: changes color to red and sends a random angle (0–180).
* `OnTriggerExit`: restores original color.
* `targetTag` is present but unused—extend as needed.

---

## Arduino Sketch Concept

* Reads lines over serial at `9600` baud.
* Parses commands like `finger angle` and maps them to servo pins and angles.
* Ensure common ground, proper voltage, and servo angle limits.

---

## Troubleshooting

* **UDP/TCP not connecting:** Check firewall and port numbers; ensure both Unity and Python configs match.
* **Serial open error:** Verify `COM` port and exclusivity—only one process can use it.
* **No hand detected:** Ensure the camera isn’t busy and lighting is good (default 640×480).
* **Calibration incorrect:** Re-measure distance accurately; ensure wrist faces camera.
* **Protocol mismatch:** Unity may send only angles, while bridge expects `finger,angle`; unify the format or adjust parsing.

---

## License

No explicit license is included; add your own LICENSE before publishing or redistributing.

---

## Future Improvements

* Parse UDP JSON inside Unity to drive 3D hand visualization or interaction depth.
* Unify command protocol with a GUI for per-finger servo mapping.
* Add safe speed interpolation and calibration workflow on Arduino side.
