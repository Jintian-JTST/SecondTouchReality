# SecondTouchReality Overview

### **SecondTouchReality is an evolution of OneTouchReality**

SecondTouchReality is a derivative of a **“VR hand-gesture based teaching-object system”**.
Its focus is:

**Natural-language → object generation**
* **pinch-gesture grabbing in 3D**
* **single-channel servo feedback of the pinch state**,

with a more modular and extensible architecture.

It’s a small but reasonably complete **end-to-end system** that keeps the whole chain while “extracting the skeleton” and making it lightweight:

* **Python side**: hand tracking + depth estimation + text classification model
* **Unity side**: 3D hand skeleton reconstruction + grabbing interaction + camera control
* **Hardware side**: Arduino / servo / glove (interface reserved)

Wired together as one pipeline:

```
Camera → Python understands your hand + your sentence → Unity generates an interactive 3D scene → drives hardware feedback
```

The repo is still at prototype stage, but it already shows a full loop of:

```text
Perception → Semantics → Interaction → Hardware
```

---

## 1. System Overview

From an end-user point of view, the system does three main things:

1. **Understand your hand**

   * Python uses MediaPipe Hands to detect 21 hand keypoints.
   * It computes palm width, palm length, finger curl, side pose, palm/backs-of-hand orientation.
   * A one-time calibration converts palm width / length into real-world wrist-to-camera distance (meters), with filtering.
   * It packs the wrist 3D position, 20 bone direction vectors and the `pinch` state into JSON and sends it to Unity over UDP.

2. **Understand your words**

   * Unity pops up a dialog where you type an English description (e.g. `"a small red apple"`).
   * A lightweight text model on Python side uses `HashingVectorizer + SGDClassifier` for multi-class classification, and outputs a discrete label (e.g. `"101"`).
   * A TCP connection returns `"101"` to Unity.

3. **Let you grab things**

   * Unity uses `HandFromVectors` to reconstruct the 3D hand skeleton from the UDP JSON, drawing it with spheres and lines.
   * `PinchGrabBall` lets you pinch objects in the scene so they follow your hand.
   * `HandOrbitCamera` lets you rotate / zoom the camera with pinch + hand movement when you’re not grabbing anything.
   * `ModelLibrary` / `RuntimeModelLoader` load the corresponding 3D model (prefab or GLB) based on the label.

4. **Directory Structure**

```text
SecondTouchReality/
├── README.md                  # English / mixed-language readme (overall design)
├── README_CHN.md              # Chinese readme (overall design)
├── requirements.txt           # Python dependencies
├── text_model.pkl             # Trained text classification model
├── main.py                    # combined_server, one-click to start the whole pipeline
├── .gitignore
├── unecessary/                # Old scripts or unused resources
├── tools/
│   ├── hand_easy.py           # Hand distance estimation + calibration logic
│   ├── hand_udp.py            # Multi-hand + bone vectors + pinch → UDP JSON
│   ├── arduino_udp_receive.py # Simple bridge: UDP → Arduino
│   ├── text_infer_server.py   # Text model inference TCP server
│   ├── run_model.py           # Load text_model.pkl, provide CLI inference
│   └── __pycache__/           # Python cache
├── test/
│   ├── collect_data.py        # Collect “description text + label” data
│   ├── clean_dataset.py       # Clean JSONL dataset
│   ├── train_model.py         # Train text classification model and output text_model.pkl
│   ├── text_object_dataset.jsonl
│   ├── cleaned_text_object_dataset.jsonl
│   └── object_models_csv.csv  # Text labels ↔ model ID mapping table
├── Game/                      # Unity demo project (can be opened directly)
│   ├── SampleScene.unity      # Main scene
│   ├── models/                # Several glb models (apple, banana, bowl, etc.)
│   └── Scripts/               # Main C# scripts
│       ├── HandFromVectors.cs
│       ├── HandOrbitCamera.cs
│       ├── PinchGrabBall.cs
│       ├── ModelLibrary.cs
│       ├── RuntimeModelLoader.cs
│       └── TextQueryClient_TMP.cs
└── ...
```

**Root directory**

* `main.py`: Recommended entry script. Opens the camera, starts the UDP → Unity hand data stream, registers the `on_payload` callback to map pinch state to serial commands, and starts the text inference server.
* `requirements.txt`: All Python dependencies; usually installed via `pip install -r requirements.txt`.
* `text_model.pkl`: Trained by `test/train_model.py`, used to map natural language descriptions to object labels.

**`tools/` – runtime tools layer**

* `hand_easy.py`: Encapsulates distance calibration and filtering logic; used by `hand_udp.py` and other scripts.

* `hand_udp.py`:

  * Uses MediaPipe, supports multiple hands.
  * Outputs wrist depth, bone directions, pinch state.
  * Sends JSON packets to Unity via UDP (default `127.0.0.1:5065`).
  * Allows external `on_payload` callback.

* `text_infer_server.py`: Starts a `socketserver`-based multithreaded TCP server, uses `infer_once` from `run_model.py` to call the text model in memory.

* `arduino_udp_receive.py`: Simplified bridge program; reads hand JSON over UDP, cares only about `hands[].pinch`, detects state changes and sends `'0'/'1'` to the Arduino serial port.

**`test/` – data & model playground**

* `collect_data.py`: Interactive CLI tool to quickly collect training data.
* `clean_dataset.py`: Filters out samples containing Chinese, keeps only clean English text + labels, outputs `cleaned_text_object_dataset.jsonl`.
* `train_model.py`: Trains a model on the cleaned data and saves it as `text_model.pkl` for the main program.

**`Game/` – Unity demo**

* `HandFromVectors.cs`: UDP client; parses JSON from Python, reconstructs joint positions and visualizes them with spheres and lines.
* `PinchGrabBall.cs`: Turns any 3D object into a “pinchable” object, handling grab/follow/release and smooth motion.
* `HandOrbitCamera.cs`: Uses pinch to control camera rotation and zoom.
* `ModelLibrary.cs`: Maintains a name → GameObject dictionary and provides `ShowModelByLabel(label)` for directly using text inference results.
* `TextQueryClient_TMP.cs`: TextMeshPro-based text input client, talks to the Python text server.
* `RuntimeModelLoader.cs`: Loads GLB models by index from `StreamingAssets/models` at runtime to expand the model library.

---

## 2. Tech Stack

### 2.1 Python side

* Python 3.x
* OpenCV (`cv2`) – camera capture + HUD drawing
* MediaPipe Hands – hand keypoint detection
* NumPy – vector operations, statistics (median, EMA, etc.)
* scikit-learn – text features (`HashingVectorizer`) + linear classifier (`SGDClassifier`)
* joblib – serialize model + label encoder (`text_model.pkl`)
* socket / socketserver – UDP + TCP communication
* pyserial – serial communication with Arduino

Main Python files:

* `hand_udp.py` – main hand tracking + UDP streaming
* `hand_easy.py` – depth estimation demo / debugging
* `collect_data.py` / `clean_dataset.py` / `train_model.py` / `run_model.py` – text model data & training toolchain
* `text_infer_server.py` – text inference TCP server
* `main.py` – combines hand tracking + text server + serial bridge into a single process
* `arduino_udp_receive.py` – alternative: standalone UDP → serial bridge

### 2.2 Unity / C# side

* Unity 202x
* C# scripts:

  * `HandFromVectors.cs` – UDP receiver + hand skeleton reconstruction + GUI tuning
  * `PinchGrabBall.cs` – grab logic for objects
  * `HandOrbitCamera.cs` – orbit/zoom camera around a scene target
  * `ModelLibrary.cs` – treat children/prefabs as a “model dictionary”
  * `RuntimeModelLoader.cs` – dynamic `.glb` loading with GLTFast
  * `TextQueryClient.cs` (class `TextQueryClient_TMP`) – Unity-side TCP client + UI for text
* TextMeshPro – input and text display
* GLTFast – runtime loading of .glb / .gltf models

### 2.3 Hardware side

* Arduino (Uno / Nano, etc.)
* One or more simple servos (for demo)
* Very simple serial protocol: send one ASCII char per update, e.g. `'0'` / `'1'`.

---

## 3. Typical Run Flow

A typical workflow (basically what you’re doing now) is:

1. **Start Python side first** (`main.py`).
2. In the camera window that pops up, press **`c`** to calibrate, `r` to reset, `q` to quit.
3. **Then open the Unity project**, and load the `Skin` scene (or your own demo scene).
4. In Unity, click **Play**:

   * The 3D skeleton hand follows your real hand.
   * Pinch to rotate the camera or grab objects.
   * Enter a sentence in the dialog box and the system will generate the corresponding 3D model in front of you.

Details follow.

---

### 3.1 Configure Python Environment

1. Create a virtual environment in the project root (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the camera is accessible by OpenCV / MediaPipe.

---

### 3.2 Start the combined server `main.py`

In the project root:

```bash
python main.py
```

You’ll get:

* A camera preview window with a HUD (FPS, calibration status, etc.).
* In the background:

  * UDP hand data server (for Unity).
  * Text TCP server (listening on `127.0.0.1:9009`).
  * Serial port (if an Arduino is connected).

In the camera window:

* Press **`c`**:

  * Open your palm, face the camera, keep still; it samples about 50 frames.
  * The terminal asks you for the real wrist-to-camera distance (meters), e.g. `0.45`.
  * It uses the median palm width/length to compute `k_w` / `k_l`, which are later used to estimate Z.

* Press **`r`**: reset calibration.

* Press **`q`**: exit Python side.

After calibration, HUD text changes from “Calib: NOT SET” to something like “Calib: OK”.

---

### 3.3 Open the Unity scene

1. Open the project in Unity. The main demo scene is typically `Skin.unity`.
   The scene should contain:

   * An object with `HandFromVectors`:

     * `listenPort = 5065` (must match Python).
     * `targetCamera` set (usually the main camera).

   * The main camera with `HandOrbitCamera` attached.

   * A node with `ModelLibrary`; its children are model templates (their names usually match label names).

   * A UI Canvas with `TextQueryClient_TMP` attached, pointing to the TMP input field and buttons.

2. Click **Play**:

   * You’ll see 3D hand bones (spheres + lines) in the camera view.
   * When you pinch (thumb + index):

     * If a `PinchGrabBall` object is nearby, it gets grabbed and follows your hand.
     * If nothing is grabbed, `HandOrbitCamera` interprets pinch as camera control—moving your hand rotates the view.

3. In the UI, click the button to open the dialog, enter an English description, for example:

   ```text
   a green apple
   ```

   and click confirm:

   * Unity sends this string to `127.0.0.1:9009`.
   * Python runs the text model and returns something like `102|0.93`.
   * `TextQueryClient_TMP` parses `label = "102"` and calls `ModelLibrary.ShowModelByLabel("102")`.
   * `ModelLibrary` / `RuntimeModelLoader` spawn the corresponding model in front of the camera and auto-attach `PinchGrabBall` so you can pinch it.

---

### 3.4 Connect Arduino

1. Flash a simple serial control sketch on Arduino, e.g.:

   * `Serial.begin(9600);`
   * `if (Serial.available()) char c = Serial.read();`
   * If `c == '1'` → servo turns to 45°, if `c == '0'` → servo returns to 0°.

2. In `main.py` or `arduino_udp_receive.py`, change `COM9` to your actual serial port.

3. Run Python:

   * When hand tracking works, `on_payload` or `arduino_udp_receive` will send `'1'` / `'0'` according to pinch state.
   * You should see the servo move as you pinch / release.

---

## 4. Main Python Scripts (by function)

### 4.1 Hand tracking: `hand_udp.py`

Summary:

* Opens the camera and uses MediaPipe Hands to detect multiple hands.

* For each frame it computes:

  * Palm width / palm length (pixels)
  * Finger curl `curl` (0–1)
  * Side pose `side` (0–1)
  * Palm/backs-of-hand orientation `palm_front`
  * Wrist depth `Z` (meters)
  * 20 bone direction vectors (unit vectors)
  * `pinch` (thumb + index pinch or not)

* Packs it into JSON and sends via UDP to:

  * Unity port (default 5065)
  * Arduino UDP bridge port (default 5066)

Key points:

* **Calibration logic**

  * Uses a `CalibState` struct to store sampled palm widths/lengths, `k_w`, `k_l`, etc.
  * When you press `c`, it samples for a while, then asks for the real distance and computes `k_w = d_real * w_med` and similar.
  * For depth estimation, it uses two channels `Z ≈ k_w / palm_width` and `Z ≈ k_l / palm_length`, then fuses them based on curl / side.

* **Pinch detection**

  * Typically checks the distance between `thumb_tip` and `index_tip`. If below threshold, it’s pinch.
  * Writes `"pinch": true/false` directly into the JSON.

* **`on_payload` callback** (registered in `main.py`)

  * Receives the whole JSON per frame, can count how many hands are pinching and do post-processing (e.g. serial output).

---

### 4.2 Depth estimation demo: `hand_easy.py`

This is for “pulling the depth estimation piece out” and playing with it separately, without UDP or Unity.

Important functions:

* `compute_palm_width_and_length(...)` – computes palm width & length in pixels given landmarks; used as depth proxies.
* `compute_curl(...)` – uses finger joint angles to determine whether the hand is open or in a fist.
* `compute_side(...)` – detects whether the hand faces the camera or is turned sideways.
* `fuse_depth(Zw, Zl, curl, side, palm_front, ...)` – fuses the two depth channels into a final `Z_final` with weighting and correction terms.
* `draw_hud(...)` – prints all intermediate values on the image for easier tuning and understanding.

---

### 4.3 Text model pipeline

* **Raw data**: `text_object_dataset.jsonl`
  Each line is a JSON record: `{"text": "...", "label": "101"}`, containing both Chinese and English.

* `collect_data.py`: interactively add data.

* `clean_dataset.py`: filters out samples containing Chinese characters, writes to `cleaned_text_object_dataset.jsonl`.

* `train_model.py`: trains / incrementally trains an `SGDClassifier` on the cleaned data, saves to `text_model.pkl`, and prints training metrics.

* `run_model.py`: tests the model on the command line, printing top-k labels + probabilities for given sentences.

* **`text_infer_server.py`**:

  * Loads the model once at startup.
  * Exposes a TCP server:

    * For each line of text it receives → runs inference → returns `"label|prob\n"`.

---

### 4.4 Combined entry: `main.py`

`main.py` glues three directions together:

1. **Hand tracking + UDP**: starts the tracking loop from `tools.hand_udp`.

2. **Serial bridge**: registers `on_payload(payload)`:

   * Counts whether any hand is pinching in the JSON.
   * If yes → `ser.write(b"1")`; otherwise → `ser.write(b"0")`.

3. **Text TCP server**: uses `TextInferHandler` + `ThreadedTCPServer` to listen on port 9009 for text from Unity.

So you only need to run `python main.py` to support Unity hand tracking + text-driven object generation + hardware feedback all at once.

---

### 4.5 UDP → serial bridge: `arduino_udp_receive.py`

If you don’t want to mix too much logic into `main.py`, you can run this script separately:

* Listens on UDP port 5066 for the same JSON as Unity.
* Parses the current pinch state.
* When the state changes, sends `'0'` or `'1'` to the Arduino serial port.
* Good for debugging the “hardware bridge” in isolation.

---

## 5. Unity Scripts

### 5.1 `HandFromVectors.cs` – UDP receiver + hand skeleton reconstruction

Responsibilities:

* Creates a `UdpClient` to listen on the given port (default 5065).

* Parses JSON from Python:

  * `wrist` pixel coordinates + normalized coords + `z_m` (depth)
  * 20 bone direction vectors (unit vectors)
  * `pinch` / `is_left` flags

* Uses:

  * Camera intrinsics (via Unity `Camera` projection).
  * Pre-configured bone lengths.

  to reconstruct 21 joint positions in Unity world space.

* Dynamically creates:

  * Sphere array `jointObjects` to visualize joints.
  * `LineRenderer` array `boneLines` to draw bones.

* Exposes API:

  * `TryGetJointPosition(handIndex, jointIndex, out Vector3 pos)`
  * `bool IsPinching(handIndex)`
  * `bool AnyHandPinching`

It also draws a GUI window in-scene that lets you:

* Adjust each bone’s length.
* Toggle debug options.
* See how many hands are active and their pinch states.

---

### 5.2 `PinchGrabBall.cs` – grabbing objects

Attach this script to any GameObject and assign a `handTracker`; the object becomes pinch-grabbable:

* When not yet grabbed:

  * Iterate all hands and check if any pinching hand has its control joint (`controlJointIndex`, index fingertip by default) within `grabDistance` of the object.
  * If so, treat that as “grabbed”, and record:

    * Which hand grabbed it: `grabbedHandIndex`
    * Initial offset from the follow joint to the object: `grabOffset`

* While grabbed:

  * If `usePhysics` is enabled:

    * Disable gravity on its `Rigidbody`, zero out velocity, and drive it by position interpolation.

  * If not using physics:

    * Directly use `Vector3.Lerp` to move `transform.position` toward the target, controlled by `followSmoothing`.

* `pinchReleaseGrace` protects against MediaPipe glitches:

  * Short pinch dropouts that immediately recover will not drop the object.
  * The object is released only if the pinch has been off for longer than the grace time.

The script has a static counter `grabbedCount`; other scripts (e.g. camera control) can query `AnyObjectGrabbed` to know if anything is currently being held.

---

### 5.3 `HandOrbitCamera.cs` – hand-driven camera

Attach this to the camera so that pinch gestures control the camera whenever nothing is grabbed:

* Choose a joint (default: index fingertip) as the control point.
* Record the hand position and yaw/pitch at the moment pinch starts.
* While pinch is held:

  * Map hand movement on screen to yaw / pitch.
  * Clamp pitch to avoid flipping behind the head.
  * Adjust radius `radius` based on zoom gestures or depth change to push/pull the camera.

Final camera update:

```csharp
orbitCamera.transform.position = pivot + dir.normalized * radius;
orbitCamera.transform.LookAt(pivot, Vector3.up);
```

---

### 5.4 `ModelLibrary.cs` – prefab library

Design: attach this script to an empty GameObject and put all model prefabs as its children:

* In `Awake()`:

  * Collect all child `GameObject`s into a `Dictionary<string, GameObject>` keyed by their names.
  * `SetActive(false)` on all children, treating them as templates.

* `ShowModelByLabel(string label)`:

  * Find the template by label.
  * If there’s already a displayed instance, hide or deactivate it.
  * Spawn the new object near `spawnAnchor.position + spawnOffset`.
  * Ensure it has `PinchGrabBall` attached and configure:

    * `handTracker`
    * `grabDistance`
    * `usePhysics`

With this, the label returned by the text model directly determines which prefab appears in the scene.

---

### 5.5 `RuntimeModelLoader.cs` – runtime GLB loading

Used to “load models on the fly instead of baking all of them into the scene”.

Main interfaces:

* `MakeFileName(int index)`:

  * Default is `101 → "101.glb"`, but you can replace this with more complex mapping (e.g. via a table).

* `LoadByIndexAsync(int index)`:

  * Builds a path (usually under `Application.streamingAssetsPath`).
  * Uses `GltfImport` to load the GLB.
  * Instantiates it as a `GameObject`.
  * If `currentInstance` exists, destroys the old one first.
  * Parents the new object under the loader and zeroes `localPosition` / `localRotation`.

* `LoadByIndex(int index)`:

  * Convenience wrapper: `_ = LoadByIndexAsync(index);` (ignores `await`).

If you have a batch of `.glb` files in StreamingAssets, you can directly map labels to filenames and truly load them on demand in Unity.

---

### 5.6 `TextQueryClient_TMP` – Unity text TCP client

Attach this to a UI GameObject; it uses TMP input + buttons to talk to the Python text service.

Flow:

1. `OpenDialog()`:

   * `dialogPanel.SetActive(true)` and focus the input field.

2. Button click → `OnClickSend()`:

   * Read user text from `descriptionInput.text`.
   * Start `SendQueryCoroutine(q)`.

3. Inside `SendQueryCoroutine`:

   * Use `TcpClient` to connect to `serverIp:serverPort` (default `127.0.0.1:9009`).
   * Send `q + "\n"` in UTF-8.
   * Block until one line of response is read.
   * Parse `<label>|<prob>`.
   * If `modelLibrary` is bound, call `modelLibrary.ShowModelByLabel(label)`.
   * Optionally show the prediction on a TMP text widget.

4. `OnDestroy()` closes the stream and socket.

---

## 6. Data & Protocols

### 6.1 UDP JSON (Python → Unity)

* Ports: 5065 (Unity) / 5066 (Arduino UDP bridge)
* Encoding: UTF-8 JSON
* Top-level fields: `timestamp`, `fps`, `num_hands`, `hands`
* Each hand contains:

  * `id` – hand index
  * `is_left` – whether it is a left hand
  * `wrist` – `{px, py, nx, ny, z_m}`
  * `bones` – list of `{from, to, vx, vy, vz}`
  * `pinch` – bool

### 6.2 Text TCP (Unity ↔ Python)

* Address: `127.0.0.1:9009`
* Request: one line of text + `\n`
* Response: `<label>|<probability>\n`

Example:

```text
a small red apple\n
→ 101|0.923\n
```

### 6.3 Serial (Python → Arduino)

* Baud rate: 9600
* Data: single-byte ASCII char

  * `'1'` – at least one hand is pinching
  * `'0'` – no hand is pinching

How Arduino interprets this is up to you; the example here is driving a servo to different angles.

---

## 7. Dataset & Object IDs

* `object_models_csv.csv`:

  * Maintains a table “object ID → English name → category”, e.g.:

    * `101, red apple, fruit`
    * `203, tomato, vegetable`
    * …

  * Used to align:

    * Text dataset labels
    * GLB filenames
    * Unity prefab names

* `text_object_dataset.jsonl` / `cleaned_text_object_dataset.jsonl`:

  * Can be extended over time to train stronger text models.
  * Current cleaning logic is: simply filter out Chinese samples and keep only English sentences.

---

## 8. Future Directions

From an engineering perspective, this repo already connects three “worlds”:

> Camera world → Algorithm world → Virtual world → (future) Hardware world

Possible extensions:

* **Semantic upgrades**

  * Replace the current simple `HashingVectorizer + SGDClassifier` with semantic retrieval or large-model embeddings.
  * Truly support mixed Chinese/English input so children can describe in Chinese and the system internally maps to English.

* **Gesture upgrades**

  * Beyond pinch, add more dynamic gestures like fist, pointing, waving.
  * Map gestures to different teaching interactions (select, confirm, delete, etc.).

* **Content generation**

  * Generate whole teaching levels at once instead of single objects.
  * Use the object CSV + text descriptions to generate contextual scenes for kids (kitchen, supermarket, classroom, …).

* **Hardware feedback**

  * More complex gloves, vibration motors, brake/force devices to “materialize” virtual objects in the real world.
  * Map Unity collisions and task completion events to multi-channel hardware feedback.

* **Online learning**

  * Log user sentences and chosen objects, and continue fine-tuning the text classifier.
  * Let the system gradually adapt to each user’s way of speaking.

---

## 9. From OneTouchReality to SecondTouchReality

OneTouchReality started as a full **“VR hand-gesture controlled robotic arm system”**:
Camera + MediaPipe for hand recognition → UDP sends 21 keypoints to Unity → Unity reconstructs the 3D hand and does collision detection → TCP sends “which finger, how much force” back to Python → Arduino controls 5 servos to turn virtual collisions into real pulling forces on the fingers.

1. **Vision pipeline preserved but cleaned up**

   * Python still uses MediaPipe for hand keypoints and pose estimation, but no longer dumps all 21 points into Unity. Instead it extracts:

     * Real-world wrist distance (meters)
     * 20 bone direction vectors
     * Discrete states like pinch / no pinch

   * The distance estimation module inherits OneTouchReality’s idea of “palm width / palm length dual channels + weighted fusion + median/EMA filtering”, but wraps it as `hand_easy.py`, with `c` for calibrate, `r` for reset, `q` for quit, so it’s easy to reuse in other projects.

2. **Interaction logic: from “robotic arm” to “teaching objects + grabbing + camera control”**

   * Unity side uses `HandFromVectors.cs` to receive JSON from Python, reconstruct 21 joint positions from bone directions and lengths, and draw a virtual hand with spheres and lines.

   * `PinchGrabBall.cs` implements “pinch to make it follow your hand”:

     * Among all pinching hands, find the one closest to the object as the “grabbing hand”.
     * Record the finger-to-object offset.
     * Update the model position based on joint position + offset with smoothing, using either rigidbody physics or direct interpolation.

   * `HandOrbitCamera.cs` turns pinch into a “general gesture mouse”:

     * Single-hand pinch → orbit the camera around the target.
     * Two-hand pinch → change camera radius (pinch-to-zoom).
     * While any object is grabbed by `PinchGrabBall`, camera temporarily stops reacting to avoid conflicts.

3. **From “collision-sensing glove” to “scene that understands language”**

   * In the original project, Unity detected which tags finger bones collided with, then sent which fingers and how much force to Python/Arduino.

   * In SecondTouchReality we shift the focus to **“text → object”**:

     * `collect_data.py` interactively gathers “description + label” pairs.
     * `clean_dataset.py` filters non-English samples, leaving only `text` / `label`.
     * `train_model.py` trains a lightweight multi-class model with `HashingVectorizer + SGDClassifier` and produces `text_model.pkl`.
     * `text_infer_server.py` opens a TCP service that receives descriptions from Unity and returns labels in real time.

   * On Unity side, `TextQueryClient_TMP.cs`:

     * Shows a TMP input dialog.
     * Sends user text over TCP to the Python text server.
     * On receiving a label, calls `ModelLibrary.ShowModelByLabel()` to activate the corresponding 3D model.
     * Displays the result in the UI and pops a short “success” panel.

4. **Hardware chain: from multi-servo cable tightening → single-channel pinch signal**

   * OneTouchReality’s end goal is 5-servo cable pulling to simulate fingertip haptics, requiring full mechanical design, cable management, springs and anti-twist structure.

   * In this derived project, we first **complete the signal chain**:

     * `hand_udp.py` marks each hand’s pinch state in every JSON frame.
     * `arduino_udp_receive.py` or `main.py` register a callback: when any hand transitions between “not pinching → pinching” or vice versa, they send a single character over serial: `'1'` means tighten, `'0'` means release.
     * Arduino runs a minimal servo sketch: `'1'` → 45°, `'0'` → 0°.

   * With this, you can tie one string to your fingertip and close the loop:

     > Camera → Python → Unity → Arduino → Finger

     to test latency, stability and safety first, then gradually scale to multi-servo and more complex mechanisms.

5. **Integration: `main.py` = one-click full pipeline**

   * `main.py` (essentially `combined_server.py`) runs four things in one process:

     1. Start `tools.hand_udp`: camera + MediaPipe + hand depth + pinch detection, send JSON to Unity via UDP.
     2. In `on_payload`, extract pinch state and send `'0'/'1'` to Arduino on edge changes.
     3. Start the text inference TCP server so Unity can request models by natural language.
     4. Manage serial lifecycle and exceptions (auto-retry on disconnection / safe close).

Conceptually, SecondTouchReality is a **teachable, extensible, fast-experiment mini trunk** carved out of the full OneTouchReality chain:

* Camera → hand geometry & depth
* Text description → object label
* Unity scene → grabbing / camera / teaching levels
* Pinch → single or multi-servo feedback

Later you can plug these modules back into a “full-size” force-feedback glove system, or treat this as a base for an **AI-driven interactive teaching platform**.

---

`SecondTouchReality` is essentially an **end-to-end playground prototype**: every module is simple enough to hack on freely, yet the chain is complete enough for you to experience the full loop from camera to virtual object to physical feedback.
