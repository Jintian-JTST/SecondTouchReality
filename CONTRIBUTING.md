# Contributing to SecondTouchReality

Thank you for your interest in contributing to **SecondTouchReality**.

SecondTouchReality is a small but complete “end-to-end” system that connects:

> **Camera → Python hand tracking & text model → Unity 3D interaction → Arduino / servo haptic feedback**

This document explains how to set up the environment, how to work on the codebase, and how to propose changes.

---

## 1. Scope and Goals

Contributions are welcome in (but not limited to) the following areas:

- Improving hand tracking, depth estimation, and gesture robustness on the Python side.
- Extending Unity interactions (grabbing, placement, UI, camera control).
- Improving the text classification pipeline for natural language → object label.
- Enhancing Arduino / servo control for haptic feedback.
- Documentation, examples, debugging tools, and performance improvements.

The main design goals of the project are:

- **Modularity**: Python, Unity, and hardware should be loosely coupled and easy to replace or extend.
- **Reproducibility**: A new contributor should be able to clone the repo and run the demo with minimal friction.
- **Clarity over cleverness**: Fewer hidden tricks, more explicit logic and comments.

---

## 2. Code of Conduct

Be respectful and constructive when interacting with other contributors.

If the repository contains a `CODE_OF_CONDUCT.md`, please read it and follow it.  
If it does not, assume a standard open-source etiquette: no harassment, no personal attacks, and no discrimination.

---

## 3. Project Structure

The repository is organized roughly as follows (names may vary slightly depending on the version you have):

- **Python side**
  - `main.py` – combined server entry point (hand tracking + text server + serial bridge).
  - `hand_easy.py`, `hand_udp.py` – hand tracking, depth estimation, and UDP streaming of hand data.
  - `arduino_udp_receive.py` – UDP → serial bridge; receives pinch state, sends simple commands to Arduino.
  - `collect_data.py`, `clean_dataset.py`, `train_model.py`, `run_model.py`, `text_infer_server.py` – data collection, preprocessing, training, and serving of the text classification model.
  - `text_model.pkl`, `cleaned_text_object_dataset.jsonl`, `object_models_csv.csv` – example trained model and datasets.

- **Unity side**
  - `ModelLibrary.cs` – maintains a dictionary of predefined models/prefabs and exposes `ShowModelByLabel(string label)`.
  - `TextQueryClient.cs` (or similar) – TCP client that sends text queries to Python and receives predicted labels.
  - `HandFromVectors.cs` – receives bone vectors via UDP and reconstructs 3D hand joints.
  - `PinchGrabBall.cs` – pinch-based grabbing and following behavior for grabbable objects.
  - `HandOrbitCamera.cs` – camera orbit / zoom controlled by pinch and hand motion.
  - `RuntimeModelLoader.cs` – runtime loading of `.glb` models using GLTFast or similar.

- **Hardware / Arduino side**
  - Sketches controlling one or more servos, reading serial commands from Python.
  - Simple protocol mapping `'0'`, `'1'` or small ints to specific servo angles / modes.

- **Docs / meta**
  - `README.md`, `README_CHN.md` – project overview.
  - `requirements.txt` – Python dependencies.
  - Scripts, temporary datasets, and other utilities as needed.

---

## 4. Environment Setup

### 4.1 Python

1. **Python version**

   Use a modern CPython 3.x (e.g. 3.10–3.12). The code uses common libraries such as:

   - `opencv-python`
   - `mediapipe`
   - `numpy`
   - `scikit-learn`
   - `joblib`
   - `pyserial`
   - `uvicorn` / `fastapi` or `socketserver` / sockets (depending on the text server implementation)

2. **Virtual environment**

   In the project root:

   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

4. **Quick checks**

   * `python hand_easy.py` – should open a camera window and show HUD / overlays.
   * `python hand_udp.py` – should stream UDP JSON hand data (see console logs).
   * `python text_infer_server.py` – should start the text model server.
   * `python main.py` – should start the combined server if provided.

### 4.2 Unity

* Use a recent Unity 202x LTS version (or as specified in the main README).
* Open the provided Unity project folder (often `Game/`).
* Make sure the active scene includes:

  * An object with `HandFromVectors` listening on the correct UDP port.
  * A camera with `HandOrbitCamera`.
  * A `ModelLibrary` object with preloaded prefabs or a `RuntimeModelLoader`.
  * UI components and a `TextQueryClient` component hooked to buttons / input fields.

### 4.3 Arduino / Hardware

* Install Arduino IDE (or equivalent).
* Ensure:

  * `Serial.begin(9600);` (or the baud rate that Python uses).
  * The board is connected to the port referenced in Python (e.g. `COM9` or `/dev/ttyACM0`).
  * Servo pin assignments and angle ranges are consistent with the sketch and with the expected feedback in Unity.

---

## 5. Development Workflow

### 5.1 Git and branches

1. **Fork (optional)** and clone the repository.

2. Create a feature branch:

   ```bash
   git checkout -b feature/new-gesture
   # or
   git checkout -b fix/udp-timeout
   ```

3. Keep your branch focused on a single problem or feature.

### 5.2 Commit messages

Use clear, descriptive messages. A loose guideline:

* `fix: correct pinch threshold for stability`
* `feat: add runtime GLB loading`
* `docs: explain text model training pipeline`
* `refactor: split UDP and serial bridges`

No strict format is enforced, but avoid messages like `update` or `misc`.

### 5.3 Pull Requests

A good PR should:

* Explain what it changes and why.
* Note any breaking changes (e.g. JSON schema changes, port changes).
* Mention how it was tested:

  * Python: script runs without errors, behaves as expected.
  * Unity: scene plays in the editor, interactions work.
  * Arduino: servos move as intended under the new protocol.

Try to keep PRs reasonably small and focused.

---

## 6. Python Guidelines

### 6.1 General style

* Follow PEP 8 where practical.
* Use `type hints` for new functions where straightforward.
* Prefer small, composable functions over large monolithic ones.
* Document top-level scripts with a brief docstring explaining input/output and overall behavior.

Example:

```python
def compute_pinch_state(landmarks: np.ndarray) -> bool:
    """
    Return True if the given hand landmarks represent a pinch gesture.

    landmarks: (21, 3) array of normalized coordinates.
    """
    ...
```

### 6.2 Hand tracking and networking

* Keep the **JSON schema stable** where possible. If you change it:

  * Update both the sender (Python) and receiver (Unity).
  * Document the change in the PR description.
* Make the UDP and TCP ports configurable if you add new scripts.
* Avoid busy-wait loops without some form of frame timing or `sleep` to prevent high CPU usage.

### 6.3 Text model and datasets

* For new text training data:

  * Extend the JSONL dataset (e.g. `cleaned_text_object_dataset.jsonl`) with additional labeled examples.
  * Do not commit large raw datasets; consider adding a script to generate them.
* When retraining the model:

  * Document the training command (e.g. `python train_model.py`).
  * If you commit a new `text_model.pkl`, mention it in the PR description.
* Aim for robustness to short phrases, synonyms, and slight typos where possible.

---

## 7. Unity / C# Guidelines

### 7.1 Script structure

* Use PascalCase for class names and methods.
* Use `[Header("...")]` and `[SerializeField]` to expose key fields in the Inspector.
* Avoid hard-coding magic numbers (e.g. thresholds, distances) inside the logic; expose them as serialized fields instead.

Example:

```csharp
[Header("Grab Settings")]
public float grabDistance = 0.10f;
public float followSmoothing = 0.15f;
```

### 7.2 Interaction logic

* Keep interaction responsibilities separated:

  * `HandFromVectors` should focus on reconstructing 3D hand joints and providing a clean API (e.g. `TryGetJointPosition`).
  * Grabbable scripts (`PinchGrabBall` or similar) should handle pinch detection and object following.
  * Camera scripts (`HandOrbitCamera`) should only control view manipulation.
* Prefer frame-rate independent logic when possible (e.g. use `Time.deltaTime` where appropriate).

### 7.3 Networking and text client

* Handle connection failures gracefully (e.g. show a warning in Unity console if the text server is not reachable).
* When modifying the text query protocol:

  * Keep the message format simple and line-based if possible.
  * Update both the Unity client and Python server together.

---

## 8. Arduino / Hardware Guidelines

* Document pin assignments at the top of the sketch:

  ```cpp
  const int servoPin = 9;
  ```

* Avoid extremely fast repeated servo updates; allow some time between angle changes.

* Consider safety thresholds:

  * Maximum allowed angle.
  * Fallback to a safe “home” position on error or invalid command.

* When changing the serial protocol (e.g. switching from `'0'/'1'` to more complex messages), update:

  * The Python sender.
  * Any documentation that describes expected behavior.

---

## 9. Testing and Validation

Currently, most testing is **manual** and interactive:

* **Python**

  * Run hand tracking scripts and confirm that:

    * The camera opens and FPS is reasonable.
    * Pinch detection is stable.
    * Depth estimation looks reasonable.
* **Unity**

  * Enter Play Mode and confirm:

    * The 3D hand skeleton matches your real hand motion.
    * Objects can be grabbed and released at the right pinch distances.
    * Camera orbit behaves as intended.
* **Text model**

  * Send a variety of example phrases through the Unity UI.
  * Ensure the predicted labels match expectations and load the correct models.
* **Hardware**

  * Verify that servo movement corresponds to the intended events (e.g. “pinch start” vs “pinch end”).

If you add automated tests (Python unit tests, Unity tests, etc.), document how to run them in your PR.

---

## 10. Style, Formatting, and Git Hygiene

* Use consistent indentation:

  * Python: 4 spaces.
  * C#: follow Unity default conventions.
* Avoid trailing whitespace and keep files ending with a newline.
* Do not commit:

  * Large binary assets that are not essential (e.g. Unity’s `Library/`).
  * Local configuration files or secrets.
* Check `.gitignore` before adding new generated files or build outputs.

---

## 11. Reporting Issues and Requesting Features

When opening an issue, include:

* **Environment**: OS, Python version, Unity version, Arduino board type.
* **Steps to reproduce**: what you did and in what order.
* **Expected behavior** and **actual behavior**.
* Logs, stack traces, screenshots, or short video recordings if relevant.

For feature requests:

* Describe the use case (what a user would experience).
* Suggest where in the pipeline it fits (Python, Unity, Arduino, or all three).
* Mention whether it is breaking or backward-compatible.

---

## 12. License and Ownership

Unless otherwise stated, contributions to this repository are assumed to be licensed under the same license as the project itself (see the `LICENSE` file if present).

By submitting a pull request, you confirm that:

* You have the right to contribute the code/content.
* You agree that it may be used under the project’s license.

---

Thank you again for contributing to SecondTouchReality.
Even small improvements—comments, example scenes, better defaults—can significantly improve the experience for future users and contributors.

```
