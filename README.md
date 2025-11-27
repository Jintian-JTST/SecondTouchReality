下面先给你一版新的 README markdown，然后在后半部分给你一个 1–2 分钟的英文口播稿，可以直接拿去讲。

---

## 📄 建议版 README.md（Markdown）

````markdown
# SecondTouchReality (WIP)

SecondTouchReality 是一个“小型全链路系统”，把 **Python 手部追踪 + 深度估计 + 文本分类模型** 和 **Unity 里的 3D 手、抓取交互与教学场景** 串成一条流水线。后面会预留一个 **硬件接口**（手套 / Arduino 等）来做真实世界的反馈。

---

## 1. 整体思路

从用户视角，这套系统做了三件事：

1. **看懂手**：Python 使用 MediaPipe 检测双手关键点，估计掌根到摄像头的真实距离，算出骨骼方向和捏合（pinch）。然后通过 UDP 把这些数据发给 Unity。  
2. **看懂文本**：用户在 Unity 里输入一句话（比如 “a small red apple”），Python 端的文本模型会判断最接近哪一个物体 ID，并把这个标签回传给 Unity。  
3. **在 Unity 里“摸到”东西**：Unity 里根据标签加载对应 3D 模型，重建 3D 手部骨骼，让你用 pinch 抓起、移动、旋转，同时也能用 pinch 控制相机视角。  

未来会增加一层 **硬件接口**：Unity 或 Python 进一步把交互事件映射成串口命令，驱动自制手套或其他教学装置（目前预留为 TODO）。

---

## 2. 系统架构概览

```text
Camera → Python(MediaPipe + Depth) → UDP → Unity HandFromVectors
                                   ↑
                     Python Text Model ← Unity TextQueryClient
                                   ↓
                           (Future) Hardware / Arduino
````

### 2.1 Python：手部深度 & 骨骼 + UDP

主要脚本：

* `hand_easy.py`

  * 单手版本的掌宽 / 掌长 / curl / side / 掌心朝向等计算，并给出一个稳定的 `Z_disp`（掌根距离，单位米）。

* `hand_two_hands_z_udp.py`

  * 支持多只手（通常 2 只），对每只手：

    * 估计掌根深度（米）；
    * 检测拇指尖–食指尖 pinch；
    * 生成 20 条骨骼方向单位向量；
  * 通过 UDP 把 JSON 发送到 Unity（默认 `127.0.0.1:5065`）。

### 2.2 Python：文本 → 物体标签 模型

数据与训练流程：

* `collect_data.py`：交互式收集数据，逐条输入 “描述文本 + 标签”，追加到 `text_object_dataset.jsonl`。
* `clean_dataset.py`：清洗数据，只保留英文样本，写入 `cleaned_text_object_dataset.jsonl`。
* `train_model_with_eval.py`：

  * 使用字符级 n-gram 的 `HashingVectorizer` + `SGDClassifier`；
  * 训练完成后保存到 `text_model.pkl`，并在训练集上做一个简单的精度报告。
* `run_model.py`：加载模型，对一条文本做 top-k 推理，可在命令行直接试。
* `text_infer_server.py`：

  * 一个多线程 TCP 服务（默认监听 `127.0.0.1:9009`）；
  * Unity 发送一行描述，它返回最可能的标签字符串。

### 2.3 Unity：3D 手、抓取与相机控制

主要 C# 组件：

* `HandFromVectors.cs`

  * 监听 UDP 端口（默认 5065），接收 Python 发来的 JSON；
  * 为每只手重建 21 个关节位置 + 20 条骨骼线，并用小球画出来；
  * 提供 API：

    * `IsHandPinching(int handIndex)`
    * `TryGetJointPosition(int handIndex, int jointIndex, out Vector3 pos)`
    * `MaxHandCount` / `IsLeftHand` / `IsRightHand`。

* `PinchGrabBall.cs`

  * 挂在“可以被抓”的物体上；
  * 当任意一只手 pinch 且食指靠近物体时，开始抓取；
  * 抓住后物体跟随掌根（或你指定的关节），带有平滑与宽限时间，防止 pinching 抖动导致频繁松手；
  * 静态属性 `AnyObjectGrabbed` 让别的脚本知道场景里是否有东西被抓着。

* `HandOrbitCamera.cs`

  * 完全不区分左右手，只要手在 pinch 就可以控制视角；
  * 没有物体被抓住时：

    * 单手 pinch：拖动视角，绕 target 旋转；
    * 双手 pinch：两手距离变化控制相机远近；
  * 有任意物体被 `PinchGrabBall` 抓住时，会让出相机控制权。

* `ModelLibrary.cs`

  * 管理挂在自己下面的所有子物体，把它们当作一个“模型字典”；
  * `ShowModelByLabel("023")` 会：

    * 激活名为 `"023"` 的子物体；
    * 把它摆在摄像机前方设定距离处；
    * 自动给它挂上 `PinchGrabBall`，能被手抓起来。

* `TextQueryClient_TMP.cs`

  * Unity 侧的 TCP 客户端；
  * 弹出一个 TMP 输入框，让用户输入英文描述；
  * 把文本发送给 `text_infer_server.py`，接收回来的标签字符串；
  * 调用 `ModelLibrary.ShowModelByLabel(resp)` 生成对应模型，并在 UI 上显示预测结果。

---

## 3. 依赖与环境

### 3.1 Python 环境

项目提供了 `requirements.txt`，包含手部追踪、机器学习和可视化相关依赖，例如：

* `mediapipe`
* `opencv-python`
* `scikit-learn`
* `torch`, `timm`
* `numpy`, `pandas` 等等。

推荐做法：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows 下使用 .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Unity 环境

* Unity 版本：使用标准内置渲染管线即可；
* 把 C# 脚本拖到合适的 GameObject 上，并在 Inspector 里连好引用（例如 `handTracker`、`modelLibrary`、按钮回调等）。

---

## 4. 典型运行流程（软件部分）

1. **启动 Python 手部追踪**

   ```bash
   python hand_two_hands_z_udp.py
   ```

   看到摄像头画面、HUD 显示 FPS。按 `c` 校准深度，按 `r` 重置标定。

2. **启动文本推理服务器**

   ```bash
   python text_infer_server.py
   ```

   这会加载 `text_model.pkl`，监听来自 Unity 的 TCP 请求。

3. **在 Unity 里运行场景**

   * 场景中放一个挂有 `HandFromVectors` 的对象，端口与 Python UDP 设置一致；
   * 主摄像机挂 `HandOrbitCamera`，把 `handTracker` 指向上面的对象，把 `target` 设成你要围绕看的物体或一个空物体；
   * 一个空物体挂 `ModelLibrary`，把所有候选模型作为子物体；
   * 一个 UI 控制器挂 `TextQueryClient_TMP`，把按钮、输入框、`modelLibrary` 都绑好。

此时你可以：

* 举起手，看到 Unity 里实时重建的 3D 手；
* pinch 时旋转 / 缩放相机视角；
* 在 UI 对话框里输入一句话，看到对应模型被生成在你面前，并用 pinch 抓起来移动。

---

## 5. 未来：硬件接口预留

当前仓库已经实现了 **Python ↔ Unity** 的数据通道。下一步计划是：

* 在 Python 侧增加一个 TCP / UDP → 串口的桥接脚本，把 Unity 的指令转发给 Arduino；
* 协议形式可以是 `"finger,angle\n"` 或类似格式，由硬件团队与软件共同约定；
* 用这些命令驱动自制手套 / 机械结构，做力反馈、灯光或其他教学反馈。

这部分目前处于设计阶段，尚未在仓库中实现，但上层架构已经为此预留了空间。

---

## 🎤 英文 1–2 分钟项目口头介绍稿

下面是你可以在 demo、答辩或者路演时使用的一段英文介绍，默认是“对评委讲解整个软件 & Unity 工作流程”的口吻：

---

Let me give you a quick overview of our software pipeline and how it connects to Unity.

Our system has two brains: one in Python, and one in Unity.  
On the Python side, we use MediaPipe to track the hands in real time and estimate how far the wrist is from the camera in meters. We also compute pinch gestures and bone directions for each finger, for up to two hands. Every frame, Python packs this into a compact JSON message and sends it to Unity over UDP.   

Unity receives this stream with a script called `HandFromVectors`. It reconstructs a full 3D skeleton with 21 joints and 20 bones for each hand, draws small spheres in the scene, and exposes simple APIs like “is this hand pinching?” or “where is joint 8 in world space?”. That lets us treat the real hand like a 3D controller.   

Interaction is handled by a couple of small components. `PinchGrabBall` can be attached to any object we want to grab. When any hand pinches near that object, it’s “picked up” and smoothly follows the hand, with a bit of tolerance so brief tracking glitches don’t drop it. `HandOrbitCamera` uses the same pinch signals to drive the camera: with one pinching hand you orbit around the target; with two pinching hands you zoom in and out, as long as nothing is currently grabbed.   

On top of that, we add a lightweight text-to-object layer. In Unity, a small UI script opens a dialog box where the user types an English description. The text is sent over TCP to a Python server with a simple character-level model trained on our own dataset. The server returns a label like “apple” or “023”, and Unity uses `ModelLibrary` to activate the corresponding prefab in front of the camera and automatically make it grabbable with pinch.   

Finally, we have reserved space in the architecture for a hardware interface. In the next step, the same signals that now drive virtual objects and camera motion will also be mapped to serial commands for a glove or other teaching devices, so the virtual interaction in Unity can directly trigger real-world feedback.

In short, the software stack links camera, AI models, Unity interaction, and—soon—physical hardware into one continuous loop.
```
