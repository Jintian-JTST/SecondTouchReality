# SecondTouchReality 项目说明

### **SecondTouchReality 是由 OneTouchReality 演化而来的。**
SecondTouchReality 是一个「**虚拟现实手势控制教学物体系统**」的衍生版本，重点在于「用自然语言生成物体 + 用捏合手势抓取物体 + 用单通道舵机反馈捏合状态」，并且把整体架构设计得更模块化、易扩展。SecondTouchReality 是一个小型但比较完整的“端到端”系统，延续了这条链路，但做到了「抽骨架」和「轻量化」：

- **Python 端**：手部追踪 + 深度估计 + 文本分类模型  
- **Unity 端**：3D 手骨重建 + 抓取交互 + 视角控制  
- **硬件端**：Arduino / 舵机 / 手套（预留接口）

串成一条流水线：

```摄像头 → Python 理解你的手 + 理解一句话 → Unity 生成 3D 场景并可交互 → 驱动硬件反馈```

目前仓库处于原型阶段，但完整体现了“感知 → 语义 → 交互 → 硬件”的闭环。

---

## 1. 系统整体结构

从最终使用者视角，这套系统做了三件事：

1. **看懂手**  
   - Python 用 MediaPipe Hands 检测 21 个手部关键点。  
   - 计算掌宽、掌长、卷曲度 curl、侧向程度 side、掌心/手背朝向。  
   - 用一次标定把掌宽/掌长换算成掌根到摄像头的真实距离（米），并做滤波。  
   - 把 wrist 的 3D 位置、20 条骨骼方向向量和 `pinch`（捏合）状态打包成 JSON，用 UDP 发给 Unity。

2. **看懂话**  
   - 在 Unity 中弹出对话框，输入一句英文描述（如："a small red apple"）。  
   - Python 端小型文本模型用 `HashingVectorizer + SGDClassifier` 做多分类，输出一个离散 label（比如 "101"）。  
   - TCP 返回 `"101"` 这种格式给 Unity。

3. **摸东西**  
   - Unity 用 `HandFromVectors` 根据 UDP JSON 重建 3D 手部骨骼，用小球和线段画出来。  
   - `PinchGrabBall` 让你捏住场景里的物体，跟着手移动。  
   - `HandOrbitCamera` 让你在没有抓任何东西的时候，通过 pinch + 移动手 来旋转/缩放视角。  
   - `ModelLibrary` / `RuntimeModelLoader` 根据 label 加载对应的 3D 模型（预制体或 GLB 文件）。

4. **目录树**

```text
SecondTouchReality/
├── README.md              # 英文/混合版说明（含总体设计）
├── README_CHN.md          # 纯中文说明（含总体设计）
├── requirements.txt       # Python 依赖
├── text_model.pkl         # 训练好的文本分类模型
├── main.py                # combined_server，一键启动整条链路
├── .gitignore
├── unecessary/            # 一些旧脚本或不再使用的资源
├── tools/
│   ├── hand_easy.py       # 手部距离估计 + 标定逻辑
│   ├── hand_udp.py        # 多手 + 骨骼向量 + pinch → UDP JSON
│   ├── arduino_udp_receive.py # 仅 UDP → Arduino 的桥接脚本
│   ├── text_infer_server.py   # 文本模型推理 TCP 服务
│   ├── run_model.py       # 加载 text_model.pkl，提供命令行推理
│   └── __pycache__/       # Python 缓存
├── test/
│   ├── collect_data.py    # 采集「描述文本 + 标签」数据
│   ├── clean_dataset.py   # 清洗 JSONL 数据集
│   ├── train_model.py     # 训练文本分类模型并输出 text_model.pkl
│   ├── text_object_dataset.jsonl
│   ├── cleaned_text_object_dataset.jsonl
│   └── object_models_csv.csv # 文本标签 ↔ 模型 ID 映射表
├── Game/                  # Unity demo 工程（可直接打开）
│   ├── SampleScene.unity  # 主场景
│   ├── models/            # 若干 glb 模型（苹果、香蕉、碗等）
│   └── Scripts/           # 主要 C# 脚本
│       ├── HandFromVectors.cs
│       ├── HandOrbitCamera.cs
│       ├── PinchGrabBall.cs
│       ├── ModelLibrary.cs
│       ├── RuntimeModelLoader.cs
│       └── TextQueryClient_TMP.cs
└── ...
```


* **根目录**

  * `main.py`：推荐的入口脚本。打开摄像头、启动 UDP → Unity 的手部数据流，同时挂载 `on_payload` 回调，把 pinch 状态映射到串口命令，再顺便启动文本推理服务器。
  * `requirements.txt`：列出所有 Python 依赖包，通常用 `pip install -r requirements.txt` 一次装完。
  * `text_model.pkl`：由 `test/train_model.py` 训练得到，用于把自然语言描述映射到物体标签。

* **`tools/` – 运行时工具层**

  * `hand_easy.py`：封装了距离标定与滤波逻辑，供 `hand_udp.py` 等脚本调用。
  * `hand_udp.py`：

    * 调用 MediaPipe，支持多只手；
    * 输出掌根深度、骨骼方向、pinch 状态；
    * 通过 UDP 把 JSON 包发往 Unity（默认 `127.0.0.1:5065`）；
    * 允许外部设置 `on_payload` 回调。
  * `text_infer_server.py`：启动一个基于 `socketserver` 的多线程 TCP 服务，使用 `run_model.py` 里的 `infer_once` 在内存中调用文本模型。
  * `arduino_udp_receive.py`：简化版桥接程序，从 UDP 收到 hand JSON 后，只关心 `hands[].pinch`，检测状态变化并向 Arduino 串口发送 `'0'/'1'`。

* **`test/` – 数据与模型实验区**

  * `collect_data.py`：交互式命令行工具，用来快速收集训练数据。
  * `clean_dataset.py`：从原始数据中过滤掉含中文的样本，只保留干净的英文文本和标签，输出到 `cleaned_text_object_dataset.jsonl`。
  * `train_model.py`：对清洗后的数据训练模型，并将结果保存为 `text_model.pkl` 供主程序加载。

* **`Game/` – Unity demo**

  * `HandFromVectors.cs`：UDP 客户端，负责解析 Python 发送的 JSON，重建关节位置，并用小球和线段可视化。
  * `PinchGrabBall.cs`：把任何 3D 物体变成“可捏”的物体，处理抓取、跟随、释放和平滑移动等逻辑。
  * `HandOrbitCamera.cs`：用 pinch 控制视角旋转与缩放。
  * `ModelLibrary.cs`：维护一个名字 → GameObject 的字典，提供 `ShowModelByLabel(label)`，供文本推理结果直接调用。
  * `TextQueryClient_TMP.cs`：基于 TextMeshPro 的文本输入客户端，负责和 Python 文本服务器通信。
  * `RuntimeModelLoader.cs`：支持运行时从 `StreamingAssets/models` 中按编号加载 glb 模型，可用于扩展模型库。

---

## 2. 技术栈

### 2.1 Python 端

- Python 3.x
- OpenCV (`cv2`) – 摄像头采集 + HUD 绘制
- MediaPipe Hands – 手部关键点检测
- NumPy – 各种向量运算、统计量（中位数、EMA 等）
- scikit-learn – 文本特征（`HashingVectorizer`）+ 线性分类器（`SGDClassifier`）
- joblib – 模型与标签编码器序列化 (`text_model.pkl`)
- socket / socketserver – UDP + TCP 通信
- pyserial – 串口连接 Arduino

主要 Python 文件：

- `hand_udp.py` – 主手部追踪 + UDP 推流
- `hand_easy.py` – 深度估计算法 demo / 调试
- `collect_data.py` / `clean_dataset.py` / `train_model.py` / `run_model.py` – 文本模型数据与训练工具链
- `text_infer_server.py` – 文本推理 TCP 服务
- `main.py` – 把手部追踪 + 文本服务 + 串口桥接综合到一个进程里
- `arduino_udp_receive.py` – 备选方案：UDP → 串口桥接

### 2.2 Unity / C# 端

- Unity 202x
- C# 脚本：
  - `HandFromVectors.cs` – UDP 收包 + 手骨重建 + GUI 调参
  - `PinchGrabBall.cs` – 物体捏取逻辑
  - `HandOrbitCamera.cs` – 摄像机绕场景目标旋转和缩放
  - `ModelLibrary.cs` – 将子物体/预制体组成一个“模型字典”
  - `RuntimeModelLoader.cs` – 用 GLTFast 动态加载 `.glb`
  - `TextQueryClient.cs`（类名 `TextQueryClient_TMP`）– Unity 端的文本 TCP 客户端 + UI
- TextMeshPro – 输入框与文字显示
- GLTFast – 运行时加载 .glb / .gltf 模型

### 2.3 硬件端

- Arduino（Uno / Nano 等）
- 简单舵机若干（演示用）
- 串口协议极简：每次发送一个 ASCII 字符，比如 `'0'` / `'1'`。

---

## 3. 运行流程

一个典型的使用流程（对应你现在的操作）是：

1. **先打开 Python 端**（`main.py`）  
2. 在 Python 弹出来的摄像头窗口中按 **`c`** 做标定，`r` 重置，`q` 退出。  
3. **再打开 Unity 工程**，进入包含 `Skin` 场景（或你自己的 demo 场景）。  
4. 在 Unity 里点击 **Play**：  
   - 看到 3D 手骨跟着你的真实手在动；  
   - pinch 时能旋转视角或抓取物体；  
   - 在对话框里输入一句话，自动生成对应的 3D 模型到你面前。

下面拆细一点。

---

### 3.1 配置 Python 环境

1. 在项目根目录创建虚拟环境（建议）：

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
    ```
    
2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 确认摄像头能被 OpenCV/MediaPipe 访问。

---

### 3.2 启动综合服务器 `main.py`

在项目根目录运行：

```bash
python main.py
```

此时将有：

* 一个摄像头预览窗口，带 HUD 提示（FPS、标定状态等）。
* 后台开着：

  * UDP 手部数据服务器（发给 Unity）。
  * 文本 TCP 服务（监听 `127.0.0.1:9009`）。
  * 串口（如果连接 Arduino 的话）。

在摄像头窗口中操作：

* 按 **`c`**：

  * 保持手掌张开，正对摄像头不动，它会采样 50 帧左右。
  * 终端里会让你输入“此时掌根到摄像头的真实距离（米）”，比如 `0.45`。
  * 程序根据采样到的掌宽/掌长中位数计算 `k_w` / `k_l`，以后就能大致估出 Z。

* 按 **`r`**：重置标定状态。

* 按 **`q`**：退出 Python 端。

标定完成后，HUD 上会从 “Calib: NOT SET” 变成类似 “Calib: OK”。

---

### 3.3 打开 Unity 场景

1. 在 Unity 里打开本工程，对应的 demo 场景通常类似 `Skin.unity`：
   场景中应该有：

   * 一个物体挂着 `HandFromVectors`：

     * `listenPort = 5065`（要跟 Python 对应）。
     * 指定 `targetCamera`（一般是主摄像机）。
   * 主摄像机上挂 `HandOrbitCamera`。
   * 某个节点挂了 `ModelLibrary`，其子物体是各种模型模板（名字一般用 label 做文件名）。
   * 一个 UI Canvas 里挂了 `TextQueryClient_TMP`，绑定 TMP 输入框和按钮。

2. 点击 **Play**：

   * 摄像机画面里会出现 3D 手骨（小球 + 线段）。
   * 当手做 pinch（拇指+食指捏合）时：

     * 如果附近有 `PinchGrabBall` 物体，就会被抓住跟着手走；
     * 如果没抓到物体，`HandOrbitCamera` 会把 pinch 解释为“控制摄像机”——移动手改变视角。

3. 在 UI 上点击按钮弹出对话框，输入一行英文描述，例如：

   ```text
   a green apple
   ```

   并点击确认按钮：

   * Unity 把这串文本发到 `127.0.0.1:9009`。
   * Python 调用文本模型，返回 `102|0.93` 类似的结果。
   * `TextQueryClient_TMP` 解析出 `label = "102"`，调用 `ModelLibrary.ShowModelByLabel("102")`。
   * `ModelLibrary` / `RuntimeModelLoader` 在镜头前生成一个对应模型，并自动挂上 `PinchGrabBall` 让你捏住它。

---

### 3.4 连接 Arduino

1. 在 Arduino 中烧入一个简单的串口控制程序，例如：

   * `Serial.begin(9600);`
   * `if (Serial.available()) char c = Serial.read();`
   * 若 `c == '1'` → 舵机转到 45°，`c == '0'` → 回到 0°。

2. 在 `main.py` 或 `arduino_udp_receive.py` 中，把 `COM9` 换成实际串口号。

3. 运行 Python：

   * 手追踪正常时，`on_payload` 或 `arduino_udp_receive` 会根据 pinch 状态发送 `'1'` / `'0'`。
   * 观察舵机随 pinch / release 动作移动。

---

## 4. 主要 Python 脚本说明（按功能）

### 4.1 手部追踪：`hand_udp.py`

功能总结：

* 打开摄像头，使用 MediaPipe Hands 检测多只手。
* 每一帧计算：

  * 掌宽 / 掌长（像素）
  * 卷曲度 `curl`（0~1）
  * 侧度 `side`（0~1）
  * 掌心/手背朝向 `palm_front`
  * 掌根深度 `Z`（米）
  * 20 条骨骼方向向量（单位向量）
  * `pinch`（拇指+食指是否捏合）
* 把上述信息打包成 JSON，UDP 发给：

  * Unity 监听端口（默认 5065）
  * Arduino UDP 桥接端口（默认 5066）

关键点：

* **标定逻辑**

  * 使用 `CalibState` 结构存储：采样的掌宽列表 / 掌长列表、`k_w`、`k_l` 等。
  * 按下 `c` 后采样一段时间，再让你输入真实距离，计算 `k_w = d_real * w_med` 之类的标定参数。
  * 深度估计中，用 `Z ≈ k_w / palm_width` 与 `Z ≈ k_l / palm_length` 为两条通道，再结合 curl / side 做加权融合。

* **pinch 检测**

  * 一般是取 `thumb_tip` 与 `index_tip` 的距离，小于阈值时认为在捏合。
  * 直接写入 `"pinch": true/false` 到 JSON。

* **回调 on_payload**（在 `main.py` 中注册）

  * 可以拿到整帧 JSON，统计有多少只手在 pinch，并进行后处理（如串口输出）。

---

### 4.2 深度估计算法 Demo：`hand_easy.py`

用途是“把复杂的深度估计单独拎出来玩”，不牵涉到 UDP 和 Unity。

里面重点函数：

* `compute_palm_width_and_length(...)` – 只要有 landmark，就能算出两个像素量，作为深度 proxy。
* `compute_curl(...)` – 根据手指关节夹角判断手是摊开还是握拳。
* `compute_side(...)` – 手是正对摄像头还是侧过来。
* `fuse_depth(Zw, Zl, curl, side, palm_front, ...)` – 用权重/修正项把两个深度通道融合成一个 `Z_final`。
* `draw_hud(...)` – 在画面旁边打印所有中间量，方便调参理解。

---

### 4.3 文本模型链路

* **原始数据**：`text_object_dataset.jsonl`
  每行一个 JSON：`{"text": "...", "label": "101"}`，包含中英文。

* `collect_data.py`：交互式添加数据。

* `clean_dataset.py`：过滤掉含中文字符的样本，写入 `cleaned_text_object_dataset.jsonl`。

* `train_model.py`：从 cleaned 数据训练/增量训练一个 `SGDClassifier`，保存到 `text_model.pkl`，同时打印训练集指标。

* `run_model.py`：命令行下测试模型，对输入句子打印 top-k label+prob。

* **`text_infer_server.py`**：

  * 启动时加载模型一次。
  * 作为 TCP 服务对外暴露：

    * 每次收到一行文本 → 做一次推理 → 返回 `"label|prob\n"`。

---

### 4.4 综合入口：`main.py`

`main.py` 把三个方向的功能粘在一起：

1. **手追踪 + UDP**：从 `tools.hand_udp` 启动手部追踪循环。
2. **串口桥接**：给 `hand_udp` 注册一个回调 `on_payload(payload)`

   * 在 JSON 里统计当前是否有手在 pinch。
   * 若有 → `ser.write(b"1")`，否则 → `ser.write(b"0")`。
3. **文本 TCP 服务**：用 `TextInferHandler` + `ThreadedTCPServer` 在 9009 端口监听 Unity 发来的文本。

这样你只要跑一个 `python main.py`，就可以同时支撑 Unity 的手追踪 + 文本生成 + 硬件反馈。

---

### 4.5 UDP → 串口脚本：`arduino_udp_receive.py`

当你不想在 `main.py` 里混合太多逻辑时，可以单独跑这个脚本：

* UDP `5066` 收到与 Unity 相同格式的 JSON。
* 解析出当前的 pinch 状态。
* 若状态变化，向 Arduino 串口发送 `'0'` 或 `'1'`。
* 适合把“硬件桥接”独立出来做调试。

---

## 5. Unity 侧脚本说明

### 5.1 `HandFromVectors.cs` – UDP 收包 + 手骨重建

核心职责：

* 从指定端口（默认 5065）创建 `UdpClient`。
* 解析 Python 发来的 JSON：

  * `wrist` 像素坐标 + 归一化坐标 + `z_m`（深度）。
  * 20 条骨骼方向向量（单位向量）。
  * `pinch` / `is_left` 信息。
* 利用：

  * 摄像机内参（统一用 `Camera` 的投影）。
  * 事先配置好的“每段骨头长度”。
    重建 21 个关节在 Unity 世界坐标系下的位置。
* 动态生成：

  * 小球 `jointObjects` 用来可视化各关节。
  * `LineRenderer` 数组 `boneLines` 画出骨骼。
* 对外提供 API：

  * `TryGetJointPosition(handIndex, jointIndex, out Vector3 pos)`
  * `bool IsPinching(handIndex)`
  * `bool AnyHandPinching`

另外还在屏幕上画一个 GUI 窗口，支持：

* 调整每段骨骼长度；
* 开关 debug 选项；
* 查看当前多少只手在线、pinch 状态等。

---

### 5.2 `PinchGrabBall.cs` – 捏取物体

只要在任意 GameObject 上挂这个脚本，并绑定 `handTracker`，就可以被捏住：

* 未被抓时：

  * 遍历所有手，查看是否有正在 pinch 的手，其控制关节（`controlJointIndex`，默认食指指尖）离该物体的距离小于 `grabDistance`。
  * 若满足，就认为“被抓住”，记录：

    * 是哪只手抓的 `grabbedHandIndex`
    * 初始偏移量 `grabOffset`

* 被抓时：

  * 如果启用了 `usePhysics`：

    * 关闭重力，`Rigidbody` 位置插值跟随跟随关节/控制关节。
  * 如果没用物理：

    * 直接用 `Vector3.Lerp` 让 `transform.position` 追踪目标位置，平滑程度由 `followSmoothing` 控制。

* `pinchReleaseGrace` 用来处理 MediaPipe 偶尔抽风：

  * pinch 短暂变成 false 但立刻又恢复时，不会立刻把物体丢掉。
  * 超过一定时长才真正释放。

脚本里有一个静态计数 `grabbedCount`，其他脚本（比如相机控制）可以通过 `AnyObjectGrabbed` 判断当前是否有人在抓东西。

---

### 5.3 `HandOrbitCamera.cs` – 用手控制视角

挂在摄像机上，让 pinch 手势在“没抓到物体”的时候用来控制视角：

* 选择一个关节（默认 index tip）作为控制点。
* 记录 pinch 开始那一帧的手的位置和当前相机的 yaw / pitch。
* 在 pinch 持续期间：

  * 把手在屏幕上的移动量映射到 yaw / pitch；
  * 用 clamp 限制俯仰角，避免翻到头顶背后；
  * 根据缩放手势或深度变化调整半径 `radius`，从而实现推远/拉近。

最后统一更新：

```csharp
orbitCamera.transform.position = pivot + dir.normalized * radius;
orbitCamera.transform.LookAt(pivot, Vector3.up);
```

---

### 5.4 `ModelLibrary.cs` – 预制体库

设计思路是：把这个脚本挂在一个空物体上，然后把所有模型作为它的子物体：

* `Awake()` 中：

  * 把所有子物体的 `GameObject` 收集到一个 `Dictionary<string, GameObject>` 中，以名字为 key。
  * 同时把这些子物体 `SetActive(false)`，当成“模板”使用。

* `ShowModelByLabel(string label)`：

  * 根据 label 找到对应的模板。
  * 如果之前已经有一个展示中的对象，先把它隐藏或销毁。
  * 在 `spawnAnchor.position + spawnOffset` 附近实例化一个新对象。
  * 确保上面挂了 `PinchGrabBall`，并设置：

    * `handTracker`
    * `grabDistance`
    * `usePhysics`

这样，从文本模型返回的 label 就可以直接决定“场景里出现哪一个 prefab”。

---

### 5.5 `RuntimeModelLoader.cs` – 运行时加载 GLB

用于“动态加载模型而不是一开始全部放在场景里”。

主要接口：

* `MakeFileName(int index)`：

  * 缺省逻辑是 `101 → "101.glb"`，你可以改成更复杂的映射（比如查表）。
* `LoadByIndexAsync(int index)`：

  * 构造路径（通常结合 `Application.streamingAssetsPath` 等）。
  * 用 `GltfImport` 载入 GLB。
  * 实例化为一个 `GameObject`。
  * 如有旧的 `currentInstance`，先销毁它。
  * 把新对象设置为 loader 的子物体，并把 `localPosition` / `localRotation` 归零。
* `LoadByIndex(int index)`：

  * 一个不关心 `await` 的简单封装 `_ = LoadByIndexAsync(index);`。

如果你有一批放在 StreamingAssets 的 `.glb`，可以让 label 直接对应文件名，然后在 Unity 里真正按需加载。

---

### 5.6 `TextQueryClient_TMP` – Unity 文本 TCP 客户端

挂在 UI 物体上，通过 TMP 输入框和按钮跟 Python 的文本服务交互。

功能流程：

1. `OpenDialog()`：

   * `dialogPanel.SetActive(true)`，并选中输入框。

2. 按钮点击 → `OnClickSend()`：

   * 读取用户输入 `descriptionInput.text`；
   * 启动 `SendQueryCoroutine(q)`。

3. `SendQueryCoroutine` 中：

   * 用 `TcpClient` 连接 `serverIp:serverPort`（默认 127.0.0.1:9009）；
   * 把 `q + "\n"` 用 UTF-8 发出去；
   * 阻塞读取一行响应；
   * 解析成 `<label>|<prob>`；
   * 若绑定了 `modelLibrary`，则 `modelLibrary.ShowModelByLabel(label)`；
   * 也可以在某个 TMP 文本上显示当前预测结果。

4. `OnDestroy()`：关闭流与 socket。

---

## 6. 数据与协议

### 6.1 UDP JSON（Python → Unity）

* 端口：5065（Unity）/ 5066（Arduino UDP 桥接）
* 编码：UTF-8 JSON
* 顶层字段：`timestamp`, `fps`, `num_hands`, `hands`
* 每只手中包括：

  * `id` – 手的索引
  * `is_left` – 是否左手
  * `wrist` – `{px, py, nx, ny, z_m}`
  * `bones` – 若干 `{from, to, vx, vy, vz}`
  * `pinch` – bool

### 6.2 文本 TCP（Unity ↔ Python）

* 地址：`127.0.0.1:9009`
* 请求：一行文本 + `\n` 结束
* 响应：`<label>|<probability>\n`

例：

```text
a small red apple\n
→ 101|0.923\n
```

### 6.3 串口（Python → Arduino）

* 波特率：9600
* 数据：单字节 ASCII 字符

  * `'1'` – 有手在 pinch
  * `'0'` – 所有手都没有 pinch

Arduino 侧如何解释完全由你决定，本仓库配套示例是驱动舵机转动角度。

---

## 7. 数据集与对象 ID

* `object_models_csv.csv`：

  * 维护了一张 “对象 ID → 英文名称 → 类别” 的表，比如：

    * `101, red apple, fruit`
    * `203, tomato, vegetable`
    * ……
  * 可以用来对齐：

    * 文本数据集的 label
    * GLB 文件名
    * Unity 预制体名称

* `text_object_dataset.jsonl` / `cleaned_text_object_dataset.jsonl`：

  * 可以持续扩展，用来训练更强的文本模型。
  * 当前清洗逻辑是：简单过滤掉中文样本，只保留英文句子。

---

## 8. 未来方向

这个仓库从工程上已经打通了三个世界：

> 摄像头世界 → 算法世界 → 虚拟世界 →（预留）硬件世界

后面可以做很多有意思的扩展：

* **语义升级**

  * 把当前简单的 `HashingVectorizer + SGDClassifier` 换成语义检索或大模型嵌入。
  * 真正支持中英文混合输入，让孩子可以用中文描述、系统内部转成英文。

* **手势升级**

  * 不只是 pinch，可以加“握拳、指点、挥手”等复杂动态手势。
  * 把这些手势映射到不同的教学交互（比如：选择、确认、删除等）。

* **内容生成**

  * 一次性生成成套的教学关卡，而不仅仅是一个物体。
  * 用 object CSV + 文本描述给小朋友生成情境化场景（厨房、超市、教室……）。

* **硬件反馈**

  * 更复杂的手套、振动马达、阻尼装置，把虚拟抓到的东西在真实世界“摸出来”。
  * 把 Unity 中的碰撞、任务完成事件映射成多通道硬件反馈。

* **在线学习**

  * 把用户输入的句子和选择的物体记录下来，继续在线微调文本分类模型。
  * 让系统越来越懂“用户自己的表达方式”。

---

## 9 从 OneTouchReality 到 SecondTouchReality

OneTouchReality 最早是一整套「**虚拟现实手势控制机械臂系统**」：摄像头 + MediaPipe 做手势识别，经 UDP 把 21 个关键点丢给 Unity，在 Unity 里重建整只三维手，做碰撞检测，再通过 TCP 把“应该拉哪根手指、拉多少”发回 Python，最后由 Arduino 驱动 5 个舵机，把虚拟世界里的“碰撞”变成真实手指上的拉力。

1. **视觉链路保留但更加干净**

   * Python 里还是用 MediaPipe 做手部关键点识别和姿态估计，不过不再把 21 个点原封不动塞给 Unity，而是提炼出：

     * 掌根的真实距离（米级）
     * 20 条骨骼方向向量
     * 是否捏合（pinch）等离散状态
   * 距离估计模块继承了 OneTouchReality 里“掌宽 / 掌长 双通道 + 权重融合 + 中值滤波/EMA”的那套思路，但封装成了 `hand_easy.py`，按 `c` 标定、`r` 重置、`q` 退出，便于在别的项目里复用。

2. **交互逻辑从“机械臂”换成“教学物体 + 抓取 + 视角控制”**

   * Unity 端用 `HandFromVectors.cs` 接收 Python 发来的 JSON，按照骨骼方向和骨长还原 21 个关节位置，并用小球 + 线段画出虚拟手。
   * `PinchGrabBall.cs` 负责「捏住就跟手走」：

     * 在所有正在 pinch 的手里找离物体最近的一只作为“抓取手”；
     * 记录手指关节到物体的位置偏移；
     * 更新时根据关节位置 + 偏移平滑移动模型，可以选用物理刚体或直接插值。
   * `HandOrbitCamera.cs` 则让 pinch 变成一种“通用手势鼠标”：

     * 单手 pinch → 旋转相机绕目标物体；
     * 双手 pinch → 调整相机半径（类似 pinch-zoom）；
     * 任意物体被 `PinchGrabBall` 抓住时，相机暂时停止响应，避免操作冲突。

3. **从“检测碰撞的手套”到“会听人话的场景”**

   * 原项目里 Unity 负责检测“手指骨骼撞到了什么 tag”，再把需要弯曲的手指和角度发给 Python / Arduino。
   * 在 SecondTouchReality 中，我们把重点挪到「**文本 → 物体**」：

     * `collect_data.py` 交互式采集「描述文本 + 标签」；
     * `clean_dataset.py` 过滤掉非英文样本，只保留 `text`/`label` 两列；
     * `train_model.py` 使用 `HashingVectorizer + SGDClassifier` 训练一个轻量级多分类模型，生成 `text_model.pkl`；
     * `text_infer_server.py` 打开一个 TCP 服务，接收 Unity 发来的描述句子，实时返回预测标签。
   * Unity 端的 `TextQueryClient_TMP.cs` 负责：

     * 弹出 TMP 输入框；
     * 把用户输入通过 TCP 发给 Python 文本服务器；
     * 收到标签后调用 `ModelLibrary.ShowModelByLabel()` 激活对应 3D 模型；
     * 在 UI 上显示结果，并弹出一个短暂的「成功」面板。

4. **硬件链路：从多路舵机收紧钢丝 → 单通道 pinch 信号**

   * OneTouchReality 的最终目标是 5 路舵机拉线，模拟指尖触觉，需要一整套机械结构、束线器、弹簧和防缠绕设计。
   * 在这个派生项目里，我们先把**信号链路打通**：

     * `hand_udp.py` 在每一帧 JSON 里标记每只手是否处于 pinch；
     * `arduino_udp_receive.py` 或 `main.py` 里注册一个回调：一旦有手从“未 pinch → pinch”或反向变化，就通过串口发出单个字符命令：`'1'` 代表抓紧，`'0'` 代表松开。
     * Arduino 端运行极简的舵机代码：收到 `'1'` 就转到 45°、收到 `'0'` 回到 0°。
   * 这样一来，你只要把一根线绑到指尖，就能 **从摄像头 → Python → Unity → Arduino → 手指** 完成一条闭环，用来验证延迟、稳定性和安全性，然后再逐步扩展到多路舵机和更复杂的机械结构。

5. **整合：`main.py` = 一键启动全链路**

   * `main.py`（实质是 `combined_server.py`）在一个进程里同时做四件事：

     1. 启动 `tools.hand_udp`：摄像头 + MediaPipe + 手部深度 + pinch 检测，通过 UDP 把 JSON 发给 Unity。
     2. 在 `on_payload` 回调里提取 pinch 状态，按变化边沿向 Arduino 发送 `'0'/'1'`。
     3. 启动文本推理 TCP 服务器，让 Unity 可以用自然语言请求某个模型。
     4. 统一管理串口生命周期和异常（串口断开时自动重试 / 安全关闭）。

从概念上看，SecondTouchReality 是在 OneTouchReality 的长链路上拆出了一条**可教学、可扩展、可快速实验**的小型主干：

* 摄像头 → 手部几何与深度
* 文本描述 → 物体标签
* Unity 场景 → 抓取 / 视角 / 教学关卡
* pinch → 单舵机或多舵机反馈

未来可以再把这些模块重新拼回一个“完整版”的力反馈手套系统，也可以直接用作**AI+互动教学平台** 的底座。


---

`SecondTouchReality` 是一个“全链路玩具原型”：所有环节都足够简单，可以随意改动；又足够完整，可以真正体验到从摄像头到虚拟物体再到硬件反馈的一条龙。

