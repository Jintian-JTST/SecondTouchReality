using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// 从 hand_easy.py 发来的 UDP JSON 中读取：
///  - 掌根像素 + 归一化坐标 + 深度（米）
///  - 20 条骨骼方向向量（单位向量）
/// 在 Unity 里:
///  1) 根据掌根像素 + depth_m 投影到 3D 空间 -> wristWorldPos；
///  2) 结合每一节骨骼长度（可在窗口里调） + 单位向量，重建 21 个关节位置；
///  3) 用 21 个小球显示出来；
///  4) 提供一个 OnGUI 面板调整每一节骨头的长度。
/// </summary>
public class HandFromVectors : MonoBehaviour
{
    [Header("UDP Settings")]
    public int listenPort = 5065;

    [Header("Camera & Projection")]
    public Camera targetCamera;        // 用来把 (nx,ny,depth) 投成 3D
    public float depthScale = 1.0f;    // Python 的米 -> Unity 单位缩放

    [Header("Hand Layout")]
    public float sphereRadius = 0.01f; // 关节小球半径
    public bool drawBones = true;      // 是否在关节间画线

    // 20 条骨头长度（按 BONE_PAIRS 顺序），单位: Unity units (通常当作米)
    [SerializeField]
    private float[] boneLengths = new float[20];

    // ============ 内部数据结构（匹配 JSON） ============

    [Serializable]
    public class RootPayload
    {
        public double timestamp;
        public float fps;
        public HandData[] hands;
    }

    [Serializable]
    public class HandData
    {
        public int hand_index;
        public WristData wrist;
        public BoneData[] bones;
    }

    [Serializable]
    public class WristData
    {
        public Pixel pixel;
        public Normalized normalized;
        public float depth_m; // 可能为 0，如果没标定就别太信
    }

    [Serializable]
    public class Pixel
    {
        public int x;
        public int y;
    }

    [Serializable]
    public class Normalized
    {
        public float x;
        public float y;
        public float z;
    }

    [Serializable]
    public class BoneData
    {
        public int id;
        public int from;
        public int to;
        public float[] dir;  // 长度为 3 的数组 [dx, dy, dz]
    }

    // 与 Python 一致的骨骼拓扑
    private readonly (int from, int to)[] bonePairs = new (int, int)[]
    {
        (0, 1), (1, 2), (2, 3), (3, 4),        // 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),        // 食指
        (0, 9), (9, 10), (10, 11), (11, 12),   // 中指
        (0, 13), (13, 14), (14, 15), (15, 16), // 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  // 小指
    };

    // UDP
    private UdpClient udp;
    private IPEndPoint remoteEndPoint;

    // 手的 21 个关节 GameObject
    private GameObject[] jointObjects;
    private Vector3[] jointPositions = new Vector3[21];

    // 最新一帧从 Python 收到的 hand 数据
    private HandData latestHand;
    private object handLock = new object();

    // GUI 窗口
    private Rect guiWindowRect = new Rect(10, 10, 260, 420);
    private Vector2 guiScroll = Vector2.zero;

    void Awake()
    {
        if (targetCamera == null)
        {
            targetCamera = Camera.main;
        }

        // 初始化骨长（如果还是全 0，就给一些默认值）
        InitDefaultBoneLengths();

        // 初始化 21 个关节小球
        jointObjects = new GameObject[21];
        for (int i = 0; i < 21; i++)
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "Joint_" + i;
            sphere.transform.SetParent(transform, false);
            sphere.transform.localScale = Vector3.one * sphereRadius * 2f;
            Destroy(sphere.GetComponent<Collider>());
            jointObjects[i] = sphere;
        }

        // 初始化 UDP
        udp = new UdpClient(listenPort);
        udp.Client.Blocking = false;
        remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
    }

    void OnDestroy()
    {
        if (udp != null)
        {
            udp.Close();
            udp = null;
        }
    }

    void Update()
    {
        ReceiveUdpPackets();
        UpdateHandPoseFromData();
    }

    // ========== 初始化骨骼长度 ==========

    private void InitDefaultBoneLengths()
    {
        if (boneLengths == null || boneLengths.Length != 20)
        {
            boneLengths = new float[20];
        }

        bool allZero = true;
        for (int i = 0; i < boneLengths.Length; i++)
        {
            if (Mathf.Abs(boneLengths[i]) > 1e-6f)
            {
                allZero = false;
                break;
            }
        }

        if (allZero)
        {
            // 给一个比较合理的默认长度（单位：米）
            // 拇指: 根 (0-1) 稍长，其余略短
            boneLengths[0] = 0.035f; // 0-1
            boneLengths[1] = 0.025f; // 1-2
            boneLengths[2] = 0.020f; // 2-3
            boneLengths[3] = 0.018f; // 3-4

            // 食指: 4 节
            boneLengths[4] = 0.045f; // 0-5
            boneLengths[5] = 0.030f; // 5-6
            boneLengths[6] = 0.025f; // 6-7
            boneLengths[7] = 0.020f; // 7-8

            // 中指: 稍长一点
            boneLengths[8]  = 0.050f; // 0-9
            boneLengths[9]  = 0.032f; // 9-10
            boneLengths[10] = 0.027f; // 10-11
            boneLengths[11] = 0.022f; // 11-12

            // 无名指
            boneLengths[12] = 0.047f;
            boneLengths[13] = 0.030f;
            boneLengths[14] = 0.025f;
            boneLengths[15] = 0.020f;

            // 小指: 略短
            boneLengths[16] = 0.043f;
            boneLengths[17] = 0.028f;
            boneLengths[18] = 0.022f;
            boneLengths[19] = 0.018f;
        }
    }

    // ========== 接收 UDP 并解析 JSON ==========

    private void ReceiveUdpPackets()
    {
        if (udp == null) return;

        // 非阻塞：有多少包就收多少，避免积压
        while (udp.Available > 0)
        {
            try
            {
                byte[] data = udp.Receive(ref remoteEndPoint);
                string json = Encoding.UTF8.GetString(data);

                RootPayload root = JsonUtility.FromJson<RootPayload>(json);
                if (root.hands != null && root.hands.Length > 0)
                {
                    lock (handLock)
                    {
                        latestHand = root.hands[0]; // 这里先只用第一只手
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("UDP/JSON error: " + e.Message);
            }
        }
    }

    // ========== 根据 latestHand 更新 21 个关节位置 ==========

    private void UpdateHandPoseFromData()
    {
        HandData handCopy = null;
        lock (handLock)
        {
            if (latestHand != null)
            {
                handCopy = latestHand;
            }
        }

        if (handCopy == null) return;
        if (targetCamera == null) return;
        if (handCopy.bones == null || handCopy.bones.Length == 0) return;

        // 1) 用 wrist 像素 + depth_m 估计 3D 掌根位置（世界坐标）
        Vector3 wristWorldPos = ComputeWristWorldPos(
            handCopy.wrist,
            targetCamera,
            depthScale
        );

        jointPositions[0] = wristWorldPos;

        // 2) 用骨骼链条依次重建 21 个点
        int boneCount = Mathf.Min(handCopy.bones.Length, bonePairs.Length);
        for (int i = 0; i < boneCount; i++)
        {
            BoneData bone = handCopy.bones[i];
            var pair = bonePairs[i];

            int from = pair.from;
            int to = pair.to;

            // 方向向量来自 Python:
            //   MediaPipe 坐标: x 右, y 下, z 朝相机(负)
            // Unity 摄像机坐标: x 右, y 上, z 向前(正)
            // 简单处理: 保持 x，翻转 y & z
            Vector3 dirCamSpace = Vector3.zero;
            if (bone.dir != null && bone.dir.Length >= 3)
            {
                float dx = bone.dir[0];
                float dy = bone.dir[1];
                float dz = bone.dir[2];
                dirCamSpace = new Vector3(dx, -dy, -dz);
                dirCamSpace.Normalize();
            }

            float length = (i < boneLengths.Length) ? boneLengths[i] : 0.03f;
            length = Mathf.Max(0.0f, length);

            Vector3 parentPos = jointPositions[from];
            // 把摄像机坐标方向转成世界坐标方向
            Vector3 dirWorld = targetCamera.transform.TransformDirection(dirCamSpace);
            Vector3 childPos = parentPos + dirWorld * length;

            jointPositions[to] = childPos;
        }

        // 3) 把 21 个小球摆到相应位置
        for (int i = 0; i < jointObjects.Length; i++)
        {
            if (jointObjects[i] != null)
            {
                jointObjects[i].transform.position = jointPositions[i];
            }
        }
    }

    /// <summary>
    /// 把 wrist 的归一化坐标 + 深度(m) 转成 3D 世界坐标
    /// 这里使用摄像机的 FOV 和深度做简单投影，假设 depth_m 大概就是到摄像机的距离。
    /// </summary>
    private Vector3 ComputeWristWorldPos(WristData wrist, Camera cam, float depthScale)
    {
        float depth = wrist.depth_m;
        if (depth <= 0.0f)
        {
            // 没标定时给个默认前方距离
            depth = 0.4f;
        }
        depth *= depthScale;

        float nx = wrist.normalized.x; // [0,1] 左->右
        float ny = wrist.normalized.y; // [0,1] 上->下

        // 垂直视角的一半
        float vHalfAngle = 0.5f * cam.fieldOfView * Mathf.Deg2Rad;
        float halfHeight = Mathf.Tan(vHalfAngle) * depth;
        float halfWidth = halfHeight * cam.aspect;

        // 把 [0,1] 映射到 [-halfWidth, +halfWidth], [-halfHeight, +halfHeight]
        float xCam = (nx - 0.5f) * 2f * halfWidth;
        float yCam = (0.5f - ny) * 2f * halfHeight; // 注意 Unity 相机 y 轴朝上
        float zCam = depth;

        Vector3 posCamSpace = new Vector3(xCam, yCam, zCam);

        // 转成世界坐标
        Vector3 posWorld = cam.transform.TransformPoint(posCamSpace);
        return posWorld;
    }

    // ========== OnGUI: 调整每节骨头长度的窗口 ==========

    void OnGUI()
    {
        guiWindowRect = GUI.Window(
            12345,
            guiWindowRect,
            DrawBoneLengthWindow,
            "Hand Bone Lengths"
        );
    }

    private void DrawBoneLengthWindow(int windowId)
    {
        GUILayout.BeginVertical();

        GUILayout.Label("调节每节手指骨长度（单位：米）");
        GUILayout.Space(5);

        guiScroll = GUILayout.BeginScrollView(guiScroll, false, true);

        // 简单地按顺序分组显示
        string[] fingerNames = { "Thumb", "Index", "Middle", "Ring", "Pinky" };
        int boneIndex = 0;

        for (int f = 0; f < 5; f++)
        {
            GUILayout.Label(fingerNames[f], EditorLabelStyle());
            for (int s = 0; s < 4; s++)
            {
                if (boneIndex >= boneLengths.Length) break;

                GUILayout.BeginHorizontal();
                GUILayout.Label(string.Format("  Bone {0:00}:", boneIndex), GUILayout.Width(80));
                float newLen = GUILayout.HorizontalSlider(boneLengths[boneIndex], 0.0f, 0.15f);
                newLen = Mathf.Round(newLen * 1000f) / 1000f; // 保留 3 位小数
                GUILayout.Label(newLen.ToString("0.000"), GUILayout.Width(50));
                GUILayout.EndHorizontal();

                boneLengths[boneIndex] = newLen;
                boneIndex++;
            }

            GUILayout.Space(4);
        }

        GUILayout.EndScrollView();

        GUILayout.Space(5);
        if (GUILayout.Button("重置为默认长度"))
        {
            InitDefaultBoneLengths();
        }

        GUILayout.EndVertical();

        GUI.DragWindow(new Rect(0, 0, 10000, 20));
    }

    private GUIStyle EditorLabelStyle()
    {
        var style = new GUIStyle(GUI.skin.label);
        style.fontStyle = FontStyle.Bold;
        return style;
    }
}
