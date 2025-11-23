using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// 从 hand_two_hands_z_udp.py 发来的 UDP JSON 中读取：
///  - 多只手的掌根 3D 信息 + 20 条骨骼方向向量；
///  - 每只手一个 pinch 布尔量（拇指+食指是否捏合）；
/// 在 Unity 里:
///  1) 根据掌根归一化坐标 + depth_m 投影到 3D 空间 -> wristWorldPos；
///  2) 结合每一节骨骼长度 + 单位向量，重建 21 个关节位置；
///  3) 支持多只手：每只手 21 个小球 + 20 条骨骼线；
///  4) 如果某只手 pinch = true，就“抓住”一个球，让球跟着这只手的食指指尖走。
/// </summary>
public class HandFromVectors : MonoBehaviour
{
    [Header("UDP Settings")]
    public int listenPort = 5065;

    [Header("Camera & Projection")]
    public Camera targetCamera;
    public float depthScale = 1.0f;

    [Header("Hand Layout")]
    public float sphereRadius = 0.01f;
    public bool drawBones = true;

    [Header("Bone Lengths (按 BONE_PAIRS 顺序)")]
    [SerializeField]
    private float[] boneLengths = new float[20];

    // 同时支持的最大手数量
    private const int MaxHands = 5;

    [Header("Grab / Pinch Object")]
    public bool enableGrabSphere = true;
    public GameObject grabSpherePrefab;      // 可选：你可以在 Inspector 里拖自己做好的球
    public float grabSphereRadius = 0.03f;   // 球的半径（米）
    public int grabJointIndex = 8;           // 默认跟随食指指尖(关节 8)

    // ============ 内部数据结构（匹配 JSON） ============

    [Serializable]
    public class RootPayload
    {
        public double timestamp;
        public float fps;
        public int num_hands;
        public HandData[] hands;
    }

    [Serializable]
    public class HandData
    {
        public int hand_index;
        public bool pinch;
        public WristData wrist;
        public BoneData[] bones;
    }

    [Serializable]
    public class WristData
    {
        public Pixel pixel;
        public Normalized normalized;
        public float depth_m;
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
        public float[] dir;  // [dx, dy, dz]
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

    // 多手：21 关节 + 20 骨骼线
    private GameObject[,] jointObjects;   // [hand, jointId]
    private Vector3[,] jointPositions;    // [hand, jointId]
    private LineRenderer[,] boneLines;    // [hand, boneIndex]

    // 最新一帧收到的所有手
    private HandData[] latestHands;
    private readonly object handLock = new object();

    // 抓球相关
    private GameObject grabSphere;
    private Rigidbody grabSphereRb;
    private int grabbedHand = -1;         // -1 表示当前没人抓球
    private bool[] lastPinch = new bool[MaxHands];

    // GUI
    private Rect guiWindowRect = new Rect(10, 10, 260, 420);
    private Vector2 guiScroll = Vector2.zero;

    void Awake()
    {
        if (targetCamera == null)
            targetCamera = Camera.main;

        InitDefaultBoneLengths();

        // ---------- 初始化多手关节 & 骨骼 ----------
        jointObjects = new GameObject[MaxHands, 21];
        jointPositions = new Vector3[MaxHands, 21];
        boneLines = new LineRenderer[MaxHands, bonePairs.Length];

        for (int h = 0; h < MaxHands; h++)
        {
            var handRoot = new GameObject("Hand_" + h);
            handRoot.transform.SetParent(transform, false);

            // 21 个关节球
            for (int j = 0; j < 21; j++)
            {
                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = $"Hand{h}_Joint_{j}";
                sphere.transform.SetParent(handRoot.transform, false);
                sphere.transform.localScale = Vector3.one * sphereRadius * 2f;

                var col = sphere.GetComponent<Collider>();
                if (col != null) Destroy(col);

                jointObjects[h, j] = sphere;
            }

            // 20 条骨骼线
            for (int i = 0; i < bonePairs.Length; i++)
            {
                var go = new GameObject($"Hand{h}_Bone_{i}");
                go.transform.SetParent(handRoot.transform, false);

                var lr = go.AddComponent<LineRenderer>();
                lr.positionCount = 2;
                lr.useWorldSpace = true;
                lr.widthMultiplier = sphereRadius * 0.8f;
                lr.material = new Material(Shader.Find("Sprites/Default"));
                lr.startColor = Color.white;
                lr.endColor = Color.white;

                boneLines[h, i] = lr;
            }
        }

        // ---------- 抓球 ----------
        if (enableGrabSphere)
        {
            if (grabSpherePrefab != null)
            {
                grabSphere = Instantiate(grabSpherePrefab);
            }
            else
            {
                grabSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            }

            grabSphere.name = "GrabSphere";
            grabSphere.transform.localScale = Vector3.one * grabSphereRadius * 2f;

            grabSphereRb = grabSphere.GetComponent<Rigidbody>();
            if (grabSphereRb == null)
                grabSphereRb = grabSphere.AddComponent<Rigidbody>();

            grabSphereRb.useGravity = true;
            grabSphereRb.mass = 0.1f;
        }

        // ---------- UDP ----------
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

    // ========== 初始化默认骨骼长度 ==========
    private void InitDefaultBoneLengths()
    {
        if (boneLengths == null || boneLengths.Length != 20)
            boneLengths = new float[20];

        bool allZero = true;
        for (int i = 0; i < boneLengths.Length; i++)
        {
            if (Mathf.Abs(boneLengths[i]) > 1e-6f)
            {
                allZero = false;
                break;
            }
        }
        if (!allZero) return;

        boneLengths[0]  = 0.038f;
        boneLengths[1]  = 0.032f;
        boneLengths[2]  = 0.037f;
        boneLengths[3]  = 0.027f;

        boneLengths[4]  = 0.077f;
        boneLengths[5]  = 0.047f;
        boneLengths[6]  = 0.025f;
        boneLengths[7]  = 0.018f;

        boneLengths[8]  = 0.070f;
        boneLengths[9]  = 0.050f;
        boneLengths[10] = 0.030f;
        boneLengths[11] = 0.022f;

        boneLengths[12] = 0.065f;
        boneLengths[13] = 0.045f;
        boneLengths[14] = 0.028f;
        boneLengths[15] = 0.022f;

        boneLengths[16] = 0.066f;
        boneLengths[17] = 0.032f;
        boneLengths[18] = 0.021f;
        boneLengths[19] = 0.022f;
    }

    // ========== UDP 接收并解析 JSON ==========
    private void ReceiveUdpPackets()
    {
        if (udp == null) return;

        while (udp.Available > 0)
        {
            try
            {
                byte[] data = udp.Receive(ref remoteEndPoint);
                string json = Encoding.UTF8.GetString(data);

                RootPayload root = JsonUtility.FromJson<RootPayload>(json);
                if (root != null && root.hands != null && root.hands.Length > 0)
                {
                    lock (handLock)
                    {
                        latestHands = root.hands;
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("UDP/JSON error: " + e.Message);
            }
        }
    }

    // ========== 根据 latestHands 更新所有手的关节 + 抓球 ==========
    private void UpdateHandPoseFromData()
    {
        HandData[] handsCopy = null;
        lock (handLock)
        {
            if (latestHands != null)
                handsCopy = (HandData[])latestHands.Clone();
        }

        if (handsCopy == null || targetCamera == null)
        {
            // 没有手的时候，把球放手，让它掉下去
            if (grabSphereRb != null && grabbedHand != -1)
            {
                grabSphereRb.useGravity = true;
                grabbedHand = -1;
                for (int i = 0; i < MaxHands; i++) lastPinch[i] = false;
            }
            return;
        }

        // 先清空所有手的显示
        for (int h = 0; h < MaxHands; h++)
        {
            for (int j = 0; j < 21; j++)
            {
                var go = jointObjects[h, j];
                if (go != null) go.SetActive(false);
            }
            for (int b = 0; b < bonePairs.Length; b++)
            {
                var lr = boneLines[h, b];
                if (lr != null) lr.enabled = false;
            }
        }

        int handCount = Mathf.Min(handsCopy.Length, MaxHands);

        for (int h = 0; h < handCount; h++)
        {
            HandData hand = handsCopy[h];
            if (hand == null || hand.wrist == null) continue;

            // 1) 掌根世界坐标
            Vector3 wristWorldPos = ComputeWristWorldPos(hand.wrist, targetCamera, depthScale);
            jointPositions[h, 0] = wristWorldPos;

            // 2) 骨骼链条重建 21 个关节
            if (hand.bones != null && hand.bones.Length > 0)
            {
                int boneCount = Mathf.Min(hand.bones.Length, bonePairs.Length);
                for (int i = 0; i < boneCount; i++)
                {
                    BoneData bone = hand.bones[i];
                    var pair = bonePairs[i];
                    int from = pair.from;
                    int to = pair.to;

                    Vector3 dirCam = Vector3.zero;
                    if (bone.dir != null && bone.dir.Length >= 3)
                    {
                        float dx = bone.dir[0];
                        float dy = bone.dir[1];
                        float dz = bone.dir[2];

                        // Python: x 右, y 下, z 朝相机(负)
                        // Unity Camera: x 右, y 上, z 向前(正)
                        dirCam = new Vector3(dx, -dy, -dz);
                        if (dirCam.sqrMagnitude > 1e-6f)
                            dirCam.Normalize();
                    }

                    float length = (i < boneLengths.Length) ? boneLengths[i] : 0.03f;
                    length = Mathf.Max(0.0f, length);

                    Vector3 parentPos = jointPositions[h, from];
                    Vector3 dirWorld = targetCamera.transform.TransformDirection(dirCam);
                    Vector3 childPos = parentPos + dirWorld * length;

                    jointPositions[h, to] = childPos;
                }
            }

            // 3) 画关节球 + 骨骼线
            Color sphereColor = hand.pinch ? Color.yellow : Color.white;

            for (int j = 0; j < 21; j++)
            {
                var sphere = jointObjects[h, j];
                if (sphere == null) continue;

                sphere.SetActive(true);
                sphere.transform.position = jointPositions[h, j];

                var renderer = sphere.GetComponent<Renderer>();
                if (renderer != null)
                    renderer.material.color = sphereColor;
            }

            for (int i = 0; i < bonePairs.Length; i++)
            {
                var lr = boneLines[h, i];
                if (lr == null) continue;

                if (drawBones)
                {
                    lr.enabled = true;
                    int from = bonePairs[i].from;
                    int to = bonePairs[i].to;
                    lr.SetPosition(0, jointPositions[h, from]);
                    lr.SetPosition(1, jointPositions[h, to]);
                }
                else
                {
                    lr.enabled = false;
                }
            }

            // 4) 抓球逻辑：检测 pinch 的前一帧/当前帧
            bool pinchNow = hand.pinch;
            bool pinchPrev = lastPinch[h];

            if (enableGrabSphere && grabSphere != null && grabSphereRb != null)
            {
                // 刚从没捏 -> 捏：开始抓球
                if (!pinchPrev && pinchNow)
                {
                    grabbedHand = h;
                    grabSphereRb.useGravity = false;
                    grabSphereRb.velocity = Vector3.zero;

                    if (grabJointIndex < 0 || grabJointIndex > 20)
                        grabJointIndex = 8; // 保底

                    grabSphere.transform.position = jointPositions[h, grabJointIndex];
                }

                // 正在被这只手捏着：球跟随这只手的指定关节（默认食指尖）
                if (grabbedHand == h && pinchNow)
                {
                    if (grabJointIndex >= 0 && grabJointIndex <= 20)
                        grabSphere.transform.position = jointPositions[h, grabJointIndex];
                }

                // 刚从捏 -> 松手：放开球，让球掉下去
                if (grabbedHand == h && pinchPrev && !pinchNow)
                {
                    grabbedHand = -1;
                    grabSphereRb.useGravity = true;
                }
            }

            lastPinch[h] = pinchNow;
        }
    }

    /// <summary>
    /// 把 wrist 的归一化坐标 + 深度(m) 转成 3D 世界坐标。
    /// </summary>
    private Vector3 ComputeWristWorldPos(WristData wrist, Camera cam, float depthScale)
    {
        float depth = wrist.depth_m;
        if (depth <= 0.0f)
            depth = 0.4f; // 没标定时默认 40cm

        depth *= depthScale;

        float nx = wrist.normalized.x;
        float ny = wrist.normalized.y;

        float vHalfAngle = 0.5f * cam.fieldOfView * Mathf.Deg2Rad;
        float halfHeight = Mathf.Tan(vHalfAngle) * depth;
        float halfWidth = halfHeight * cam.aspect;

        float xCam = (nx - 0.5f) * 2f * halfWidth;
        float yCam = (0.5f - ny) * 2f * halfHeight;
        float zCam = depth;

        Vector3 posCam = new Vector3(xCam, yCam, zCam);
        return cam.transform.TransformPoint(posCam);
    }

    // ========== OnGUI: 调整骨骼长度的小窗口 ==========
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

        string[] fingerNames = { "Thumb", "Index", "Middle", "Ring", "Pinky" };
        int boneIndex = 0;

        for (int f = 0; f < 5; f++)
        {
            GUILayout.Label(fingerNames[f], EditorLabelStyle());
            for (int s = 0; s < 4; s++)
            {
                if (boneIndex >= boneLengths.Length) break;

                GUILayout.BeginHorizontal();
                GUILayout.Label($"  Bone {boneIndex:00}:", GUILayout.Width(80));
                float newLen = GUILayout.HorizontalSlider(boneLengths[boneIndex], 0.0f, 0.15f);
                newLen = Mathf.Round(newLen * 1000f) / 1000f;
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
            InitDefaultBoneLengths();

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
