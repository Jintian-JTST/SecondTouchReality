using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// 从 Python 手部 UDP（基于 rec + hand_easy）重建一只 3D 手。
/// 挂在 HandRoot 上即可。
/// </summary>
public class HandReconstructionFromUDP : MonoBehaviour
{
    [Header("UDP config")]
    public int listenPort = 5065;

    [Header("Hand layout")]
    public int landmarkCount = 21;
    public float handPlaneWidth = 0.20f;   // 0.2m 宽
    public float handPlaneHeight = 0.20f;  // 0.2m 高
    public float depthScale = 1.0f;        // 米 -> Unity 单位 的缩放
    public float zRelScale = 0.05f;        // 相对 z_rel 的缩放系数

    [Header("Sphere appearance")]
    public float sphereRadius = 0.01f;

    // --- 内部状态 ---
    UdpClient udp;
    IPEndPoint remoteEP;

    GameObject[] joints;
    bool handInitialized = false;

    // 用来匹配 JSON 结构的类（要和 Python 发的一致）
    [Serializable]
    public class Root
    {
        public double timestamp;
        public float fps;
        public HandData[] hands;
    }

    [Serializable]
    public class HandData
    {
        public int hand_index;
        public Landmark[] landmarks;
        public float wrist_z_m;  // hand_easy 融合后的掌根距离
    }

    [Serializable]
    public class Landmark
    {
        public int id;
        public Normalized normalized;
        // pixel 字段其实不用解析也可以，为了完整性保留
        // public Pixel pixel;
        public float z_rel;
        public float z_vis;
        // public float z_m; // 可以忽略
    }

    [Serializable]
    public class Normalized
    {
        public float x;
        public float y;
        public float z;
    }

    // 如果你想用像素信息，可以再加：
    // [Serializable]
    // public class Pixel
    // {
    //     public int x;
    //     public int y;
    // }

    void Start()
    {
        // 初始化 UDP（非阻塞）
        udp = new UdpClient(listenPort);
        udp.Client.Blocking = false;
        remoteEP = new IPEndPoint(IPAddress.Any, 0);

        joints = new GameObject[landmarkCount];
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
        // 1. 尝试收一帧数据（非阻塞，有就收，没有就算了）
        if (udp == null) return;

        while (udp.Available > 0)
        {
            byte[] data = udp.Receive(ref remoteEP);
            string json = Encoding.UTF8.GetString(data);

            try
            {
                Root root = JsonUtility.FromJson<Root>(json);
                if (root.hands == null || root.hands.Length == 0)
                    continue;

                // 这里简单地只用第一只手
                HandData hand = root.hands[0];
                if (hand.landmarks == null || hand.landmarks.Length < landmarkCount)
                    continue;

                // 第一次收到时，创建 21 个小球
                if (!handInitialized)
                {
                    InitHandObjects();
                    handInitialized = true;
                }

                UpdateHandPose(hand);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"JSON parse error: {e.Message}");
            }
        }
    }

    /// <summary>
    /// 创建 21 个小球作为关节可视化。
    /// </summary>
    void InitHandObjects()
    {
        for (int i = 0; i < landmarkCount; i++)
        {
            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "Joint_" + i;
            sphere.transform.SetParent(this.transform, false);
            sphere.transform.localScale = Vector3.one * sphereRadius * 2f;

            // 为了简单一点，把 Collider 删掉
            Destroy(sphere.GetComponent<Collider>());

            joints[i] = sphere;
        }
    }

    /// <summary>
    /// 根据一帧 HandData 更新 21 个关节的位置。
    /// </summary>
    void UpdateHandPose(HandData hand)
    {
        float wristZ = hand.wrist_z_m; // 米
        if (wristZ <= 0f)
        {
            // 没标定好时，先用默认值
            wristZ = 0.4f;
        }

        float depthBase = wristZ * depthScale;

        for (int i = 0; i < landmarkCount; i++)
        {
            Landmark lm = hand.landmarks[i];
            if (joints[i] == null) continue;

            // MediaPipe 归一化坐标：
            // x: [0,1] 左到右
            // y: [0,1] 上到下
            float nx = lm.normalized.x;
            float ny = lm.normalized.y;
            float zRel = lm.z_rel; // 通常是负的（朝向相机）

            // 映射到局部平面坐标，以 (0,0) 为屏幕中心
            float xLocal = (nx - 0.5f) * handPlaneWidth;
            float yLocal = (0.5f - ny) * handPlaneHeight;

            // 相对 z：让它在掌根距离附近稍微有一点前后起伏
            float zOffset = -zRel * zRelScale;  // 负的 z_rel 表示更靠近相机，这里取反
            float zLocal = depthBase + zOffset;

            joints[i].transform.localPosition = new Vector3(xLocal, yLocal, zLocal);
        }
    }
}
