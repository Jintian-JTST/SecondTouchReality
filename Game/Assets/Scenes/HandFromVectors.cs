using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

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

    [Header("Bone Lengths")]
    [SerializeField]
    private float[] boneLengths = new float[20];

    private const int MaxHands = 5;

    [Header("Grab / Pinch Object")]
    public bool enableGrabSphere = true;
    public GameObject grabSpherePrefab;     
    public float grabSphereRadius = 0.03f;  
    public int grabJointIndex = 8;          


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

        public bool is_left;
        public string hand_label;
        public float hand_score;

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
        public float[] dir; 
    }

    private readonly (int from, int to)[] bonePairs = new (int, int)[]
    {
        (0, 1), (1, 2), (2, 3), (3, 4),    
        (0, 5), (5, 6), (6, 7), (7, 8),    
        (0, 9), (9, 10), (10, 11), (11, 12), 
        (0, 13), (13, 14), (14, 15), (15, 16), 
        (0, 17), (17, 18), (18, 19), (19, 20)  
    };

    private UdpClient udp;
    private IPEndPoint remoteEndPoint;

    private GameObject[,] jointObjects;   
    private Vector3[,] jointPositions;   
    private LineRenderer[,] boneLines;  

    private bool[] hasHand = new bool[MaxHands]; 
    private bool[] currentPinch = new bool[MaxHands]; 
    private bool[] isLeftHand = new bool[MaxHands]; 
    private HandData[] latestHands;
    private readonly object handLock = new object();
    private GameObject grabSphere;
    private Rigidbody grabSphereRb;
    private int grabbedHand = -1;       
    private bool[] lastPinch = new bool[MaxHands];

    private Rect guiWindowRect = new Rect(10, 10, 260, 420);
    private Vector2 guiScroll = Vector2.zero;

    void Awake()
    {
        if (targetCamera == null)
            targetCamera = Camera.main;

        InitDefaultBoneLengths();

        jointObjects = new GameObject[MaxHands, 21];
        jointPositions = new Vector3[MaxHands, 21];
        boneLines = new LineRenderer[MaxHands, bonePairs.Length];

        for (int h = 0; h < MaxHands; h++)
        {
            var handRoot = new GameObject("Hand_" + h);
            handRoot.transform.SetParent(transform, false);

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

    private void UpdateHandPoseFromData()
    {
        HandData[] handsCopy = null;
        lock (handLock)
        {
            if (latestHands != null)
                handsCopy = (HandData[])latestHands.Clone();
        }

        for (int i = 0; i < MaxHands; i++)
        {
            hasHand[i] = false;
            currentPinch[i] = false;
            isLeftHand[i] = false;
        }


        if (handsCopy == null || targetCamera == null)
        {
            if (grabSphereRb != null && grabbedHand != -1)
            {
                grabSphereRb.useGravity = true;
                grabbedHand = -1;
                for (int i = 0; i < MaxHands; i++) lastPinch[i] = false;
            }
            return;
        }


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
            if (hand == null || hand.wrist == null)
            {
                hasHand[h] = true;
                currentPinch[h] = hand.pinch;
                isLeftHand[h] = hand.is_left;
                continue;
            }

            hasHand[h] = true;
            currentPinch[h] = hand.pinch;

            Vector3 wristWorldPos = ComputeWristWorldPos(hand.wrist, targetCamera, depthScale);
            jointPositions[h, 0] = wristWorldPos;
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

            bool pinchNow = hand.pinch;
            bool pinchPrev = lastPinch[h];

            lastPinch[h] = pinchNow;
        }
    }

    private Vector3 ComputeWristWorldPos(WristData wrist, Camera cam, float depthScale)
    {
        float depth = wrist.depth_m;
        if (depth <= 0.0f)
            depth = 0.4f;

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

    public bool IsHandPinching(int handIndex)
    {
        if (handIndex < 0 || handIndex >= MaxHands) return false;
        return hasHand[handIndex] && currentPinch[handIndex];
    }

    public bool TryGetJointPosition(int handIndex, int jointIndex, out Vector3 position)
    {
        position = Vector3.zero;
        if (handIndex < 0 || handIndex >= MaxHands) return false;
        if (!hasHand[handIndex]) return false;
        if (jointIndex < 0 || jointIndex >= 21) return false;

        position = jointPositions[handIndex, jointIndex];
        return true;
    }


    public int MaxHandCount
    {
        get { return MaxHands; }
    }





    public bool IsLeftHand(int handIndex)
    {
        if (handIndex < 0 || handIndex >= MaxHands) return false;
        return hasHand[handIndex] && isLeftHand[handIndex];
    }

    public bool IsRightHand(int handIndex)
    {
        if (handIndex < 0 || handIndex >= MaxHands) return false;
        return hasHand[handIndex] && !isLeftHand[handIndex];
    }





    private void DrawBoneLengthWindow(int windowId)
    {
        GUILayout.BeginVertical();

        GUILayout.Label("Adjust Bone Lengths");
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
        if (GUILayout.Button("Reset to Default Lengths"))
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
