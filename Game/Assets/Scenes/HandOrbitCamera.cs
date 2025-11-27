using UnityEngine;

/// <summary>
/// 用“左手 pinch”控制视角：
/// - 左手 + pinch：绕 target 旋转 + 缩放；
/// - 右手：继续由 PinchGrabBall 控制物体本身。
/// 依赖 HandFromVectors：
///   - TryGetJointPosition(handIndex, jointIndex, out pos)
///   - IsHandPinching(handIndex)
///   - IsLeftHand(handIndex)
/// </summary>
public class HandOrbitCamera : MonoBehaviour
{
    [Header("必填引用")]
    public HandFromVectors handTracker;   // 场景里挂 HandFromVectors 的物体
    public Camera orbitCamera;            // 一般就是 Main Camera
    public Transform target;              // 围绕看的对象（模型或一个空物体）

    [Header("旋转参数")]
    [Tooltip("左手在屏幕上横向移动 1.0（整屏宽）对应的水平旋转角度")]
    public float horizontalDegrees = 200f;

    [Tooltip("左手在屏幕上纵向移动 1.0（整屏高）对应的垂直旋转角度")]
    public float verticalDegrees = 150f;

    [Tooltip("用哪个关节当控制点：8 = 食指指尖，0 = 掌根")]
    public int controlJointIndex = 8;

    [Tooltip("屏幕移动小于这个值时忽略（防抖），单位：viewport")]
    public float moveDeadZone = 0.002f;

    [Header("缩放参数")]
    public bool enableZoom = true;

    [Tooltip("左手沿着深度方向移动 0.1 米，对应半径变化多少米")]
    public float zoomSensitivity = 1.0f;

    [Tooltip("相机距离 target 的最小半径")]
    public float minRadius = 0.3f;

    [Tooltip("相机距离 target 的最大半径")]
    public float maxRadius = 5.0f;

    [Tooltip("深度变化小于这个值时忽略，单位：米")]
    public float depthDeadZone = 0.01f;

    // 内部状态
    private bool isOrbiting = false;
    private int currentLeftHandIndex = -1;
    private Vector3 lastViewportPos;
    private float lastDepth;
    private float currentRadius;

    void Start()
    {
        if (orbitCamera == null)
            orbitCamera = Camera.main;

        if (orbitCamera != null && target != null)
            currentRadius = Vector3.Distance(orbitCamera.transform.position, target.position);
        else
            currentRadius = 1.0f;

        if (handTracker == null)
        {
            Debug.LogWarning("HandOrbitCamera: handTracker 没有设置。");
        }

        if (target == null)
        {
            Debug.LogWarning("HandOrbitCamera: target 没有设置，视角不会围绕任何东西。");
        }
    }

    void Update()
    {
        if (handTracker == null || orbitCamera == null || target == null)
            return;

        // 1. 选择“左手”：根据 HandFromVectors 的 IsLeftHand
        int leftHand = -1;
        Vector3 leftJointWorld = Vector3.zero;
        Vector3 leftJointViewport = Vector3.zero;

        int maxHands = handTracker.MaxHandCount;
        for (int h = 0; h < maxHands; h++)
        {
            if (!handTracker.IsLeftHand(h))
                continue;

            if (!handTracker.TryGetJointPosition(h, controlJointIndex, out Vector3 jointWorld))
                continue;

            Vector3 vp = orbitCamera.WorldToViewportPoint(jointWorld);
            if (vp.z <= 0f)
                continue; // 在相机后面

            // 理论上只有一只左手，找到第一只就够了
            leftHand = h;
            leftJointWorld = jointWorld;
            leftJointViewport = vp;
            break;
        }

        bool hasLeftHand = (leftHand != -1);
        bool leftPinching = hasLeftHand && handTracker.IsHandPinching(leftHand);

        if (!hasLeftHand || !leftPinching)
        {
            // 没有左手 或 左手没捏 -> 退出旋转模式
            isOrbiting = false;
            currentLeftHandIndex = -1;
            return;
        }

        // 2. 左手刚开始 pinch：记录当前位置 & 深度，不立即旋转
        if (!isOrbiting || leftHand != currentLeftHandIndex)
        {
            isOrbiting = true;
            currentLeftHandIndex = leftHand;
            lastViewportPos = leftJointViewport;

            // 用相机局部坐标系的 z 记录深度
            Vector3 camLocal = orbitCamera.transform.InverseTransformPoint(leftJointWorld);
            lastDepth = camLocal.z;

            // 同步一下当前半径
            currentRadius = Vector3.Distance(orbitCamera.transform.position, target.position);
            return;
        }

        // 3. 计算屏幕移动量 -> 旋转
        Vector3 deltaVp = leftJointViewport - lastViewportPos;
        lastViewportPos = leftJointViewport;

        float yawDeg = 0f;
        float pitchDeg = 0f;

        if (deltaVp.sqrMagnitude >= moveDeadZone * moveDeadZone)
        {
            yawDeg = deltaVp.x * horizontalDegrees;
            pitchDeg = -deltaVp.y * verticalDegrees;
        }

        // 4. 计算深度变化 -> 缩放
        float zoomDelta = 0f;
        if (enableZoom)
        {
            Vector3 camLocal = orbitCamera.transform.InverseTransformPoint(leftJointWorld);
            float depth = camLocal.z; // 正值：在相机前方
            float depthDelta = depth - lastDepth;
            lastDepth = depth;

            if (Mathf.Abs(depthDelta) >= depthDeadZone)
            {
                // 手向前伸（离相机更远） -> depth 增大 -> 半径增大 -> 画面变小（缩小）
                // 如果你想反过来，可以改成 zoomDelta = -depthDelta;
                zoomDelta = depthDelta;
            }
        }

        OrbitAroundTarget(yawDeg, pitchDeg, zoomDelta);
    }

    private void OrbitAroundTarget(float yawDeg, float pitchDeg, float zoomDelta)
    {
        if (orbitCamera == null || target == null)
            return;

        Vector3 pivot = target.position;
        Vector3 dir = orbitCamera.transform.position - pivot;

        if (dir.sqrMagnitude < 1e-6f)
            dir = orbitCamera.transform.forward * -1.0f;

        float radius = dir.magnitude;
        if (radius <= 0.0001f)
            radius = Mathf.Max(currentRadius, 0.1f);

        currentRadius = radius;

        // 先旋转方向向量（绕世界 Y 轴 + 相机右轴）
        Quaternion yawRot = Quaternion.AngleAxis(yawDeg, Vector3.up);
        Quaternion pitchRot = Quaternion.AngleAxis(pitchDeg, orbitCamera.transform.right);

        dir = yawRot * dir;
        dir = pitchRot * dir;

        // 再根据 zoomDelta 调整半径
        if (enableZoom)
        {
            currentRadius = Mathf.Clamp(
                currentRadius + zoomDelta * zoomSensitivity,
                minRadius,
                maxRadius
            );
        }

        orbitCamera.transform.position = pivot + dir.normalized * currentRadius;
        orbitCamera.transform.LookAt(pivot, Vector3.up);
    }
}
