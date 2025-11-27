using UnityEngine;

/// <summary>
/// 非左右手区分的“通用相机控制”脚本：
/// - 不再区分左手/右手，所有手完全对等；
/// - 当有物体被 PinchGrabBall 抓住时：由抓取脚本接管，这里不动相机；
/// - 当没有物体被抓住时：
///     * 单手 pinch -> 用这只手在屏幕上的移动来绕 target 旋转视角；
///     * 双手 pinch -> 用两只手之间的距离变化来缩放（拉近/拉远）视角。
/// </summary>
public class HandOrbitCamera : MonoBehaviour
{
    [Header("引用")]
    public HandFromVectors handTracker;       // 场景里挂 HandFromVectors 的物体
    public Camera orbitCamera;                // 一般就是 Main Camera
    public Transform target;                  // 围绕看的对象（模型 / 空物体）

    [Header("单手旋转设置")]
    [Tooltip("用哪个关节当控制点，默认 8 = 食指指尖")]
    public int controlJointIndex = 8;

    [Tooltip("单手横向移动 1.0（整个屏幕宽度）对应的水平旋转角度")]
    public float horizontalDegrees = 200f;

    [Tooltip("单手纵向移动 1.0（整个屏幕高度）对应的垂直旋转角度")]
    public float verticalDegrees = 150f;

    [Tooltip("屏幕归一化坐标的最小移动阈值（防抖）")]
    public float moveDeadZone = 0.002f;

    [Header("双手缩放设置")]
    public bool enableTwoHandZoom = true;

    [Tooltip("两只手的屏幕距离变化 1.0（从完全重合到两端）对应的半径变化")]
    public float twoHandZoomSensitivity = 3.0f;

    [Tooltip("两只手距离变化小于这个值就忽略，避免抖动")]
    public float zoomDeadZone = 0.001f;

    [Tooltip("相机距离 target 的最小半径")]
    public float minRadius = 0.3f;

    [Tooltip("相机距离 target 的最大半径")]
    public float maxRadius = 5.0f;

    [Header("与抓取脚本协同")]
    [Tooltip("为 true 时，只要有任意 PinchGrabBall 被抓住，就完全停止相机旋转/缩放")]
    public bool disableWhenGrabbing = true;

    // 内部状态：单手旋转
    private bool isOrbiting = false;
    private int orbitHandIndex = -1;
    private Vector3 lastOrbitViewportPos;

    // 内部状态：双手缩放
    private bool isZooming = false;
    private int zoomHandA = -1;
    private int zoomHandB = -1;
    private float lastHandsViewportDist;

    // 相机当前半径（距离 target 的距离）
    private float currentRadius = 1.0f;

    void Start()
    {
        if (orbitCamera == null)
            orbitCamera = Camera.main;

        if (orbitCamera != null && target != null)
        {
            currentRadius = Vector3.Distance(orbitCamera.transform.position, target.position);
        }

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

        // 如果有物体被抓住，就把相机控制权全部让给 PinchGrabBall
        if (disableWhenGrabbing && PinchGrabBall.AnyObjectGrabbed)
        {
            ResetStates();
            return;
        }

        // 收集当前所有“正在 pinch 且在相机前方”的手
        int maxHands = handTracker.MaxHandCount;

        const int maxCandidates = 4; // 实际上你一般只有两只手，这里留点余量
        int[] handIndices = new int[maxCandidates];
        Vector3[] worldPositions = new Vector3[maxCandidates];
        Vector3[] viewportPositions = new Vector3[maxCandidates];
        int pinchCount = 0;

        for (int h = 0; h < maxHands; h++)
        {
            if (!handTracker.IsHandPinching(h))
                continue;

            if (!handTracker.TryGetJointPosition(h, controlJointIndex, out Vector3 jointWorld))
                continue;

            Vector3 vp = orbitCamera.WorldToViewportPoint(jointWorld);
            if (vp.z <= 0f)
                continue; // 在相机后面的忽略

            if (pinchCount < maxCandidates)
            {
                handIndices[pinchCount] = h;
                worldPositions[pinchCount] = jointWorld;
                viewportPositions[pinchCount] = vp;
                pinchCount++;
            }
        }

        if (pinchCount == 0)
        {
            ResetStates();
            return;
        }

        if (pinchCount == 1 || !enableTwoHandZoom)
        {
            // 只有一只手在 pinch（或者关闭双手缩放） -> 单手旋转模式
            int h = handIndices[0];
            Vector3 vp = viewportPositions[0];
            UpdateSingleHandOrbit(h, vp);
            // 进入单手模式时，停止双手缩放状态
            isZooming = false;
            zoomHandA = zoomHandB = -1;
        }
        else
        {
            // 有两只及以上手在 pinch -> 使用前两只做双手缩放
            int hA = handIndices[0];
            int hB = handIndices[1];
            Vector3 vpA = viewportPositions[0];
            Vector3 vpB = viewportPositions[1];

            UpdateTwoHandZoom(hA, hB, vpA, vpB);

            // 进入双手模式时，停止单手旋转状态
            isOrbiting = false;
            orbitHandIndex = -1;
        }
    }

    private void ResetStates()
    {
        isOrbiting = false;
        orbitHandIndex = -1;
        isZooming = false;
        zoomHandA = -1;
        zoomHandB = -1;
    }

    /// <summary>
    /// 单手旋转：根据这只手在屏幕上的移动，绕 target 做 yaw/pitch 旋转
    /// </summary>
    private void UpdateSingleHandOrbit(int handIndex, Vector3 viewportPos)
    {
        if (!isOrbiting || handIndex != orbitHandIndex)
        {
            // 新开始用这只手旋转：记录起点，不立即转
            isOrbiting = true;
            orbitHandIndex = handIndex;
            lastOrbitViewportPos = viewportPos;

            // 同步当前半径
            if (orbitCamera != null && target != null)
            {
                currentRadius = Vector3.Distance(orbitCamera.transform.position, target.position);
            }
            return;
        }

        Vector3 delta = viewportPos - lastOrbitViewportPos;
        lastOrbitViewportPos = viewportPos;

        if (delta.sqrMagnitude < moveDeadZone * moveDeadZone)
            return;

        float yawDeg = delta.x * horizontalDegrees;   // 左右移动 -> 绕世界 Y 轴
        float pitchDeg = -delta.y * verticalDegrees;  // 上下移动 -> 抬头/低头（反向）

        OrbitAroundTarget(yawDeg, pitchDeg, 0f);
    }

    /// <summary>
    /// 双手缩放：根据两只手的屏幕距离变化来调整相机和 target 之间的半径
    /// </summary>
    private void UpdateTwoHandZoom(int handA, int handB, Vector3 viewportA, Vector3 viewportB)
    {
        Vector2 pA = new Vector2(viewportA.x, viewportA.y);
        Vector2 pB = new Vector2(viewportB.x, viewportB.y);
        float dist = Vector2.Distance(pA, pB);

        if (!isZooming || handA != zoomHandA || handB != zoomHandB)
        {
            // 刚进入双手模式：记录当前距离，不立即缩放
            isZooming = true;
            zoomHandA = handA;
            zoomHandB = handB;
            lastHandsViewportDist = dist;

            if (orbitCamera != null && target != null)
            {
                currentRadius = Vector3.Distance(orbitCamera.transform.position, target.position);
            }
            return;
        }

        float delta = dist - lastHandsViewportDist;
        lastHandsViewportDist = dist;

        if (Mathf.Abs(delta) < zoomDeadZone)
            return;

        // 距离变大（两手分开） -> 视角拉近（半径减小）
        float radiusDelta = -delta * twoHandZoomSensitivity;

        OrbitAroundTarget(0f, 0f, radiusDelta);
    }

    /// <summary>
    /// 实际执行绕 target 的旋转 + 半径调整
    /// </summary>
    private void OrbitAroundTarget(float yawDeg, float pitchDeg, float radiusDelta)
    {
        if (orbitCamera == null || target == null)
            return;

        Vector3 pivot = target.position;
        Vector3 dir = orbitCamera.transform.position - pivot;
        float radius = dir.magnitude;

        if (radius < 1e-4f)
        {
            radius = currentRadius > 0f ? currentRadius : 1.0f;
            if (dir.sqrMagnitude < 1e-6f)
            {
                // 如果相机就在 target 上，随便给一个方向
                dir = -orbitCamera.transform.forward;
            }
        }

        // 先根据 yaw/pitch 旋转方向向量
        if (Mathf.Abs(yawDeg) > Mathf.Epsilon)
        {
            Quaternion yawRot = Quaternion.AngleAxis(yawDeg, Vector3.up);
            dir = yawRot * dir;
        }

        if (Mathf.Abs(pitchDeg) > Mathf.Epsilon)
        {
            Quaternion pitchRot = Quaternion.AngleAxis(pitchDeg, orbitCamera.transform.right);
            dir = pitchRot * dir;
        }

        // 再根据 radiusDelta 调整半径
        radius = Mathf.Clamp(radius + radiusDelta, minRadius, maxRadius);
        currentRadius = radius;

        orbitCamera.transform.position = pivot + dir.normalized * radius;
        orbitCamera.transform.LookAt(pivot, Vector3.up);
    }
}
