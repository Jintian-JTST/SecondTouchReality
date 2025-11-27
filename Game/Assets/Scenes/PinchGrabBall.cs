using UnityEngine;

/// <summary>
/// 挂在“可被 pinch 抓取的物体”上：
/// - pinch 布尔量来自 HandFromVectors（由 Python 计算 hand.pinch）；
/// - 不在这里重新算 pinch，只用一个距离阈值判断“手指是否靠近物体”；
/// - 当某只手 pinch 且食指尖靠近物体时：抓住物体；
///   pinch 持续 -> 物体跟随这只手；
///   pinch 短暂中断 -> 给一个“宽限时间”，避免抖动导致立刻松手；
///   pinch 真正松开或手丢失 -> 物体松手（可选掉落）。
/// </summary>
public class PinchGrabBall : MonoBehaviour
{
    [Header("Hand Source")]
    public HandFromVectors handTracker;   // 场景里挂 HandFromVectors 的物体，拖进来

    [Header("抓取设置")]
    public int controlJointIndex = 8;     // 用哪个关节来“碰撞检测”，默认 8 = 食指指尖
    public float grabDistance = 0.10f;    // 食指尖到物体中心小于这个距离就认为“碰到物体”(米)

    [Header("物理设置")]
    public bool usePhysics = false;       // 想松手后掉下去就勾上 + 物体上挂 Rigidbody

    [Header("跟随设置")]
    public int followJointIndex = 0;      // 用哪个关节来“跟随移动”，0 = 掌根/手腕

    [Range(0f, 1f)]
    public float followSmoothing = 0.15f; // 0 = 立刻跟到位, 0.1~0.3 = 有一点粘滞感

    [Header("Pinch 容错")]
    public float pinchReleaseGrace = 0.3f;   // 松手前允许“短暂非 pinch”的宽限时间（秒）
    private float pinchOffTimer = 0f;        // 已经连续“非 pinch”的累计时间

    [Header("调试")]
    public bool isGrabbed = false;           // 当前是否被某只手抓住
    public int grabbedHandIndex = -1;        // 正在抓住它的那只手的 handIndex（-1 = 没有）

    private Rigidbody rb;
    private Vector3 grabOffset;              // 物体中心相对于跟随关节的偏移

    // ===== 全局静态：让别的脚本知道“场景里是否有东西被抓着” =====
    private static int grabbedCount = 0;     // 当前被抓住的 PinchGrabBall 个数
    public static bool AnyObjectGrabbed     // 只要 grabbedCount > 0，就算“有东西被抓着”
    {
        get { return grabbedCount > 0; }
    }

    // 为了防止重复 +1 / -1，这里给每个实例记一下自己有没有登记过
    private bool registeredGrab = false;

    private void RegisterGrab()
    {
        if (!registeredGrab)
        {
            grabbedCount++;
            registeredGrab = true;
        }
    }

    private void UnregisterGrab()
    {
        if (registeredGrab)
        {
            grabbedCount = Mathf.Max(0, grabbedCount - 1);
            registeredGrab = false;
        }
    }

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void OnDisable()
    {
        // 避免物体被禁用/销毁时静态 grabbedCount 没有同步
        if (isGrabbed)
        {
            isGrabbed = false;
            grabbedHandIndex = -1;
            UnregisterGrab();
        }
    }

    void Update()
    {
        if (handTracker == null)
            return;

        if (!isGrabbed)
        {
            // 还没有被抓：在所有手里找一个“正在 pinch 且 食指靠近物体”的手
            float bestDist = float.MaxValue;
            int bestHand = -1;
            Vector3 bestJointPos = Vector3.zero;

            int maxHands = handTracker.MaxHandCount;
            for (int h = 0; h < maxHands; h++)
            {
                if (!handTracker.IsHandPinching(h))
                    continue;

                // 拿这只手控制关节（默认 8 = 食指尖）的世界坐标
                if (!handTracker.TryGetJointPosition(h, controlJointIndex, out Vector3 jointWorld))
                    continue;

                float dist = Vector3.Distance(transform.position, jointWorld);
                if (dist < grabDistance && dist < bestDist)
                {
                    bestDist = dist;
                    bestHand = h;
                    bestJointPos = jointWorld;
                }
            }

            // 找到合适的手 -> 开始抓物体
            if (bestHand != -1)
            {
                isGrabbed = true;
                grabbedHandIndex = bestHand;
                pinchOffTimer = 0f;

                // 决定跟随哪个关节（可以用掌根，也可以继续用食指）
                if (!handTracker.TryGetJointPosition(bestHand, followJointIndex, out Vector3 followPos))
                {
                    // 如果 followJointIndex 不合法，就退回到控制关节
                    followPos = bestJointPos;
                }

                grabOffset = transform.position - followPos;

                if (usePhysics && rb != null)
                {
                    rb.useGravity = false;
                    rb.velocity = Vector3.zero;
                    rb.angularVelocity = Vector3.zero;
                }

                RegisterGrab();
            }
        }
        else
        {
            // 已经被某只手抓住
            if (grabbedHandIndex < 0 || grabbedHandIndex >= handTracker.MaxHandCount)
            {
                // 索引异常，直接松手
                ReleaseObject();
                return;
            }

            // 1) 处理 pinch 宽限时间
            if (handTracker.IsHandPinching(grabbedHandIndex))
            {
                // 仍然是 pinch，计时器清零
                pinchOffTimer = 0f;
            }
            else
            {
                // 这一帧不是 pinch，累积非 pinch 时间
                pinchOffTimer += Time.deltaTime;
                if (pinchOffTimer >= pinchReleaseGrace)
                {
                    // 超过宽限时间，认为“真的松手了”
                    ReleaseObject();
                    return;
                }
            }

            // 2) 继续跟随那只手
            if (handTracker.TryGetJointPosition(grabbedHandIndex, followJointIndex, out Vector3 followPos))
            {
                Vector3 targetPos = followPos + grabOffset;

                if (followSmoothing <= 0f)
                {
                    // 不要平滑，直接跟
                    transform.position = targetPos;
                }
                else
                {
                    // 每帧往目标靠一点
                    transform.position = Vector3.Lerp(transform.position, targetPos, followSmoothing);
                }
            }
            else
            {
                // 这一帧拿不到跟随关节（比如整只手掉线了） -> 直接松手
                ReleaseObject();
            }
        }
    }

    private void ReleaseObject()
    {
        if (!isGrabbed)
            return;

        isGrabbed = false;
        grabbedHandIndex = -1;
        pinchOffTimer = 0f;

        if (usePhysics && rb != null)
        {
            rb.useGravity = true;
        }

        UnregisterGrab();
    }
}
