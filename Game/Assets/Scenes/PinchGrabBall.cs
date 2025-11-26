using UnityEngine;

/// <summary>
/// 挂在“被捏的球”上：
/// - pinch 状态来自 HandFromVectors（由 Python 计算并传入的 hand.pinch）
/// - 不在这里重新算 pinch，只用一个距离阈值判断“手指是否碰到球”
/// - 当某只手 pinch 且食指尖靠近球时：抓住球；
///   pinch 继续 -> 球跟着这只手的食指走；
///   pinch 松开 -> 球放手（可选掉下去）。
/// </summary>
public class PinchGrabBall : MonoBehaviour
{
    [Header("Hand Source")]
    public HandFromVectors handTracker;   // 场景里挂 HandFromVectors 的物体，拖进来

    [Header("抓取设置")]
    public int controlJointIndex = 8;     // 控制球的关节，默认食指指尖 8
    public float grabDistance = 0.10f;    // 食指尖到球中心小于这个距离就认为“碰到球”(米)

    [Header("物理设置")]
    public bool usePhysics = false;       // 想松手后掉下去就勾上 + 球上挂 Rigidbody

    [Header("跟随设置")]
    public int followJointIndex = 0;      // 用来“跟随移动”的关节，0 = 掌根 / 手腕

    [Range(0f, 1f)]
    public float followSmoothing = 0.15f;   // 0 = 立刻跟到位, 0.1~0.3 = 有一点粘滞感

    [Header("Pinch 容错")]
    public float pinchReleaseGrace = 0.3f;   // 松手前允许掉线多久（秒），可以在 Inspector 里调
    private float pinchOffTimer = 0f;         // 已经连续“非 pinch”的累计时间


    [Header("Debug")]
    public bool isGrabbed = false;
    public int grabbedHandIndex = -1;

    private Rigidbody rb;
    private Vector3 grabOffset;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        if (handTracker == null)
            return;

        if (!isGrabbed)
        {
            // 还没有被抓：在所有手里找一个 “正在 pinch 且 食指靠近球” 的手
            float bestDist = float.MaxValue;
            int bestHand = -1;
            Vector3 bestJointPos = Vector3.zero;

            int maxHands = handTracker.MaxHandCount;
            for (int h = 0; h < maxHands; h++)
            {
                // pinch 状态完全从 Python/HandFromVectors 来
                if (!handTracker.IsHandPinching(h))
                    continue;

                // 拿这只手控制关节（默认 8 = 食指尖）的世界坐标
                if (!handTracker.TryGetJointPosition(h, controlJointIndex, out Vector3 jointPos))
                    continue;

                float dist = Vector3.Distance(transform.position, jointPos);
                if (dist < grabDistance && dist < bestDist)
                {
                    bestDist = dist;
                    bestHand = h;
                    bestJointPos = jointPos;
                }
            }

            // 找到合适的手 -> 开始抓球
            // 找到合适的手 -> 开始抓球
            if (bestHand != -1)
            {
                isGrabbed = true;
                grabbedHandIndex = bestHand;
                pinchOffTimer = 0f;   // ✅ 一旦抓住，清空“松手计时器”

                // 用“跟随关节”的位置来算偏移，优先用掌根（followJointIndex）
                Vector3 refPos = bestJointPos; // 兜底：万一掌根没拿到，就用当前那根手指
                if (handTracker.TryGetJointPosition(bestHand, followJointIndex, out Vector3 followPos))
                {
                    refPos = followPos;
                }
                grabOffset = transform.position - refPos;

                if (usePhysics && rb != null)
                {
                    rb.useGravity = false;
                    rb.velocity = Vector3.zero;
                    rb.angularVelocity = Vector3.zero;
                }
            }

        }
        else
        {
            // 已经被某只手抓着

            if (handTracker.IsHandPinching(grabbedHandIndex))
            {
                // 这一帧还是 pinch，说明手还在捏 -> 清零掉线计时器
                pinchOffTimer = 0f;
            }
            else
            {
                // 这一帧检测到不是 pinch 了 -> 累加“掉线时间”
                pinchOffTimer += Time.deltaTime;

                if (pinchOffTimer >= pinchReleaseGrace)
                {
                    // 掉线太久了，才真的认为“松手了”
                    isGrabbed = false;
                    if (usePhysics && rb != null)
                        rb.useGravity = true;
                    grabbedHandIndex = -1;
                    return; // 很关键：不要再往下走跟随逻辑
                }

                // 如果还没超过 pinchReleaseGrace，就当作“手还勉强抓着”，继续跟随
            }

            // 只要没被真正判定为松手，就继续跟随（你现在是跟掌根还是跟食指，看你上次怎么改的）
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
                    // 每帧往目标靠一点点
                    transform.position = Vector3.Lerp(transform.position, targetPos, followSmoothing);
                }
            }

            
            else
            {
                // 跟随关节本身丢失了（整只手消失） -> 直接松手
                isGrabbed = false;
                if (usePhysics && rb != null)
                    rb.useGravity = true;
                grabbedHandIndex = -1;
            }
        }

    }
}
