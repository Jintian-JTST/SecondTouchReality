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
            if (bestHand != -1)
            {
                isGrabbed = true;
                grabbedHandIndex = bestHand;
                grabOffset = transform.position - bestJointPos;

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
            if (!handTracker.IsHandPinching(grabbedHandIndex))
            {
                // 这只手不再 pinch -> 松手
                isGrabbed = false;
                if (usePhysics && rb != null)
                    rb.useGravity = true;
                grabbedHandIndex = -1;
                return;
            }

            // 手还在 pinch，就跟着那只手的控制关节走
            if (handTracker.TryGetJointPosition(grabbedHandIndex, controlJointIndex, out Vector3 jointPos2))
            {
                transform.position = jointPos2 + grabOffset;
            }
            else
            {
                // 这一帧那只手丢失了 -> 松手
                isGrabbed = false;
                if (usePhysics && rb != null)
                    rb.useGravity = true;
                grabbedHandIndex = -1;
            }
        }
    }
}
