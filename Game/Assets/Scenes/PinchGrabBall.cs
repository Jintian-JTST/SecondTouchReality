using UnityEngine;

public class PinchGrabBall : MonoBehaviour
{
    [Header("Hand Source")]
    public HandFromVectors handTracker; 
    [Header("抓取设置")]
    public int controlJointIndex = 8;   
    public float grabDistance = 0.10f;  
    [Header("物理设置")]
    public bool usePhysics = false;      

    [Header("跟随设置")]
    public int followJointIndex = 0;     

    [Range(0f, 1f)]
    public float followSmoothing = 0.15f; 

    [Header("Pinch 容错")]
    public float pinchReleaseGrace = 0.3f;  
    private float pinchOffTimer = 0f;       

    [Header("调试")]
    public bool isGrabbed = false;           
    public int grabbedHandIndex = -1;       

    private Rigidbody rb;
    private Vector3 grabOffset;            

    private static int grabbedCount = 0;    
    public static bool AnyObjectGrabbed    
        get { return grabbedCount > 0; }
    }

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
            float bestDist = float.MaxValue;
            int bestHand = -1;
            Vector3 bestJointPos = Vector3.zero;

            int maxHands = handTracker.MaxHandCount;
            for (int h = 0; h < maxHands; h++)
            {
                if (!handTracker.IsHandPinching(h))
                    continue;

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

            if (bestHand != -1)
            {
                isGrabbed = true;
                grabbedHandIndex = bestHand;
                pinchOffTimer = 0f;

                if (!handTracker.TryGetJointPosition(bestHand, followJointIndex, out Vector3 followPos))
                {
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
            if (grabbedHandIndex < 0 || grabbedHandIndex >= handTracker.MaxHandCount)
            {
                ReleaseObject();
                return;
            }

            if (handTracker.IsHandPinching(grabbedHandIndex))
            {
                pinchOffTimer = 0f;
            }
            else
            {
                pinchOffTimer += Time.deltaTime;
                if (pinchOffTimer >= pinchReleaseGrace)
                {
                    ReleaseObject();
                    return;
                }
            }

            if (handTracker.TryGetJointPosition(grabbedHandIndex, followJointIndex, out Vector3 followPos))
            {
                Vector3 targetPos = followPos + grabOffset;

                if (followSmoothing <= 0f)
                {
                    transform.position = targetPos;
                }
                else
                {
                    transform.position = Vector3.Lerp(transform.position, targetPos, followSmoothing);
                }
            }
            else
            {
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
