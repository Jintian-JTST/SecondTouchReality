using UnityEngine;
public class HandOrbitCamera : MonoBehaviour
{
    [Header("References")]
    public HandFromVectors handTracker;    
    public Camera orbitCamera;           
    public Transform target;               

    [Header("Single Hand Orbit Settings")]
    [Tooltip("Use which joint as the control point, default 8 = Index Finger Tip")]
    public int controlJointIndex = 8;

    [Tooltip("Single hand horizontal movement 1.0 (entire screen width) corresponds to horizontal rotation angle")]
    public float horizontalDegrees = 200f;

    [Tooltip("Single hand vertical movement 1.0 (entire screen height) corresponds to vertical rotation angle")]
    public float verticalDegrees = 150f;

    [Tooltip("Minimum movement threshold for screen normalized coordinates (anti-shake)")]
    public float moveDeadZone = 0.002f;

    [Header("Two Hand Zoom Settings")]
    public bool enableTwoHandZoom = true;

    [Tooltip("Two hands' screen distance change 1.0 (from completely overlapping to both ends) corresponds to radius change")]
    public float twoHandZoomSensitivity = 3.0f;

    [Tooltip("Ignore changes in distance between two hands smaller than this value to avoid jitter")]
    public float zoomDeadZone = 0.001f;

    [Tooltip("Minimum radius from the camera to the target")]
    public float minRadius = 0.3f;

    [Tooltip("Maximum radius from the camera to the target")]
    public float maxRadius = 5.0f;

    [Header("Sample Settings")]
    [Tooltip("When true, camera rotation/zoom completely stops as long as any PinchGrabBall is grabbed")]
    public bool disableWhenGrabbing = true;

    private bool isOrbiting = false;
    private int orbitHandIndex = -1;
    private Vector3 lastOrbitViewportPos;

    private bool isZooming = false;
    private int zoomHandA = -1;
    private int zoomHandB = -1;
    private float lastHandsViewportDist;

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
            Debug.LogWarning("HandOrbitCamera: handTracker is not set.");
        }

        if (target == null)
        {
            Debug.LogWarning("HandOrbitCamera: target is not set, the view will not orbit around anything.");
        }
    }

    void Update()
    {
        if (handTracker == null || orbitCamera == null || target == null)
            return;

        if (disableWhenGrabbing && PinchGrabBall.AnyObjectGrabbed)
        {
            ResetStates();
            return;
        }

        int maxHands = handTracker.MaxHandCount;

        const int maxCandidates = 4;
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
                continue;

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
            int h = handIndices[0];
            Vector3 vp = viewportPositions[0];
            UpdateSingleHandOrbit(h, vp);
            isZooming = false;
            zoomHandA = zoomHandB = -1;
        }
        else
        {
            int hA = handIndices[0];
            int hB = handIndices[1];
            Vector3 vpA = viewportPositions[0];
            Vector3 vpB = viewportPositions[1];

            UpdateTwoHandZoom(hA, hB, vpA, vpB);

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


    private void UpdateSingleHandOrbit(int handIndex, Vector3 viewportPos)
    {
        if (!isOrbiting || handIndex != orbitHandIndex)
        {
            isOrbiting = true;
            orbitHandIndex = handIndex;
            lastOrbitViewportPos = viewportPos;

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

        float yawDeg = delta.x * horizontalDegrees; 
        float pitchDeg = -delta.y * verticalDegrees;

        OrbitAroundTarget(yawDeg, pitchDeg, 0f);
    }


    private void UpdateTwoHandZoom(int handA, int handB, Vector3 viewportA, Vector3 viewportB)
    {
        Vector2 pA = new Vector2(viewportA.x, viewportA.y);
        Vector2 pB = new Vector2(viewportB.x, viewportB.y);
        float dist = Vector2.Distance(pA, pB);

        if (!isZooming || handA != zoomHandA || handB != zoomHandB)
        {
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

        float radiusDelta = -delta * twoHandZoomSensitivity;

        OrbitAroundTarget(0f, 0f, radiusDelta);
    }


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
                dir = -orbitCamera.transform.forward;
            }
        }

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

        radius = Mathf.Clamp(radius + radiusDelta, minRadius, maxRadius);
        currentRadius = radius;

        orbitCamera.transform.position = pivot + dir.normalized * radius;
        orbitCamera.transform.LookAt(pivot, Vector3.up);
    }
}
