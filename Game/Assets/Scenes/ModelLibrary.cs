using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 把场景里挂在自己下面的所有子物体，当成一个“模型字典”。
/// Python 返回一个字符串 label（比如 "023"），就调用 ShowModelByLabel("023")，
/// 把对应名字的子物体激活、摆到指定位置，并自动挂上 PinchGrabBall，
/// 这样就能被你的手 Pinch 抓起来了。
/// </summary>
public class ModelLibrary : MonoBehaviour
{
    [Header("生成位置设置")]
    public Transform spawnAnchor;             // 一般拖主摄像机或者某个空物体
    public Vector3 spawnOffset = new Vector3(0f, 0f, 0.4f); // 相对于 anchor 的偏移（米）

    [Header("手部跟踪（用于 Pinch 抓取）")]
    public HandFromVectors handTracker;       // 场景里那个有 HandFromVectors 的物体
    public float grabDistance = 0.10f;        // 手指尖到物体中心小于这个距离就算碰到了
    public bool usePhysics = false;           // 如果模型上有 Rigidbody 且想松手掉下去就勾上

    private readonly Dictionary<string, GameObject> _models = new Dictionary<string, GameObject>();
    private GameObject _lastModel;            // 方便“只显示一个模型”的情况

    private void Awake()
    {
        _models.Clear();

        // 把所有子物体按名字收进字典，并全部先隐藏
        foreach (Transform child in transform)
        {
            var go = child.gameObject;
            if (!_models.ContainsKey(go.name))
            {
                _models.Add(go.name, go);
            }
            go.SetActive(false);
        }
    }

    /// <summary>
    /// 根据 label（比如 "023"）显示一个模型。
    /// </summary>
    public GameObject ShowModelByLabel(string label)
    {
        if (string.IsNullOrWhiteSpace(label))
        {
            Debug.LogWarning("ModelLibrary.ShowModelByLabel: label 为空");
            return null;
        }

        label = label.Trim();

        if (!_models.TryGetValue(label, out var go))
        {
            Debug.LogWarning($"ModelLibrary: 找不到名为 '{label}' 的子对象，" +
                             $"请检查该模型名字是否和 Python 返回的一致。");
            return null;
        }

        // 如果你想一次只显示一个模型，可以把上一个关掉
        if (_lastModel != null && _lastModel != go)
        {
            _lastModel.SetActive(false);
        }

        go.SetActive(true);

        // ===== 计算生成位置 =====
        Vector3 basePos = Vector3.zero;
        Quaternion baseRot = Quaternion.identity;

        if (spawnAnchor != null)
        {
            basePos = spawnAnchor.position;
            baseRot = spawnAnchor.rotation;
        }
        else if (Camera.main != null)
        {
            basePos = Camera.main.transform.position;
            baseRot = Camera.main.transform.rotation;
        }

        go.transform.position = basePos + baseRot * spawnOffset;
        // 如果你想让模型朝向和摄像机一致，可以打开这一行：
        // go.transform.rotation = baseRot;

        _lastModel = go;

        // ===== 自动配置 Pinch 抓取 =====
        if (handTracker != null)
        {
            var grab = go.GetComponent<PinchGrabBall>();
            if (grab == null)
            {
                grab = go.AddComponent<PinchGrabBall>();
            }

            grab.handTracker = handTracker;
            grab.grabDistance = grabDistance;
            grab.usePhysics = usePhysics;
            // 其他参数（controlJointIndex 等）用 PinchGrabBall 里的默认就行
        }

        return go;
    }
}
