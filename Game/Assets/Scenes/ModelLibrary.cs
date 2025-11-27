using System.Collections.Generic;
using UnityEngine;


public class ModelLibrary : MonoBehaviour
{
    [Header("General Settings")]
    public Transform spawnAnchor;             
    public Vector3 spawnOffset = new Vector3(0f, 0f, 0.4f); 

    [Header("Grab Settings")]
    public HandFromVectors handTracker;      
    public float grabDistance = 0.10f;      
    public bool usePhysics = false;          

    private readonly Dictionary<string, GameObject> _models = new Dictionary<string, GameObject>();
    private GameObject _lastModel;            

    private void Awake()
    {
        _models.Clear();

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

    public GameObject ShowModelByLabel(string label)
    {
        if (string.IsNullOrWhiteSpace(label))
        {
            Debug.LogWarning("ModelLibrary.ShowModelByLabel: label is empty");
            return null;
        }

        label = label.Trim();

        if (!_models.TryGetValue(label, out var go))
        {
            Debug.LogWarning($"ModelLibrary: Can't find child object '{label}', " +
                             $"please check if the model name matches the one returned by Python.");
            return null;
        }

        if (_lastModel != null && _lastModel != go)
        {
            _lastModel.SetActive(false);
        }

        go.SetActive(true);

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

        _lastModel = go;

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
        }

        return go;
    }
}
