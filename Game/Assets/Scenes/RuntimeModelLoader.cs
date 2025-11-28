using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using GLTFast;

public class RuntimeModelLoader : MonoBehaviour
{
    // 当前已经实例化的模型（方便切换时销毁旧的）
    private GameObject currentInstance;

    // 如果你的文件名不是纯数字，可以在这里改成字典：
    // index -> fileName
    private string MakeFileName(int index)
    {
        // 最简单：101 → "101.glb"
        return index.ToString() + ".glb";
        // 如果你是 "model_101.glb"，改成：return $"model_{index}.glb";
    }

    public async Task<GameObject> LoadByIndexAsync(int index)
    {
        string fileName = MakeFileName(index);
        string path = Path.Combine(Application.streamingAssetsPath, "models", fileName);

        if (!File.Exists(path))
        {
            Debug.LogError($"[RuntimeModelLoader] File not found: {path}");
            return null;
        }

        // 清掉上一只模型
        if (currentInstance != null)
        {
            Destroy(currentInstance);
            currentInstance = null;
        }

        var importer = new GltfImport();
        // 有些版本支持直接本地路径，有些需要 file:// URI，这样写最稳
        string uri = new System.Uri(path).AbsoluteUri;

        bool ok = await importer.Load(uri);
        if (!ok)
        {
            Debug.LogError("[RuntimeModelLoader] Failed to load glb: " + uri);
            return null;
        }

        // 创建一个空对象挂在 MODEL LIBRARY 下
        currentInstance = new GameObject($"Model_{index}");
        importer.InstantiateMainScene(currentInstance.transform);

        // 把它对齐到 loader 所在物体下
        currentInstance.transform.SetParent(this.transform, false);
        currentInstance.transform.localPosition = Vector3.zero;
        currentInstance.transform.localRotation = Quaternion.identity;

        return currentInstance;
    }

    // 提供一个方便给别的脚本调用的 API（不关心 await）
    public void LoadByIndex(int index)
    {
        _ = LoadByIndexAsync(index);
    }
}
