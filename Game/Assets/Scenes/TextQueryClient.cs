using System;
using System.Net.Sockets;
using System.Text;
using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class TextQueryClient_TMP : MonoBehaviour
{
    [Header("Server")]
    public string serverIp = "127.0.0.1";
    public int serverPort = 9009;

    [Header("UI Elements (TMP)")]
    public GameObject dialogPanel;        
    public TMP_InputField descriptionInput;
    public Button openDialogButton;       
    public Button sendButton;             
    public TMP_Text resultText;
    public GameObject successPanel;       

    private TcpClient client;
    private NetworkStream stream;
    private bool connected = false;

    void Start()
    {
        if (dialogPanel != null) dialogPanel.SetActive(false);
        if (successPanel != null) successPanel.SetActive(false);

        StartCoroutine(ConnectToServer());

        if (openDialogButton != null)
            openDialogButton.onClick.AddListener(OpenDialog);

        if (sendButton != null)
            sendButton.onClick.AddListener(OnClickSendDescription);
    }

    private IEnumerator ConnectToServer()
    {
        // 先等一帧，保证场景初始化完
        yield return null;

        try
        {
            client = new TcpClient();
            // 同步连接，局域网 localhost 基本一瞬间完成
            client.Connect(serverIp, serverPort);
            stream = client.GetStream();
            connected = true;
            Debug.Log("Connected to text server.");
        }
        catch (Exception e)
        {
            Debug.LogWarning("Could not connect to server: " + e.Message);
            connected = false;
        }
    }


    public void OpenDialog()
    {
        Debug.Log("OpenDialog() 被调用了！！！");

        if (dialogPanel == null)
        {
            Debug.LogWarning("dialogPanel 未绑定！");
            return;
        }

        dialogPanel.SetActive(true);

        if (descriptionInput != null)
        {
            descriptionInput.text = "";
            descriptionInput.ActivateInputField();
            descriptionInput.Select();
        }
    }



    public void CloseDialog()
    {
        if (dialogPanel != null)
            dialogPanel.SetActive(false);
    }

    public void OnClickSendDescription()
    {
        if (descriptionInput == null)
        {
            Debug.LogWarning("descriptionInput 未绑定");
            return;
        }

        string q = descriptionInput.text;
        StartCoroutine(SendQueryCoroutine(q));
    }

    private IEnumerator SendQueryCoroutine(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
        {
            Debug.Log("Empty query, skip.");
            yield break;
        }

        if (!connected || stream == null)
        {
            Debug.LogWarning("Not connected to server.");
            yield break;
        }

        byte[] data = Encoding.UTF8.GetBytes(query + "\n");
        try
        {
            stream.Write(data, 0, data.Length);
            stream.Flush();
            Debug.Log("Sent query: " + query);
        }
        catch (Exception e)
        {
            Debug.LogError("Write error: " + e.Message);
            yield break;
        }

        float start = Time.time;
        while (!stream.DataAvailable && Time.time - start < 5.0f)
            yield return null;

        if (!stream.DataAvailable)
        {
            Debug.LogWarning("No response from server.");
            yield break;
        }

        byte[] buffer = new byte[512];
        int n = 0;
        try
        {
            n = stream.Read(buffer, 0, buffer.Length);
        }
        catch (Exception e)
        {
            Debug.LogError("Read error: " + e.Message);
            yield break;
        }

        if (n <= 0)
        {
            Debug.LogWarning("Zero bytes received.");
            yield break;
        }

        string resp = Encoding.UTF8.GetString(buffer, 0, n).Trim();
        Debug.Log("Server resp: " + resp);

        if (resultText != null)
            resultText.text = $"预测模型: {resp}";

        if (successPanel != null)
        {
            successPanel.SetActive(true);
            StartCoroutine(HideAfter(successPanel, 1.0f));
        }

        CloseDialog();
    }

    private IEnumerator HideAfter(GameObject go, float seconds)
    {
        yield return new WaitForSeconds(seconds);
        if (go != null) go.SetActive(false);
    }

    void OnDestroy()
    {
        try
        {
            if (stream != null) stream.Close();
            if (client != null) client.Close();
        }
        catch { }
    }
}
