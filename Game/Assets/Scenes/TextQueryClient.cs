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

    [Header("Model Library")]
    public ModelLibrary modelLibrary; 

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
        yield return null;

        try
        {
            client = new TcpClient();
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
        Debug.Log("OpenDialog() called.");

        if (dialogPanel == null)
        {
            Debug.LogWarning("dialogPanel is not bound!");
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
            Debug.LogWarning("descriptionInput Not Bound!");
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

        if (modelLibrary != null)
        {
            var go = modelLibrary.ShowModelByLabel(resp);
            if (go == null)
            {
                Debug.LogWarning($"TextQueryClient_TMP: ModelLibrary did not find model named {resp}.");
            }
        }
        else
        {
            Debug.LogWarning("TextQueryClient_TMP: modelLibrary is not set, cannot instantiate model.");
        }
        // ===================================

        if (resultText != null)
            resultText.text = $"Predicted: {resp}";

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
