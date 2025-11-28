using System.IO.Ports;
using UnityEngine;

public class ArduinoServoController : MonoBehaviour
{
    public string portName = "COM5"; // 换成你实际的端口
    public int baudRate = 9600;

    private SerialPort sp;

    void Start()
    {
        sp = new SerialPort(portName, baudRate);
        sp.ReadTimeout = 50;
        try
        {
            sp.Open();
            Debug.Log("Serial opened: " + portName);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to open serial: " + e.Message);
        }
    }

    void OnDestroy()
    {
        if (sp != null && sp.IsOpen)
        {
            sp.Close();
        }
    }

    public void SendZero()
    {
        Send("0");
    }

    public void SendOne()
    {
        Send("1");
    }

    private void Send(string msg)
    {
        if (sp != null && sp.IsOpen)
        {
            sp.Write(msg);
            Debug.Log("Sent to Arduino: " + msg);
        }
    }
}
    