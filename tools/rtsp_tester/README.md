# CP Plus PoE IP Camera - Low-Latency RTSP Tester & Recorder

Standalone, 100% offline, zero-cloud desktop application and CLI tool designed to directly test, stream, snapshot, and record from a CP Plus PoE IP camera connected directly to your Windows laptop's Ethernet port (and modularly deployable to NVIDIA Jetson EdgeBox).

---

## 1. Physical Hardware Setup

```
[ CP Plus PoE IP Camera ]
           │
           │ Ethernet Cable (RJ45)
           ▼
[ PoE Injector (Data + Power Out) ]
           ▲
           │ AC Power Cord (230V/110V)
           │
[ PoE Injector (Data In / LAN Port) ]
           │
           │ Ethernet Cable (RJ45)
           ▼
[ Windows Laptop Ethernet / LAN Port ]
```

> **Note**: No Router, NVR, DVR, Switch, Wi-Fi, or Internet connection is required.

---

## 2. Laptop Ethernet Adapter IP Configuration (Direct Connection)

When an IP camera is connected directly to a laptop without a router (DHCP server), both devices must be on the **same static IPv4 subnet**.

### Default CP Plus IP Addresses:
- **Default IP**: `192.168.1.250` (or `192.168.0.250`)
- **Default Port**: `554` (RTSP), `80` (HTTP), `37777` (TCP)
- **Default Username**: `admin`
- **Default Password**: `admin` (or the password created during first-time camera activation)

### Set Laptop's Ethernet IP on Windows:
1. Press `Win + R`, type `ncpa.cpl` and press **Enter** (opens Network Connections).
2. Right-click **Ethernet** (or Local Area Connection) → **Properties**.
3. Select **Internet Protocol Version 4 (TCP/IPv4)** → Click **Properties**.
4. Choose **"Use the following IP address"**:
   - **IP address**: `192.168.1.100` *(Must be in same subnet `192.168.1.x` as camera, but not `.250`)*
   - **Subnet mask**: `255.255.255.0`
   - **Default gateway**: `192.168.1.1` *(or leave blank)*
5. Click **OK** → **OK**.

---

## 3. Verify Connection (Ping Test)

Open PowerShell / Command Prompt and test direct connectivity:
```powershell
ping 192.168.1.250
```
- If you receive `Reply from 192.168.1.250: bytes=32 time<1ms`, the camera is connected and ready.
- If it times out, try pinging `192.168.0.250` (and adjust your laptop IP to `192.168.0.100` if needed).

---

## 4. CP Plus Standard RTSP URL Formats

CP Plus (and Dahua OEM) IP cameras expose the following standard RTSP streaming endpoints:

| Stream Type | Description | RTSP URL Format |
|---|---|---|
| **Main Stream** (High-Res 1080p/4K) | Maximum image clarity | `rtsp://<username>:<password>@<IP>:554/cam/realmonitor?channel=1&subtype=0` |
| **Sub Stream** (Lower-Res 720p/D1) | **Ultra Low Latency** | `rtsp://<username>:<password>@<IP>:554/cam/realmonitor?channel=1&subtype=1` |

#### Example:
```
rtsp://admin:admin@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0
```

---

## 5. Software Installation (Windows)

### Step 1: Install Python Requirements
From your activated Python environment:
```powershell
cd d:\ai-video-based-inventory\ai-video-based-inventory\tools\rtsp_tester
pip install -r requirements.txt
```

---

## 6. How to Run the Application

### Option A: Modern Desktop GUI (Recommended)
Launch the dark-themed desktop dashboard:
```powershell
python camera_gui.py
```
1. Enter the **Camera IP** (e.g. `192.168.1.250`).
2. Enter your **Username** & **Password**.
3. Select **Main Stream (subtype=0)** or **Sub Stream (subtype=1)**.
4. Click **`▶ Connect`**.
5. Use **`📸 Snapshot`** to save full-resolution JPEG frames into `output/captures/`.
6. Use **`⏺ Start Recording`** to record real-time MP4 video into `output/recordings/`.

### Option B: Headless Command-Line Interface (Jetson / Server Testing)
Run non-blocking benchmark tests from terminal:
```powershell
# 30-second low-latency streaming benchmark:
python rtsp_cli.py --url "rtsp://admin:admin@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"

# Continuous test with automatic snapshot & recording:
python rtsp_cli.py --url "rtsp://admin:admin@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0" --snapshot --record --duration 60
```

---

## 7. GStreamer Installation & Configuration

The app defaults to **OpenCV FFmpeg with Low-Latency TCP flags**, which works out-of-the-box on Windows. 

If you wish to test with **GStreamer**:
1. Download **GStreamer 1.24+ MSVC binaries** from [gstreamer.freedesktop.org](https://gstreamer.freedesktop.org/download/).
2. Install both **Runtime** and **Development** packages.
3. Add `C:\gstreamer\1.0\msvc_x86_64\bin` to your Windows `PATH`.
4. In the GUI, switch **Backend Engine** to **`GStreamer Pipeline`**.

### For NVIDIA Jetson EdgeBox Deployment:
When migrating to NVIDIA Jetson, switch the backend to **`NVIDIA Jetson (HW-Dec)`**. It automatically invokes the hardware-accelerated pipeline:
```
rtspsrc location=rtsp://... latency=0 protocols=tcp ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink drop=1
```

---

## 8. Troubleshooting Guide

| Issue | Likely Cause | Solution |
|---|---|---|
| **Ping fails / Request timed out** | Subnet mismatch between laptop and camera | Ensure your laptop Ethernet IP is set to `192.168.1.100` and subnet is `255.255.255.0`. |
| **Authentication Failed (401 Unauthorized)** | Incorrect username or password | Open browser at `http://192.168.1.250` and verify login credentials. CP Plus cameras require password activation on first boot. |
| **Black screen / Connecting... loop** | Camera RTSP service disabled or port blocked | Ensure port `554` is not blocked by Windows Firewall. Log into camera web interface → **Network** → **RTSP** → Ensure **Enabled** is checked. |
| **Video freezing / Stuttering** | Packet drops on UDP | The application forces **`rtsp_transport=tcp`** to prevent UDP packet loss and corrupted macroblocks. |
| **High Latency (> 500ms)** | Main stream encoder bitrate or resolution too high | Switch Stream Type in the GUI to **Sub Stream (`subtype=1`)** for sub-100ms real-time latency. |
