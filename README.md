# ⚡ Edge-AI Photography Culling & RAW Sync Engine

An ultra-low latency, real-time automated photo culling and RAW sync pipeline designed for high-volume event photographers (Nikon Z8/Z9, Canon R3).

This system receives auto-pushed 2MP JPEG previews over Wi-Fi (FTP) directly from professional cameras, runs lightweight AI quality inspection (motion blur and eye-blink detection) in **<15ms per frame**, broadcasts results to a real-time web dashboard, and automatically pulls matching high-resolution RAW files (`.NEF`, `.CR3`) directly into Adobe Lightroom Classic's Auto-Import directory.

---

## 🏗️ System Architecture

```
                                  [ Professional Camera ]
                                (Nikon Z8/Z9 / Canon R3)
                                           │
                                           │ (1) Auto-Push 2MP JPEGs via FTP (5GHz Wi-Fi)
                                           ▼
                                 [ Laptop FTP Receiver ]
                                    (ftp_server.py)
                                           │
                                           │ (2) Writes to 1_Incoming_FTP/
                                           ▼
                          ┌────────────────────────────────┐
                          │     AI Culling Daemon          │
                          │    (culling_daemon.py)         │
                          ├────────────────────────────────┤
                          │ • Laplacian Blur Check (<2ms)  │
                          │ • ONNX Eye Landmark EAR (<10ms)│
                          └───────────────┬────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │ (3) Rejected                                │ (3) Passed
                   ▼                                             ▼
            [ 2_AI_Trash/ ]                            [ 3_Lightroom_Watch/ ]
                   │                                             │
                   │                                             ├──────────────┐
                   ▼                                             ▼              ▼
           [ Web Dashboard ]                           [ Active RAW Pull ] [ Lightroom ]
            (Alpine/Tailwind)                         (Camera FTP Client)  (Auto-Import)
```

---

## ✨ Key Features

- **⚡ Sub-15ms Latency:** Uses OpenCV (Laplacian variance) and ONNX Runtime for facial landmark tracking. Bypasses heavy deep-learning frameworks like PyTorch for near-zero memory footprint (~150MB RAM).
- **🎯 Dual Quality Inspection:**
  1. **Blur Detection:** Filters out motion blur and missed focus frames via Laplacian Variance analysis.
  2. **Blink Detection:** Extracts 68 facial landmarks to compute **Eye Aspect Ratio (EAR)** and rejects closed-eye or mid-blink shots.
- **🔄 Active Option-B RAW Sync:** When a JPEG passes inspection, the daemon connects back to the camera's internal FTP file structure to pull the matching high-res RAW (`.NEF` / `.CR3`).
- **📊 Real-Time Web Dashboard:** Web-based UI built with Tailwind CSS, Alpine.js, and FastAPI WebSockets for live scoring visualization.
- **⏯️ Filtering Control:** The dashboard's **Auto Run Filtering** checkbox enables or pauses automatic filtering of incoming JPEG uploads.
- **▶️ Manual Filtering:** Use **Run Filter** in the dashboard to filter JPEGs already waiting in the incoming folder.
- **⭐ Quality Ranking:** The dashboard ranks filtered photos from 5 to 1 using the configured blur and eye thresholds, and auto-selects score-5 photos.
- **📁 Automated Lightroom Routing:** Directs approved JPEGs and synced RAWs straight into Lightroom Classic's watched folder for seamless auto-importing.

---

## 🛠️ Project Structure

```text
.
├── .env                  # Cross-module configuration & credentials
├── data/models/
│   └── face_landmarks_68.onnx # ONNX model for facial landmark detection
├── docs/
├── frontend/
│   └── index.html        # Tailwind CSS + Alpine.js Web Dashboard
├── scripts/
│   ├── ftp_server.py     # Standalone FTP Server (Receives camera JPEG uploads)
│   ├── culling_daemon.py # AI Culling Engine & Active RAW puller
│   └── server.py         # FastAPI & WebSocket server for live UI streaming
└── Culling_Workflow/     # Working directory structure
    ├── 1_Incoming_FTP/   # Target folder for camera auto-uploads
    ├── 2_AI_Trash/       # Folder for rejected JPEGs
    └── 3_Lightroom_Watch/# Folder for approved JPEGs & pulled RAWs
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.9+**
- **Hardware:** Works on any modest CPU (Intel i5/Ryzen 5 or Apple Silicon M1+). No dedicated GPU required.

### 2. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/DebbyLeung/opencv-photo-culling.git
cd opencv-photo-culling

cp sample.env .env
pip install -r requirements.txt
mkdir -p data/model && curl -L -o data/face_landmarks_68.onnx "https://huggingface.co"

```

*Note: For NVIDIA GPU acceleration, install `onnxruntime-gpu` instead of `onnxruntime`.*

### 3. Environment Configuration (`.env`)

Edit a `.env` file in the root directory:

```ini
# Camera FTP Connection (For Active RAW Pull)
CAMERA_IP=192.168.8.150
CAMERA_FTP_USER=nikon
CAMERA_FTP_PASS=12345
CAMERA_RAW_PATH=/Card1/DCIM/100NC_Z8/
RAW_EXTENSION=.NEF

# Local Laptop FTP Server Settings (For Incoming Camera JPEGs)
LAPTOP_FTP_PORT=21
LAPTOP_FTP_USER=camera
LAPTOP_FTP_PASS=12345

# Pipeline Directory
BASE_DIR=./Culling_Workflow
DASHBOARD_URL=http://127.0.0.1:8000

# Culling Thresholds
AUTO_PULL_RAW=true
BLUR_THRESHOLD=110.0
EAR_THRESHOLD=0.21
```

If the dashboard server runs on a different port, set `DASHBOARD_URL` to that
server address so filtered photo scores can be sent to the dashboard.

---

## 💻 Running the System

To run the complete pipeline, open three separate terminal windows:

### Terminal 1: Local FTP Receiver
Start the background FTP server to receive incoming 2MP JPEGs from your camera:
```bash
python scripts/ftp_server.py
```

### Terminal 2: Culling Daemon & WebSocket Server
Start the AI inspection daemon and live WebSocket server:
```bash
python scripts/photo_culling.py
```

### Terminal 3: Web Dashboard
Open `index.html` in your web browser, or serve it using Python's static server:
```bash
# Runs app = FastAPI() inside scripts/server.py
uvicorn scripts.server:app --port 8080
```
Navigate to `http://localhost:8080` to view the live dashboard.

---

## 🎨 Setting Up Lightroom Classic

1. Open Adobe Lightroom Classic.
2. Go to **File** ➔ **Auto Import** ➔ **Auto Import Settings...**
3. Check **Enable Auto Import**.
4. Set **Watched Folder** to: `./Culling_Workflow/3_Lightroom_Watch/`
5. Set **Destination Folder** to your preferred local or external event drive.

---

## 📸 Camera Wi-Fi Setup Guide

1. Connect your laptop to your portable Wi-Fi router (or camera's Wi-Fi hotspot).
2. Configure your camera's Wi-Fi FTP profile:
   - **Target IP:** Laptop IP address on the network
   - **Port:** `21`
   - **User/Password:** As specified in `.env` (`LAPTOP_FTP_USER` / `LAPTOP_FTP_PASS`)
   - **Transfer Target:** JPEG Only (2MP / Slot 2)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
