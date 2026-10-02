[ Camera (Nikon/Canon) ]
           │
           ▼ (Auto-push 2MP JPEG via Wi-Fi 5GHz FTP)
[ Local Laptop FTP Server (Port 21) ]
           │
           ▼ (Watchdog Event Trigger)
┌───────────────────────────────────────────────────────────┐
│ ONNX Runtime + OpenCV Filtering Daemon                    │
│                                                           │
│ 1. Laplacian Variance Check ───(Blurry)─────▸ [ Trash ]   │
│ 2. ONNX Face & Eye Landmark Engine                        │
│    └── Calculate EAR ──────────(Closed Eye)─▸ [ Trash ]   │
└───────────────────────────────────────────────────────────┘
           │
           ▼ (PASSED)
[ Moved to Lightroom Watched Folder ] ──▸ [ Lightroom Auto-Import ]
           │
           ▼ (Optional Stage 2)
[ Active Pull Matching .NEF / .CR3 RAW File From Camera FTP ]
