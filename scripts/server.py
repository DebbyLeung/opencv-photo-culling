import asyncio
import json
import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()


# Enable CORS for local Tailwind frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FolderConfig(BaseModel):
    watched_folder: str
    destination_folder: str


# 儲存目前設定
config_data = {
    "watched_folder": "./Culling_Workflow/3_Lightroom_Watch",
    "destination_folder": "./Culling_Workflow/Destination",
}


@app.get("/api/config")
async def get_config():
    return config_data


@app.post("/api/config")
async def update_config(config: FolderConfig):
    # 自動建立不存在的資料夾目錄
    os.makedirs(config.watched_folder, exist_ok=True)
    os.makedirs(config.destination_folder, exist_ok=True)

    config_data["watched_folder"] = config.watched_folder
    config_data["destination_folder"] = config.destination_folder
    return {"message": "Folder paths updated successfully", "config": config_data}


# ---------------------------------------------------------------------
# WEBSOCKET CONNECTION MANAGER
# ---------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    @property
    def active_con_count(self):
        return len(self.active_connections)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[+] Client connected to WS. Active clients: {self.active_con_count}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[-] Client disconnected. Active clients: {self.active_con_count}")

    async def broadcast(self, data: dict):
        """Sends data payload to all connected frontend clients."""
        if not self.active_connections:
            return

        message = json.dumps(data)
        # Send concurrently to all open sockets
        await asyncio.gather(
            *[connection.send_text(message) for connection in self.active_connections],
            return_exceptions=True,
        )


# Global Manager Variable
manager = ConnectionManager()

# Global Event Loop reference for thread-safe broadcasting
MAIN_EVENT_LOOP: asyncio.AbstractEventLoop = None

# Mount Static Directories so Frontend can stream preview JPEGs
# Mount the frontend directory at the root path
BASE_DIR = "./Culling_Workflow"
os.makedirs(BASE_DIR, exist_ok=True)
app.mount("/previews", StaticFiles(directory=BASE_DIR), name="previews")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep socket alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------
# BROADCAST HELPER FUNCTION (Thread-Safe Trigger)
# ---------------------------------------------------------------------
def async_broadcast(
    filename: str,
    status: str,
    blur_score: float,
    ear_score: float,
    reason: str = "",
    folder_path: str = "3_Lightroom_Watch",
):
    """
    Thread-safe helper called from culling_daemon.py.
    Schedules the broadcast on the main asyncio event loop.
    """
    payload = {
        "event": "NEW_PHOTO",
        "filename": filename,
        "url": f"http://localhost:8000/previews/{folder_path}/{filename}",
        "status": status,  # "passed" or "rejected"
        "blurScore": round(float(blur_score), 1),
        "earScore": round(float(ear_score), 2),
        "reason": reason,
    }

    # Dispatch to the running FastAPI asyncio loop safely from worker threads
    if MAIN_EVENT_LOOP and MAIN_EVENT_LOOP.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), MAIN_EVENT_LOOP)
    else:
        print(f"[!] Warning: Event loop not ready. Could not broadcast {filename}")


def start_server_in_loop(loop: asyncio.AbstractEventLoop, port: int = 8000):
    """Sets the global loop reference and starts Uvicorn."""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = loop

    config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Run server inside the passed event loop
    loop.run_until_complete(server.serve())
