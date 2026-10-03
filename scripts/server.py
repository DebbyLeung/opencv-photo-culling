import asyncio
import json
import os
import threading

# Tkinter for native OS folder picker
import tkinter as tk
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from tkinter import filedialog
from typing import Literal
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from scripts.filtering_settings import is_filtering_enabled, set_filtering_enabled
from scripts.ftp_server import run_ftp_server


@asynccontextmanager
async def app_lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    _app.main_event_loop = asyncio.get_running_loop()
    yield


class FilteringConfig(BaseModel):
    enabled: bool


class FilteredPhotoResult(BaseModel):
    filename: str
    status: Literal["passed", "rejected"]
    blur_score: float
    ear_score: float | None = None
    reason: str = ""
    folder: Literal["2_AI_Trash", "3_Lightroom_Watch"]


class FolderConfig(BaseModel):
    ftp_folder: str
    watched_folder: str
    destination_folder: str

    @model_validator(mode="after")
    def create_directories(self):
        # 此時 self 的欄位已經完成初始化與型別檢查
        os.makedirs(self.watched_folder, exist_ok=True)
        os.makedirs(self.destination_folder, exist_ok=True)
        os.makedirs(self.ftp_folder, exist_ok=True)

        return self


def notify_media_changed(filename: str):
    loop = app.main_event_loop
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(
                {"event": "MEDIA_CHANGED", "filename": os.path.basename(filename)}
            ),
            loop,
        )


class AppManager(FastAPI):
    BASE_DIR = os.getenv("BASE_DIR", "./Culling_Workflow")

    def __init__(self):
        super().__init__(lifespan=app_lifespan)
        # Enable CORS for local Tailwind frontend
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.set_config_data(self.BASE_DIR)
        # 儲存目前設定
        self.main_event_loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        # Run server inside the passed event loop
        self.ftp_thread = threading.Thread(
            target=run_ftp_server,
            args=(self._config_data.ftp_folder, notify_media_changed),
            daemon=True,
        )

    @property
    def config_data(self):
        return self._config_data.model_dump()

    def set_config_data(self, new_dir: str):
        if not new_dir or getattr(self, "base_dir", None) == new_dir:
            return self._config_data
        self.base_dir = new_dir
        self._config_data = FolderConfig(
            **{
                "ftp_folder": self.base_dir + "/1_Incoming_FTP",
                "watched_folder": self.base_dir + "/3_Lightroom_Watch",
                "destination_folder": self.base_dir + "/Destination",
            }
        )
        os.makedirs(self.base_dir + "/2_AI_Trash", exist_ok=True)
        return self._config_data

    def start_server_in_loop(self, port: int = 8000):
        """Sets the global loop reference and starts Uvicorn."""
        config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)
        self.ftp_thread.start()
        self.main_event_loop.run_until_complete(server.serve())


app = AppManager()


@app.get("/api/config")
async def get_config():
    return {
        **app.config_data,
        "workflow_folder": app.base_dir,
        "auto_run_filtering": is_filtering_enabled(app.BASE_DIR),
    }


@app.post("/api/config/workflow-folder")
async def set_workflow_folder():
    workflow_folder = await browse_folder()
    app.set_config_data(workflow_folder)
    return {"message": "Workflow folder updated successfully", "path": workflow_folder}


executor = ThreadPoolExecutor(max_workers=1)


async def browse_folder():
    def _open_native_folder_picker():
        """Opens OS Native File Explorer / Finder window."""
        root = tk.Tk()
        root.withdraw()  # Hide main empty window
        root.attributes("-topmost", True)  # Bring window to front

        selected_path = filedialog.askdirectory(
            initialdir=app.base_dir,
            title="Select Folder",
        )
        root.destroy()
        return selected_path

    # Run GUI file picker in background thread
    chosen_path = await app.main_event_loop.run_in_executor(
        executor, _open_native_folder_picker
    )
    return chosen_path if chosen_path else app.base_dir


@app.post("/api/config/auto-run-filtering")
async def set_auto_run_filtering(config: FilteringConfig):
    set_filtering_enabled(app.base_dir, config.enabled)
    return {"auto_run_filtering": config.enabled}


@app.post("/api/filter/run")
async def run_filter():
    incoming_dir = app._config_data.ftp_folder
    file_paths = [
        entry.path
        for entry in os.scandir(incoming_dir)
        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg"))
    ]
    if not file_paths:
        return {"processed": 0}

    from scripts.photo_culling import process_photo

    def process_pending_files():
        for file_path in file_paths:
            process_photo(
                file_path,
                watched_dir=app._config_data.watched_folder,
                trash_dir=os.path.join(app.base_dir, "2_AI_Trash"),
            )

    await asyncio.to_thread(process_pending_files)
    return {"processed": len(file_paths)}


@app.get("/api/media")
async def get_media():
    incoming_dir = app._config_data.ftp_folder
    raw_dir = app._config_data.watched_folder

    jpg_files = []
    if os.path.isdir(incoming_dir):
        jpg_files = [
            {
                "filename": entry.name,
                "url": f"/api/media/jpg/{quote(entry.name)}",
            }
            for entry in os.scandir(incoming_dir)
            if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg"))
        ]

    raw_files = []
    if os.path.isdir(raw_dir):
        raw_files = [
            {"filename": entry.name, "size": entry.stat().st_size}
            for entry in os.scandir(raw_dir)
            if entry.is_file()
            and entry.name.lower().endswith(
                (
                    ".nef",
                    ".cr2",
                    ".cr3",
                    ".arw",
                    ".dng",
                    ".orf",
                    ".rw2",
                    ".raf",
                    ".pef",
                    ".srw",
                )
            )
        ]

    return {
        "jpg": sorted(jpg_files, key=lambda item: item["filename"].lower()),
        "raw": sorted(raw_files, key=lambda item: item["filename"].lower()),
    }


@app.get("/api/media/jpg/{filename}")
async def get_incoming_jpg(filename: str):
    if os.path.basename(filename) != filename or not filename.lower().endswith(
        (".jpg", ".jpeg")
    ):
        raise HTTPException(status_code=404, detail="JPG not found")

    file_path = os.path.join(app._config_data.ftp_folder, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="JPG not found")
    return FileResponse(file_path)


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


manager = ConnectionManager()


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
    ear_score: float | None,
    reason: str = "",
    folder_path: str = "3_Lightroom_Watch",
    blur_threshold: float = 110.0,
    ear_threshold: float = 0.21,
):
    """
    Thread-safe helper called from culling_daemon.py.
    Schedules the broadcast on the main asyncio event loop.
    """
    # Dispatch to the running FastAPI asyncio loop safely from worker threads
    if not (app.main_event_loop and app.main_event_loop.is_running()):
        print(f"[!] Warning: Event loop not ready. Could not broadcast {filename}")
        return
    payload = {
        "event": "NEW_PHOTO",
        "filename": filename,
        "url": f"/api/media/filtered/{folder_path}/{quote(filename)}",
        "status": status,  # "passed" or "rejected"
        "blurScore": round(float(blur_score), 1),
        "earScore": round(float(ear_score), 2) if ear_score is not None else None,
        "blurThreshold": blur_threshold,
        "earThreshold": ear_threshold,
        "reason": reason,
    }

    asyncio.run_coroutine_threadsafe(manager.broadcast(payload), app.main_event_loop)


@app.get("/api/media/filtered/{folder}/{filename}")
async def get_filtered_photo(folder: str, filename: str):
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=404, detail="Photo not found")

    folders = {
        "2_AI_Trash": os.path.join(app.base_dir, "2_AI_Trash"),
        "3_Lightroom_Watch": app._config_data.watched_folder,
    }
    directory = folders.get(folder)
    if directory is None:
        raise HTTPException(status_code=404, detail="Photo not found")

    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(file_path)


@app.post("/api/filter/result", status_code=202)
async def publish_filtered_photo(result: FilteredPhotoResult):
    if os.path.basename(result.filename) != result.filename:
        raise HTTPException(status_code=400, detail="Invalid photo filename")
    directory = (
        os.path.join(app.base_dir, result.folder)
        if result.folder == "2_AI_Trash"
        else app._config_data.watched_folder
    )
    if not os.path.isfile(os.path.join(directory, result.filename)):
        raise HTTPException(status_code=404, detail="Filtered photo not found")

    payload = {
        "event": "NEW_PHOTO",
        "filename": result.filename,
        "url": f"/api/media/filtered/{result.folder}/{quote(result.filename)}",
        "status": result.status,
        "blurScore": round(result.blur_score, 1),
        "earScore": (
            round(result.ear_score, 2) if result.ear_score is not None else None
        ),
        "blurThreshold": float(os.getenv("BLUR_THRESHOLD", "110.0")),
        "earThreshold": float(os.getenv("EAR_THRESHOLD", "0.21")),
        "reason": result.reason,
    }
    await manager.broadcast(payload)
    return {"accepted": True}


# Mount the frontend only after all API and WebSocket routes are registered.
# Otherwise the catch-all ``/`` static mount can intercept /ws and reject it with 403.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
if __name__ == "__main__":
    app.start_server_in_loop()
