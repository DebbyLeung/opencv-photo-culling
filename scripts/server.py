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
from pydantic import BaseModel

from scripts.ftp_server import pull_matching_raw, run_ftp_server
from scripts.photo_culling import IncomingPhotoHandler
from scripts.utils import FolderConfig
from scripts.utils.folder_config import FolderNames


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
    folder: Literal[FolderNames.TRASH_FOLDER, FolderNames.WATCHED_FOLDER]


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
        # 儲存目前設定
        self.set_config_data(self.BASE_DIR)
        # Run server inside the passed event loop
        self.ftp_thread = threading.Thread(
            target=run_ftp_server,
            args=(self._config_data.ftp_folder, notify_media_changed),
            daemon=True,
        )
        self.photo_handler = IncomingPhotoHandler(self._config_data)
        self.photo_handler.consumer_thread.start()

    @property
    def config_data(self):
        return self._config_data.model_dump()

    def set_config_data(self, new_dir: str):
        if not new_dir or getattr(self, "base_dir", None) == new_dir:
            return self._config_data
        self.base_dir = new_dir
        self._config_data = FolderConfig(
            ftp_folder=self.base_dir + "/1_Incoming_FTP",
            trash_folder=self.base_dir + "/2_AI_Trash",
            watched_folder=self.base_dir + "/3_Lightroom_Watch",
            destination_folder=self.base_dir + "/Destination",
        )
        if hasattr(self, "observer"):
            self.observer.stop()
            self.observer.join()
        self.setup_observer()
        return self._config_data

    def setup_observer(self):
        """Sets up the watchdog observer to monitor the incoming FTP folder."""
        self.observer.schedule(
            self.photo_handler, path=self._config_data.ftp_folder, recursive=False
        )
        self.observer.start()

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
        "auto_run_filtering": not app.photo_handler.is_manual,
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
    app.photo_handler.is_manual = not config.enabled
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
                incoming_dir=app._config_data.ftp_folder,
                watched_dir=app._config_data.watched_folder,
                trash_dir=app._config_data.trash_folder,
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
@app.get("/api/media/pull-raw")
def start_pull_photos():
    global continue_pull
    if continue_pull:
        return {"message": "Pulling RAW files is already in progress."}
    continue_pull = True
    thread_pool = ThreadPoolExecutor(max_workers=4)
    while continue_pull:
        if app.photo_handler.pull_raw_queue.empty():
            continue
        filename = app.photo_handler.pull_raw_queue.queue.pop()
        thread_pool.submit(
            pull_matching_raw, filename, app._config_data.destination_dir
        )


@app.get("/api/media/stop-pull-raw")
def stop_pull_photos():
    """Stops the pull raw thread pool."""
    # This is a placeholder; implement logic to stop the thread pool if needed.
    global continue_pull
    continue_pull = False


@app.get("/api/media/filtered/{folder}/{filename}")
async def get_filtered_photo(folder: str, filename: str):
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=404, detail="Photo not found")

    folders = {
        FolderNames.TRASH_FOLDER: app._config_data.trash_folder,
        FolderNames.WATCHED_FOLDER: app._config_data.watched_folder,
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
        app._config_data.trash_folder
        if result.folder == FolderNames.TRASH_FOLDER
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
