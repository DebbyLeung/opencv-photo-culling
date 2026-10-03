import json
import logging
import os
import queue
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv
from fastapi import Request
from utils import FolderConfig
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from scripts.photo_culling import ONNXEyeChecker, calculate_blur_score, save_score_cache
from scripts.utils.folder_config import FolderNames

load_dotenv()

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 110.0))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.21))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
APP_BASE_URL = os.getenv("APP_BASE_URL", "localhost:8000")
BASE_DIR = os.getenv("BASE_DIR", "./Culling_Workflow")
BASE_DIR_SAFE = BASE_DIR.replace("\\", "/").replace("/", "_")

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cached_score")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, f"{BASE_DIR_SAFE}.json")
AUTO_PULL_RAW = os.getenv("AUTO_PULL_RAW", "true").lower() in ("true", "1", "yes")


# =====================================================================
# WATCHDOG FILE LISTENER
# =====================================================================
class IncomingPhotoHandler(FileSystemEventHandler):
    def __init__(self, folder_config: FolderConfig):
        self.folder_config = folder_config
        self.workflow_dir = os.path.dirname(os.path.normpath(folder_config.ftp_folder))
        self.photo_queue = queue.Queue()
        self.pull_raw_queue = queue.Queue()
        self.in_progress_files = set()
        self.queued_files = set()
        self.consumer_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.consumer_thread.start()
        self._is_manual = True
        self.eye_checker = ONNXEyeChecker()

    def __del__(self):
        self.photo_queue.put(None)  # Signal to stop the consumer thread
        self.consumer_thread.join()  # Wait for the consumer thread to finish

    @property
    def is_manual(self):
        return self._is_manual

    @is_manual.setter
    def is_manual(self, value: bool):
        self._is_manual = value
        if value is False:
            self.drain_existing_photos(self.folder_config)

    def on_created(self, event):
        if self.is_manual:
            return
        if event.is_directory:
            return

        if event.src_path.lower().endswith((".jpg", ".jpeg")):
            self.enqueue_photo(event.src_path)

    def enqueue_photo(self, file_path):
        """Add JPG/JPEG uploads to the culling queue when they are not already queued."""
        if not os.path.isfile(file_path):
            return
        if not file_path.lower().endswith((".jpg", ".jpeg")):
            return

        normalized = os.path.abspath(file_path)
        if normalized in self.queued_files or normalized in self.in_progress_files:
            return

        self.queued_files.add(normalized)
        self.photo_queue.put(file_path)
        print(f"[+] Queued for culling: {os.path.basename(file_path)}")

    def process_pending_photos(self):
        for entry in os.scandir(self.folder_config.ftp_folder):
            self.enqueue_photo(entry.path)

    def drain_existing_photos(self, folder_config: FolderConfig):
        """Queue any files already in the incoming folder when the service starts."""
        incoming_dir = folder_config.ftp_folder
        if not os.path.isdir(incoming_dir):
            return
        for filename in sorted(os.listdir(incoming_dir)):
            file_path = os.path.join(incoming_dir, filename)
            self.enqueue_photo(file_path)

    def process_queue(self):
        """Consume the incoming photo queue continuously with the culling workflow."""
        while True:
            file_path = self.photo_queue.get()
            normalized = os.path.abspath(file_path)
            try:
                self.in_progress_files.add(normalized)
                if not os.path.exists(file_path):
                    continue
                self.process_photo(file_path)
            except Exception as exc:
                print(f"[!] Culling queue error for {file_path}: {exc}")
            finally:
                self.queued_files.discard(normalized)
                self.in_progress_files.discard(normalized)
                self.photo_queue.task_done()

    def process_photo(
        self,
        file_path,
    ):
        incoming_dir = self.folder_config.incoming_dir
        trash_dir = self.folder_config.trash_dir
        watched_dir = self.folder_config.watched_dir
        filename = os.path.basename(file_path)
        time.sleep(0.05)  # Buffer to allow OS write-lock release

        blur_score = calculate_blur_score(file_path)

        # Step 1: Laplacian Blur Check
        if blur_score < BLUR_THRESHOLD:
            print(f"[X] REJECTED (Blur): {filename} -> Trash")
            os.rename(file_path, os.path.join(trash_dir, filename))
            save_score_cache(
                filename,
                "rejected",
                blur_score,
                0.0,
                "Blur",
                os.path.basename(trash_dir),
                app_base_dir=incoming_dir,
            )
            _broadcast_filtered_photo(
                filename,
                "rejected",
                blur_score,
                None,
                "Blur",
                FolderNames.TRASH_FOLDER,
            )
            return

        # Step 2: ONNX Eye Aspect Ratio Check
        eyes_open, reason, ear_score = self.eye_checker.check_eyes_open(file_path)
        if not eyes_open:
            print(f"[X] REJECTED ({reason}): {filename} -> Trash")
            os.rename(file_path, os.path.join(trash_dir, filename))
            save_score_cache(
                filename,
                "rejected",
                blur_score,
                ear_score,
                reason,
                os.path.basename(trash_dir),
                app_base_dir=incoming_dir,
            )
            _broadcast_filtered_photo(
                filename,
                "rejected",
                blur_score,
                ear_score,
                reason,
                FolderNames.TRASH_FOLDER,
            )
            return

        # Step 3: PASSED -> Move to Lightroom Watched Folder
        print(f"[✓] PASSED: {filename} ({reason}) -> Lightroom Watch Folder")
        passed_jpg_path = os.path.join(watched_dir, filename)
        os.rename(file_path, passed_jpg_path)
        save_score_cache(
            filename,
            "passed",
            blur_score,
            ear_score,
            reason,
            os.path.basename(watched_dir),
            app_base_dir=incoming_dir,
        )
        _broadcast_filtered_photo(
            filename,
            "passed",
            blur_score,
            ear_score,
            reason,
            FolderNames.WATCHED_FOLDER,
        )
        if AUTO_PULL_RAW:
            self.pull_raw_queue.put(filename)


if __name__ == "__main__":
    print("[+] Starting Culling Engine...")
    INCOMING_DIR = os.path.join(BASE_DIR, FolderNames.FTP_FOLDER)
    TRASH_DIR = os.path.join(BASE_DIR, FolderNames.TRASH_FOLDER)
    LIGHTROOM_WATCH_DIR = os.path.join(BASE_DIR, FolderNames.WATCHED_FOLDER)

    folder_config = FolderConfig(
        ftp_folder=INCOMING_DIR,
        trash_folder=TRASH_DIR,
        watched_folder=LIGHTROOM_WATCH_DIR,
        destination_folder=os.path.join(BASE_DIR, FolderNames.DESTINATION_FOLDER),
    )

    handler = IncomingPhotoHandler(folder_config)
    print("[+] Setup consumer")
    handler.consumer_thread.start()
    handler.drain_existing_photos(folder_config)
    print("[+] Setup observer")
    observer = Observer()
    observer.schedule(handler, path=folder_config.ftp_folder, recursive=False)
    observer.start()
    print(f"[+] Active Culling Engine watching: {folder_config.ftp_folder}")
    print(f"[+] Approved JPEGs/RAWs routing to: {folder_config.watched_folder}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _broadcast_filtered_photo(
    filename: str,
    status: str,
    blur_score: float,
    ear_score: float | None,
    reason: str,
    folder_path: str,
) -> None:
    payload = {
        "filename": filename,
        "status": status,
        "blur_score": blur_score,
        "ear_score": ear_score,
        "reason": reason,
        "folder": folder_path,
    }
    try:
        DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://127.0.0.1:8000").rstrip("/")
        request = Request(
            f"{DASHBOARD_URL}/api/filter/result",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            if response.status != 202:
                raise RuntimeError(
                    f"Dashboard returned unexpected status {response.status}"
                )
    except (URLError, OSError, RuntimeError):
        logging.exception("Could not publish filtered photo result: %s", filename)
