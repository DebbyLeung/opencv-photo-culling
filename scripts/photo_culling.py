import json
import os
import queue
import threading
import time
from ftplib import FTP

import cv2
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Load Environment Variables
load_dotenv()

CAMERA_IP = os.getenv("CAMERA_IP", "192.168.8.150")
CAMERA_FTP_USER = os.getenv("CAMERA_FTP_USER", "nikon")
CAMERA_FTP_PASS = os.getenv("CAMERA_FTP_PASS", "12345")
CAMERA_RAW_PATH = os.getenv("CAMERA_RAW_PATH", "/Card1/DCIM/100NC_Z8/")
RAW_EXTENSION = os.getenv("RAW_EXTENSION", ".NEF")

BASE_DIR = os.getenv("BASE_DIR", "./Culling_Workflow")
INCOMING_DIR = os.path.join(BASE_DIR, "1_Incoming_FTP")
TRASH_DIR = os.path.join(BASE_DIR, "2_AI_Trash")
LIGHTROOM_WATCH_DIR = os.path.join(BASE_DIR, "3_Lightroom_Watch")

for directory in [INCOMING_DIR, TRASH_DIR, LIGHTROOM_WATCH_DIR]:
    os.makedirs(directory, exist_ok=True)

AUTO_PULL_RAW = os.getenv("AUTO_PULL_RAW", "true").lower() in ("true", "1", "yes")
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 110.0))
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.21))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cached_score")
os.makedirs(CACHE_DIR, exist_ok=True)
APP_BASE_URL = os.getenv("APP_BASE_URL", "localhost:8000")
BASE_DIR_SAFE = BASE_DIR.replace("\\", "/").replace("/", "_")
CACHE_FILE = os.path.join(CACHE_DIR, f"{BASE_DIR_SAFE}.json")

PHOTO_QUEUE = queue.Queue()
QUEUED_FILES = set()
IN_PROGRESS_FILES = set()


# =====================================================================
# ONNX EYE CHECKER CLASS
# =====================================================================
class ONNXEyeChecker:
    def __init__(self, model_path="face_landmarks_68.onnx"):
        self.model_path = model_path
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )

        if os.path.exists(self.model_path):
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.has_model = True
            print(f"[+] Loaded ONNX Model via {providers[0]}")
        else:
            self.has_model = False
            print(
                f"[!] Model '{self.model_path}' not found. Operating in Blur-Only Mode."
            )

    def calculate_ear(self, landmarks):
        l_v1 = np.linalg.norm(landmarks[37] - landmarks[41])
        l_v2 = np.linalg.norm(landmarks[38] - landmarks[40])
        l_h = np.linalg.norm(landmarks[36] - landmarks[39])
        left_ear = (l_v1 + l_v2) / (2.0 * l_h) if l_h > 0 else 0.3

        r_v1 = np.linalg.norm(landmarks[43] - landmarks[47])
        r_v2 = np.linalg.norm(landmarks[44] - landmarks[46])
        r_h = np.linalg.norm(landmarks[42] - landmarks[45])
        right_ear = (r_v1 + r_v2) / (2.0 * r_h) if r_h > 0 else 0.3

        return (left_ear + right_ear) / 2.0

    def check_eyes_open(self, image_path):
        if not self.has_model:
            return True, "Model Skipped", 0.0

        image = cv2.imread(image_path)
        if image is None:
            return False, "Unreadable Image", 0.0

        h, w = image.shape[:2]
        resized = cv2.resize(image, (128, 128))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})

        landmarks = outputs[0].reshape(-1, 2) * np.array([w / 128.0, h / 128.0])
        ear_score = self.calculate_ear(landmarks)

        if ear_score < EAR_THRESHOLD:
            return False, f"Closed Eyes (EAR: {ear_score:.2f})", float(ear_score)

        return True, f"Open Eyes (EAR: {ear_score:.2f})", float(ear_score)


eye_checker = ONNXEyeChecker()


# =====================================================================
# CULLING & RAW PULL LOGIC
# =====================================================================
def calculate_blur_score(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("-inf")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def is_blurry(file_path, threshold=BLUR_THRESHOLD):
    return calculate_blur_score(file_path) < threshold


def save_score_cache(
    filename, status, blur_score, ear_score, reason, folder_path, app_base_dir=BASE_DIR
):
    """Persist the last score result for this app base URL as JSON cache."""
    cache_payload = {"app_base_url": APP_BASE_URL, "photos": []}
    app_base_dir_safe = app_base_dir.replace("\\", "/").replace("/", "_")
    cache_file = os.path.join(CACHE_DIR, f"{app_base_dir_safe}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as cache_handle:
                loaded_payload = json.load(cache_handle)
            if isinstance(loaded_payload, dict):
                cache_payload = loaded_payload
        except (json.JSONDecodeError, OSError):
            cache_payload = {"app_base_url": APP_BASE_URL, "photos": []}

    if not isinstance(cache_payload.get("photos"), list):
        cache_payload["photos"] = []

    safe_blur_score = float(blur_score) if np.isfinite(float(blur_score)) else 0.0
    safe_ear_score = float(ear_score) if np.isfinite(float(ear_score)) else 0.0

    record = {
        "filename": filename,
        "status": status,
        "blur_score": round(safe_blur_score, 2),
        "ear_score": round(safe_ear_score, 2),
        "reason": reason,
        "folder_path": folder_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    existing_index = next(
        (
            index
            for index, item in enumerate(cache_payload["photos"])
            if item.get("filename") == filename
        ),
        None,
    )
    if existing_index is None:
        cache_payload["photos"].append(record)
    else:
        cache_payload["photos"][existing_index] = record

    cache_payload["app_base_url"] = APP_BASE_URL
    cache_payload["last_updated"] = record["timestamp"]

    with open(cache_file, "w", encoding="utf-8") as cache_handle:
        json.dump(cache_payload, cache_handle, indent=2, sort_keys=True)


def pull_matching_raw(raw_filename):
    """Logs into the camera's native FTP server to retrieve matching RAW file."""
    local_raw_path = os.path.join(LIGHTROOM_WATCH_DIR, raw_filename)
    try:
        ftp = FTP(CAMERA_IP, timeout=5)
        ftp.login(user=CAMERA_FTP_USER, passwd=CAMERA_FTP_PASS)
        ftp.cwd(CAMERA_RAW_PATH)

        print(f"[*] Actively Pulling {raw_filename} from Camera ({CAMERA_IP})...")
        with open(local_raw_path, "wb") as f:
            ftp.retrbinary(f"RETR {raw_filename}", f.write)
        ftp.quit()
        print(f"[✓] RAW Download Complete -> {local_raw_path}")
    except Exception as e:
        print(f"[X] Camera RAW Pull Failed: {e}")


def process_photo(
    file_path,
    incoming_dir=INCOMING_DIR,
    trash_dir=TRASH_DIR,
    lightroom_watch_dir=LIGHTROOM_WATCH_DIR,
):
    filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(filename)
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
        return

    # Step 2: ONNX Eye Aspect Ratio Check
    eyes_open, reason, ear_score = eye_checker.check_eyes_open(file_path)
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
        return

    # Step 3: PASSED -> Move to Lightroom Watched Folder
    print(f"[✓] PASSED: {filename} ({reason}) -> Lightroom Watch Folder")
    passed_jpg_path = os.path.join(lightroom_watch_dir, filename)
    os.rename(file_path, passed_jpg_path)
    save_score_cache(
        filename,
        "passed",
        blur_score,
        ear_score,
        reason,
        os.path.basename(lightroom_watch_dir),
        app_base_dir=incoming_dir,
    )

    # Step 4: Optional Active RAW Pull over Camera FTP
    if AUTO_PULL_RAW:
        raw_filename = base_name + RAW_EXTENSION
        threading.Thread(target=pull_matching_raw, args=(raw_filename,)).start()


# =====================================================================
# WATCHDOG FILE LISTENER
# =====================================================================
def enqueue_photo(file_path, incoming_dir=INCOMING_DIR):
    """Add JPG/JPEG uploads to the culling queue when they are not already queued."""
    if not os.path.isfile(file_path):
        return
    if not file_path.lower().endswith((".jpg", ".jpeg")):
        return

    normalized = os.path.abspath(file_path)
    if normalized in QUEUED_FILES or normalized in IN_PROGRESS_FILES:
        return

    QUEUED_FILES.add(normalized)
    PHOTO_QUEUE.put((file_path, incoming_dir))
    print(f"[+] Queued for culling: {os.path.basename(file_path)}")


def drain_existing_photos(incoming_dir=INCOMING_DIR):
    """Queue any files already in the incoming folder when the service starts."""
    if not os.path.isdir(incoming_dir):
        return
    for filename in sorted(os.listdir(incoming_dir)):
        file_path = os.path.join(incoming_dir, filename)
        if os.path.isfile(file_path):
            enqueue_photo(file_path, incoming_dir=incoming_dir)


def process_queue():
    """Consume the incoming photo queue continuously with the culling workflow."""
    while True:
        file_path, incoming_dir = PHOTO_QUEUE.get()
        normalized = os.path.abspath(file_path)
        try:
            IN_PROGRESS_FILES.add(normalized)
            if not os.path.exists(file_path):
                continue
            process_photo(file_path, incoming_dir=incoming_dir)
        except Exception as exc:
            print(f"[!] Culling queue error for {file_path}: {exc}")
        finally:
            QUEUED_FILES.discard(normalized)
            IN_PROGRESS_FILES.discard(normalized)
            PHOTO_QUEUE.task_done()


class IncomingPhotoHandler(FileSystemEventHandler):
    def __init__(
        self,
        incoming_dir=INCOMING_DIR,
        trash_dir=TRASH_DIR,
        lightroom_watch_dir=LIGHTROOM_WATCH_DIR,
    ):
        self.incoming_dir = incoming_dir
        self.trash_dir = trash_dir
        self.lightroom_watch_dir = lightroom_watch_dir

    def on_created(self, event):
        if not event.is_directory:
            enqueue_photo(event.src_path, incoming_dir=self.incoming_dir)


if __name__ == "__main__":
    consumer_thread = threading.Thread(target=process_queue, daemon=True)
    consumer_thread.start()

    drain_existing_photos(incoming_dir=INCOMING_DIR)

    observer = Observer()
    observer.schedule(
        IncomingPhotoHandler(
            incoming_dir=INCOMING_DIR,
            trash_dir=TRASH_DIR,
            lightroom_watch_dir=LIGHTROOM_WATCH_DIR,
        ),
        path=INCOMING_DIR,
        recursive=False,
    )
    observer.start()
    print(f"[+] Active Culling Engine watching: {INCOMING_DIR}")
    print(f"[+] Approved JPEGs/RAWs routing to: {LIGHTROOM_WATCH_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
