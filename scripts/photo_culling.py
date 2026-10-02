import os
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
            return True, "Model Skipped"

        image = cv2.imread(image_path)
        if image is None:
            return False, "Unreadable Image"

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
            return False, f"Closed Eyes (EAR: {ear_score:.2f})"

        return True, f"Open Eyes (EAR: {ear_score:.2f})"


eye_checker = ONNXEyeChecker()


# =====================================================================
# CULLING & RAW PULL LOGIC
# =====================================================================
def is_blurry(file_path, threshold=BLUR_THRESHOLD):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    return cv2.Laplacian(image, cv2.CV_64F).var() < threshold


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


def process_photo(file_path):
    filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(filename)
    time.sleep(0.05)  # Buffer to allow OS write-lock release

    # Step 1: Laplacian Blur Check
    if is_blurry(file_path):
        print(f"[X] REJECTED (Blur): {filename} -> Trash")
        os.rename(file_path, os.path.join(TRASH_DIR, filename))
        return

    # Step 2: ONNX Eye Aspect Ratio Check
    eyes_open, reason = eye_checker.check_eyes_open(file_path)
    if not eyes_open:
        print(f"[X] REJECTED ({reason}): {filename} -> Trash")
        os.rename(file_path, os.path.join(TRASH_DIR, filename))
        return

    # Step 3: PASSED -> Move to Lightroom Watched Folder
    print(f"[✓] PASSED: {filename} ({reason}) -> Lightroom Watch Folder")
    passed_jpg_path = os.path.join(LIGHTROOM_WATCH_DIR, filename)
    os.rename(file_path, passed_jpg_path)

    # Step 4: Optional Active RAW Pull over Camera FTP
    if AUTO_PULL_RAW:
        raw_filename = base_name + RAW_EXTENSION
        threading.Thread(target=pull_matching_raw, args=(raw_filename,)).start()


# =====================================================================
# WATCHDOG FILE LISTENER
# =====================================================================
class IncomingPhotoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(
            (".jpg", ".jpeg")
        ):
            threading.Thread(target=process_photo, args=(event.src_path,)).start()


if __name__ == "__main__":
    observer = Observer()
    observer.schedule(IncomingPhotoHandler(), path=INCOMING_DIR, recursive=False)
    observer.start()
    print(f"[+] Active Culling Engine watching: {INCOMING_DIR}")
    print(f"[+] Approved JPEGs/RAWs routing to: {LIGHTROOM_WATCH_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
