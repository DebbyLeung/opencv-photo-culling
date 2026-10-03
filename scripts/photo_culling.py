import json
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv
from utils.ort_loader import load_my_model

# Load Environment Variables
load_dotenv()

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD") or 110.0)
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD") or 0.21)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
APP_BASE_URL = os.getenv("APP_BASE_URL", "localhost:8000")
BASE_DIR = os.getenv("BASE_DIR", "./Culling_Workflow")
BASE_DIR_SAFE = BASE_DIR.replace("\\", "/").replace("/", "_")

MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cached_score")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, f"{BASE_DIR_SAFE}.json")
AUTO_PULL_RAW = os.getenv("AUTO_PULL_RAW", "true").lower() in ("true", "1", "yes")


# =====================================================================
# ONNX EYE CHECKER CLASS
# =====================================================================
class ONNXEyeChecker:
    def __init__(self, model_path="face_landmarks_68.onnx"):
        self.model_path = os.path.join(MODEL_DIR, model_path)
        if os.path.exists(self.model_path):
            self.session = load_my_model(self.model_path)
            self.has_model = True
            print(f"[+] Loaded ONNX Model via {self.session.get_providers()[0]}")
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
            return True, "Model Skipped", None

        image = cv2.imread(image_path)
        if image is None:
            return False, "Unreadable Image", None

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
            return False, f"Closed Eyes (EAR: {ear_score:.2f})", ear_score

        return True, f"Open Eyes (EAR: {ear_score:.2f})", ear_score


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

    safe_blur_score = (
        float(blur_score)
        if blur_score is not None and np.isfinite(float(blur_score))
        else 0.0
    )
    safe_ear_score = (
        float(ear_score)
        if ear_score is not None and np.isfinite(float(ear_score))
        else 0.0
    )

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


if __name__ == "__main__":
    print("[+] Starting Culling Engine...")
    eye_checker = ONNXEyeChecker()
    test_image_path = os.path.join(PROJECT_ROOT, "test_images", "test.jpg")
    if os.path.exists(test_image_path):
        blur_score = calculate_blur_score(test_image_path)
        eyes_open, reason, ear_score = eye_checker.check_eyes_open(test_image_path)
        print(f"[+] Test Image: {test_image_path}")
        print(f"    Blur Score: {blur_score:.2f}")
        print(f"    Eyes Open: {eyes_open}, Reason: {reason}, EAR Score: {ear_score}")
    else:
        print(f"[!] Test image not found at {test_image_path}. Skipping test.")
