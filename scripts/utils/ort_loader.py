import os
import platform
import sys

# 1. 核心前置優化：限制執行緒防止底層 C++ 崩潰（不論什麼系統都很建議加上）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def check_in_docker():
    """檢查目前 Python 是否運行於 Docker 容器內部"""
    # 檢查容器內常見的特徵檔案
    return os.path.exists("/.dockerenv") or os.path.exists("/run/secrets/kubernetes.io")


def get_best_onnx_provider():
    """根據當前 OS 與硬體環境，自動回傳最佳的 ONNX 執行加速器與載入提示"""
    current_os = platform.system().lower()  # 'windows', 'linux', 'darwin' (Mac)

    # 情況 A：在 Docker 容器內（不論宿主機是 Mac 還其他系統）
    if check_in_docker():
        return ["CPUExecutionProvider"], "🐳 偵測為 Docker 環境：強制啟用純 CPU 模式"

    # 情況 B：Mac 本機環境 (macOS)
    if current_os == "darwin":
        # 檢查 Mac 的架構（ARM64 代表 M1/M2/M3/M4，Intel 代表 X86_64）
        is_apple_silicon = (platform.machine().lower() == "arm64") or (
            sys.platform == "darwin" and platform.processor() == "arm"
        )

        if is_apple_silicon:
            return [
                "MpsExecutionProvider",
                "CPUExecutionProvider",
            ], "🍏 偵測為 Apple Silicon Mac 本機：啟用 MPS 顯示卡加速"
        else:
            return ["CPUExecutionProvider"], "💻 偵測為 Intel Mac 本機：啟用純 CPU 模式"

    # 情況 C：其他本機環境 (Windows 或 Linux 本機，嘗試探索有無 CUDA 顯示卡)
    # （注意：這需要您本機有正確裝好 NVIDIA 驅動、CUDA 與對應版本的 onnxruntime-gpu）
    return [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ], f"🖥️ 偵測為 {platform.system()} 本機：優先嘗試 CUDA 加速"


# 2. 安全匯入 ONNX Runtime
try:
    import onnxruntime as ort

    print("目前版本:", ort.__version__)
    # 3. 取得當前環境的最佳加速方案
    BEST_PROVIDERS, environment_msg = get_best_onnx_provider()

    # 4. 校正：如果指定了硬體加速（如 CUDA/MPS），但使用者沒裝對應擴充導致系統不支援，則 fallback 回 CPU
    available_providers = ort.get_available_providers()
    final_providers = [p for p in BEST_PROVIDERS if p in available_providers]

    # 如果過濾後為空，防禦性地補上 CPU 模式
    if not final_providers:
        final_providers = ["CPUExecutionProvider"]

    print(f"✅ ONNX Runtime 載入成功 (v{ort.__version__})")
    print(f"ℹ️  {environment_msg}")
    print(f"🚀 最終啟用的加速器清單: {final_providers}")

except ImportError as e:
    print("❌ 核心載入失敗！請確認您的 pip 套件安裝是否正確。")
    print(f"錯誤細節: {e}")
    sys.exit(1)


def load_my_model(model_path):
    # 建立 Session 時直接帶入我們自動判定好的 final_providers
    session = ort.InferenceSession(model_path, providers=final_providers)
    return session
