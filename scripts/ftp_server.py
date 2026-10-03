import os
from ftplib import FTP

from dotenv import load_dotenv
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# Load Environment Variables
load_dotenv()
RAW_EXTENSION = os.getenv("RAW_EXTENSION", ".NEF")

LAPTOP_FTP_PORT = int(os.getenv("LAPTOP_FTP_PORT", 21))
LAPTOP_FTP_USER = os.getenv("LAPTOP_FTP_USER", "camera")
LAPTOP_FTP_PASS = os.getenv("LAPTOP_FTP_PASS", "12345")

CAMERA_IP = os.getenv("CAMERA_IP", "192.168.8.150")
CAMERA_FTP_USER = os.getenv("CAMERA_FTP_USER", "nikon")
CAMERA_FTP_PASS = os.getenv("CAMERA_FTP_PASS", "12345")
CAMERA_RAW_PATH = os.getenv("CAMERA_RAW_PATH", "/Card1/DCIM/100NC_Z8/")


def run_ftp_server(incoming_dir: str, destination_dir: str, on_media_changed=None):
    authorizer = DummyAuthorizer()
    # Grant camera full permissions (Read/Write/Delete/Create) inside incoming folder
    authorizer.add_user(LAPTOP_FTP_USER, LAPTOP_FTP_PASS, incoming_dir, perm="elradfmw")

    class IncomingFTPHandler(FTPHandler):
        def on_file_received(self, file):
            if on_media_changed is not None:
                on_media_changed(file)

        def on_file_removed(self, file):
            if on_media_changed is not None:
                on_media_changed(file)

    handler = IncomingFTPHandler
    handler.authorizer = authorizer

    server = FTPServer(("0.0.0.0", LAPTOP_FTP_PORT), handler)
    print(f"[+] Dedicated FTP Server running on Port {LAPTOP_FTP_PORT}...")
    print(f"[+] Receiving camera uploads to: {os.path.abspath(incoming_dir)}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[-] FTP Server stopped.")


def pull_matching_raw(filename, destination_dir):
    """Logs into the camera's native FTP server to retrieve matching RAW file."""
    base_name, _ = os.path.splitext(filename)
    raw_filename = base_name + RAW_EXTENSION
    local_raw_path = os.path.join(destination_dir, raw_filename)
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


if __name__ == "__main__":
    run_ftp_server()
