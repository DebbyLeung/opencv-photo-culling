import os

from dotenv import load_dotenv
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

# Load Environment Variables
load_dotenv()

LAPTOP_FTP_PORT = int(os.getenv("LAPTOP_FTP_PORT", 21))
LAPTOP_FTP_USER = os.getenv("LAPTOP_FTP_USER", "camera")
LAPTOP_FTP_PASS = os.getenv("LAPTOP_FTP_PASS", "12345")


def run_ftp_server(incoming_dir: str, on_media_changed=None):
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


if __name__ == "__main__":
    run_ftp_server()
