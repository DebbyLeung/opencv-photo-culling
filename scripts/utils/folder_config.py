import enum
import os

from pydantic import BaseModel, model_validator


class FolderNames(enum.Enum):
    FTP_FOLDER = "1_Incoming_FTP"
    TRASH_FOLDER = "2_AI_Trash"
    WATCHED_FOLDER = "3_Lightroom_Watch"
    DESTINATION_FOLDER = "Destination"


class FolderConfig(BaseModel):
    ftp_folder: str
    trash_folder: str
    watched_folder: str
    destination_folder: str

    @model_validator(mode="after")
    def create_directories(self):
        os.makedirs(self.ftp_folder, exist_ok=True)
        os.makedirs(self.trash_folder, exist_ok=True)
        os.makedirs(self.watched_folder, exist_ok=True)
        os.makedirs(self.destination_folder, exist_ok=True)
        return self
