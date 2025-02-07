import os
import glob

class Config:
    SCRIPT_FOLDER = "/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/split_process/sub_server"
    REMOTE_PATH = "/data/ephemeral/home"
    VIDEOS_DIR = "/data/ephemeral/home/sample"
    SSH_KEY_PATH = "/data/ephemeral/home/CH_1.pem"

    REMOTE_VIDEO_PATH = os.path.join(REMOTE_PATH, "split_process_videos")
    REMOTE_JSON_PATH = os.path.join(REMOTE_PATH, "split_process_json")
    REMOTE_SCRIPT_PATH = os.path.join(REMOTE_PATH, "split_process_script")
    SUB_SCRIPT_FILE = os.path.join(REMOTE_SCRIPT_PATH, "sub_server.py")

    FILE_LIST = glob.glob(f"{SCRIPT_FOLDER}/*")
