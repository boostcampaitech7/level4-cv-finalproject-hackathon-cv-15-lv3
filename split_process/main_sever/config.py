import os
import glob

class Config:
    SCRIPT_FOLDER = "/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/split_process/sub_server" #Sub에 보낼 Script가 있는 폴더
    REMOTE_PATH = "/data/ephemeral/home" #Sub에서 작업폴더
    VIDEOS_DIR = "/data/ephemeral/home/test_sample" #비디오들이 입력되어있는 폴더
    SPLIT_VIDEOS_DIR='/data/ephemeral/home/test_sample/split' #비디오들이 쪼개져서 들어갈 폴더
    SSH_KEY_PATH = "/data/ephemeral/home/CH_1.pem" 

    REMOTE_VIDEO_PATH = os.path.join(REMOTE_PATH, "split_process_videos")
    REMOTE_JSON_PATH = os.path.join(REMOTE_PATH, "split_process_json")
    REMOTE_SCRIPT_PATH = os.path.join(REMOTE_PATH, "split_process_script")
    SUB_SCRIPT_FILE = os.path.join(REMOTE_SCRIPT_PATH, "sub_server.py")

    FILE_LIST = glob.glob(f"{SCRIPT_FOLDER}/*")
