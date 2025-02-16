import os
import glob
import yaml

try:
    with open('text2video_input.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"❌ 설정 파일 로드 실패: {str(e)}")

class Config:
    SCRIPT_FOLDER = "/data/ephemeral/home/jaehuni/level4-cv-finalproject-hackathon-cv-15-lv3/final-pipeline/split_process/sub_server" #Sub에 보낼 Script가 있는 폴더
    REMOTE_PATH = "/data/ephemeral/home/split_process" #Sub에서 작업폴더
    VIDEOS_DIR = config['new_videos_dir'] #비디오들이 입력되어있는 폴더
    SPLIT_VIDEOS_DIR='/data/ephemeral/home/split_process/main_split_videos' #비디오들이 쪼개져서 들어갈 폴더
    SSH_KEY_PATH = "/data/ephemeral/home/CH_1.pem" 

    REMOTE_VIDEO_PATH = os.path.join(REMOTE_PATH, "split_process_videos")
    REMOTE_JSON_PATH = os.path.join(REMOTE_PATH, "split_process_json")
    REMOTE_SCRIPT_PATH = os.path.join(REMOTE_PATH, "split_process_script")
    SUB_SCRIPT_FILE = os.path.join(REMOTE_SCRIPT_PATH, "sub_server_run.py")

    FILE_LIST = glob.glob(f"{SCRIPT_FOLDER}/*")
