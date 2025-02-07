import os
import json
from config import Config

def process():
    video_files = os.listdir(Config.video_dir)

    # JSON 파일로 저장
    with open(Config.output_file, "w", encoding="utf-8") as f:
        json.dump(video_files, f, indent=4, ensure_ascii=False)

    print(f"{len(video_files)}개의 파일명을 {Config.output_file}에 JSON 형식으로 저장했습니다.")

