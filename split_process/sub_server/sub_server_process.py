import os
#import time
from config import Config
# 서버의 IP 주소 가져오기
 
# Config.video_dir 여기에 비디오 파일이 들어있음.
# Config.output_file 여기에 json 파일 만들면됨.
def process():
    video_files = os.listdir(Config.video_dir)
    #time.sleep(5)
    # 파일명을 txt 파일로 저장
    with open(Config.output_file, "w") as f:
        for file in video_files:
            f.write(file + "\n")

    print(f"{len(video_files)}개의 파일명을 {Config.output_file}에 저장했습니다.")


