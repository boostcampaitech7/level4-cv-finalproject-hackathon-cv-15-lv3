import os
import socket

# 서버의 IP 주소 가져오기
server_ip = socket.gethostbyname(socket.gethostname())

# 고유한 파일명 생성
output_file = f"/data/ephemeral/home/split_process_json/video_files_{server_ip}.txt"

def process():
    # 비디오 파일이 있는 디렉토리 경로
    video_dir = "/data/ephemeral/home/split_process_videos"

    # 디렉토리 내 모든 파일 목록 가져오기
    video_files = os.listdir(video_dir)

    # 파일명을 txt 파일로 저장
    with open(output_file, "w") as f:
        for file in video_files:
            f.write(file + "\n")

    print(f"{len(video_files)}개의 파일명을 {output_file}에 저장했습니다.")


