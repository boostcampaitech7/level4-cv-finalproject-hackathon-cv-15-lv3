import os
from config import Config
from server_info import SERVERS
from utils import create_remote_directory, get_video_files, distribute_files, scp_transfer, run_scene_splitter

def main():
    """메인 실행 함수"""
    os.chmod(Config.SSH_KEY_PATH, 0o600)

    # 원격 서버 디렉토리 생성
    for server in SERVERS:
        if not create_remote_directory(server):
            return

    # 비디오 파일 가져오기
    video_files = get_video_files(Config.VIDEOS_DIR)
    if not video_files:
        print("처리할 비디오 파일이 없습니다.")
        return

    # 파일 분배
    distribution = distribute_files(video_files, len(SERVERS) + 1)

    # 서버에 파일 전송 및 스크립트 실행
    for server_idx, server in enumerate(SERVERS, 1):
        files_to_transfer = distribution.get(server_idx, [])
        for file in files_to_transfer:
            if not scp_transfer(os.path.join(Config.VIDEOS_DIR, file), server):
                continue
        run_scene_splitter(server)


if __name__ == "__main__":
    main()
