import os
import threading
from .config import Config
from .server_info import SERVERS
from .main_utils import create_remote_directory, distribute_files_round_robin, scp_transfer, run_scene_splitter,split_process_videos,get_video_files


def process_server(server_idx, server, files_to_transfer):
    """각 서버에서 파일 전송 후 스크립트 실행"""
    for file in files_to_transfer:
        if not scp_transfer(os.path.join(Config.SPLIT_VIDEOS_DIR, file), server):
            continue
    run_scene_splitter(server)


def main():
    """메인 실행 함수"""
    os.chmod(Config.SSH_KEY_PATH, 0o600)

    # 원격 서버 디렉토리 생성 (순차 실행)
    for server in SERVERS:
        if not create_remote_directory(server):
            return

    # 비디오 파일 가져오기
    split_process_videos(videos_dir=Config.VIDEOS_DIR, output_dir=Config.SPLIT_VIDEOS_DIR)
    video_files = get_video_files(Config.SPLIT_VIDEOS_DIR)
    if not video_files:
        print("처리할 비디오 파일이 없습니다.")
        return

    # 파일 분배
    distribution = distribute_files_round_robin(video_files, len(SERVERS))
    print(distribution)
    # 서버에 파일 전송 및 스크립트 실행 (병렬 처리)
    threads = []
    for server_idx, server in enumerate(SERVERS, 0):
        files_to_transfer = distribution.get(server_idx, [])
        thread = threading.Thread(target=process_server, args=(server_idx, server, files_to_transfer))
        threads.append(thread)
        thread.start()

    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
