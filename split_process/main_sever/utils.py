import os
import subprocess
import math
from typing import List, Dict
from config import Config
from server_info import ServerInfo


def execute_command(cmd: List[str], error_message: str) -> bool:
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{error_message}: {e}")
        return False


def scp_transfer(source_path: str, server: ServerInfo) -> bool:
    """SCP를 사용하여 파일을 서버로 전송"""
    cmd = ['scp', '-i', Config.SSH_KEY_PATH, '-P', str(server.port), source_path,
           f'{server.username}@{server.ip}:{Config.REMOTE_VIDEO_PATH}']
    return execute_command(cmd, f"전송 실패: {source_path} -> {server.ip}")


def create_remote_directory(server: ServerInfo) -> bool:
    """원격 서버에 필요한 디렉토리를 생성"""
    directories = [Config.REMOTE_VIDEO_PATH, Config.REMOTE_JSON_PATH, Config.REMOTE_SCRIPT_PATH]
    return all(execute_command(
        ['ssh', '-i', Config.SSH_KEY_PATH, '-p', str(server.port),
         f'{server.username}@{server.ip}', f'mkdir -p {dir}'],
        f"원격 디렉토리 생성 실패: {server.ip}") for dir in directories)


def run_scene_splitter(server: ServerInfo) -> bool:
    """서버에서 스크립트 실행"""
    scp_script_cmd = ['scp', '-i', Config.SSH_KEY_PATH, '-P', str(server.port)] + Config.FILE_LIST + \
                     [f'{server.username}@{server.ip}:{Config.REMOTE_SCRIPT_PATH}']
    if not execute_command(scp_script_cmd, f"스크립트 전송 실패: {server.ip}"):
        return False

    run_script_cmd = ['ssh', '-i', Config.SSH_KEY_PATH, '-p', str(server.port),
                      f'{server.username}@{server.ip}', f'/opt/conda/bin/python {Config.SUB_SCRIPT_FILE}']
    return execute_command(run_script_cmd, f"scene_splitter 실행 실패: {server.ip}")


def get_video_files(videos_dir: str) -> List[str]:
    """비디오 파일 리스트 가져오기"""
    return [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]


def distribute_files(files: List[str], num_servers: int) -> Dict[int, List[str]]:
    """파일을 서버 개수에 맞게 분배"""
    files_per_server = math.ceil(len(files) / num_servers)
    return {i: files[i * files_per_server: (i + 1) * files_per_server] for i in range(num_servers)}
