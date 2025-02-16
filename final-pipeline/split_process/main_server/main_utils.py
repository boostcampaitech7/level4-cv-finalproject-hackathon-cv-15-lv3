import os
import subprocess
import math
from typing import List, Dict
from .config import Config
from .server_info import ServerInfo


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

    run_script_cmd = ['ssh','-o StrictHostKeyChecking=no', '-i', Config.SSH_KEY_PATH, '-p', str(server.port),
                      f'{server.username}@{server.ip}', f'/opt/conda/bin/python {Config.SUB_SCRIPT_FILE}']
    return execute_command(run_script_cmd, f"scene_splitter 실행 실패: {server.ip}")


def get_video_files(videos_dir: str) -> List[str]:
    """비디오 파일 리스트 가져오기"""
    return [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]


def distribute_files_round_robin(files: List[str], num_servers: int) -> Dict[int, List[str]]:
    """파일을 서버 개수에 맞게 순환 방식으로 균등 분배"""
    distribution = {i: [] for i in range(num_servers)}
    
    for idx, file in enumerate(files):
        server_idx = idx % num_servers  # 순환 방식으로 배정
        distribution[server_idx].append(file)

    return distribution


####ffmpeg -st
def save_segment(video_path, output_path, start_time, end_time):
    """비디오에서 지정된 구간을 잘라서 저장하는 함수
    
    Args:
        video_path (str): 원본 비디오 파일 경로
        output_path (str): 저장될 클립 파일 경로
        start_time (float): 클립 시작 시간
        end_time (float): 클립 종료 시간
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 폴더가 없으면 생성
    
    command = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(end_time - start_time),
        "-c", "copy",
        output_path,
        "-y"
    ]
    

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def split_video(video_path: str, output_dir: str, segment_duration: int = 5):
    """비디오를 segment_duration 초 단위로 분할하여 저장하는 함수"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 길이 가져오기
    command = [
        "ffprobe", "-i", video_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for start in range(0, int(duration), segment_duration):
        end = min(start + segment_duration, duration)
        output_path = os.path.join(output_dir, f"{video_name}_{start}_{end}.mp4")
        save_segment(video_path, output_path, start, end)


def split_process_videos(videos_dir: str, output_dir: str):
    """디렉토리에 있는 모든 비디오를 분할하여 저장하는 함수"""
    video_files = get_video_files(videos_dir)
    for video in video_files:
        video_path = os.path.join(videos_dir, video)
        split_video(video_path, output_dir)
####ffmpeg -ed


########moviepy
'''
from moviepy.video.io.VideoFileClip import VideoFileClip
#비디오 파일을 7초 단위로 분할하여 저장
def split_video(video_path: str, output_dir: str, segment_duration: int = 7):
    os.makedirs(output_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for start in range(0, int(clip.duration), segment_duration):
        end = min(start + segment_duration, clip.duration)
        subclip = clip.subclipped(start, end)
        output_path = os.path.join(output_dir, f"{video_name}_{start}_{end}.mp4")
        subclip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    clip.close()

def split_process_videos(videos_dir: str):
    video_files = get_video_files(videos_dir)
    for video in video_files:
        video_path = os.path.join(videos_dir, video)
        split_video(video_path, Config.SPLIT_VIDEOS_DIR)
'''