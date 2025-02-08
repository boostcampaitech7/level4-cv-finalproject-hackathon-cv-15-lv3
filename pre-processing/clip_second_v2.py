import json
import os
import re  # 숫자 추출을 위한 정규표현식
import subprocess  # ffmpeg 실행을 위한 모듈
from moviepy import VideoFileClip
from abc import ABC, abstractmethod
from tqdm import tqdm

class VideoSegmenter(ABC):
    """비디오 세그먼테이션을 위한 기본 클래스"""

    @abstractmethod
    def get_segments(self, video_path):
        """비디오 파일을 세그먼트(작은 클립)로 나누는 함수

        Args:
            video_path (str): 비디오 파일 경로

        Returns:
            list of tuple: (시작 시간, 종료 시간) 형태의 세그먼트 리스트
        """
        pass

class FixedDurationSegmenter(VideoSegmenter):
    """고정된 길이로 비디오를 나누는 세그먼터"""

    def __init__(self, segment_duration=5):
        self.segment_duration = segment_duration  # 세그먼트 길이 (초 단위)

    def get_segments(self, video_path):
        """고정된 길이로 비디오를 나누는 메서드"""
        with VideoFileClip(video_path) as video:
            duration = video.duration  # 비디오 전체 길이
            segments = []
            start_time = 0

            while start_time < duration:
                end_time = min(start_time + self.segment_duration, duration)
                if end_time - start_time >= 1:  # 최소 1초 이상인 세그먼트만 추가
                    segments.append((start_time, end_time))
                start_time = end_time

            return segments  # (시작 시간, 종료 시간) 리스트 반환

def create_segmenter(method="fixed", **kwargs):
    """세그먼터 생성 함수

    Args:
        method (str): 세그먼테이션 방법 ("fixed", "scene", "shot")
        **kwargs: 각 세그먼터의 파라미터

    Returns:
        VideoSegmenter: 세그먼터 인스턴스
    """
    segmenters = {
        "fixed": FixedDurationSegmenter  # 현재는 고정된 길이의 세그먼터만 지원
    }

    if method not in segmenters:
        raise ValueError(f"알 수 없는 세그먼테이션 방법: {method}")

    return segmenters[method](**kwargs)

def extract_video_number(video_name):
    """비디오 파일 이름에서 숫자를 추출하는 함수

    Args:
        video_name (str): 비디오 파일 이름 (예: "video_1215.mp4")

    Returns:
        int: 추출된 비디오 번호 (예: 1215), 없으면 None 반환
    """
    match = re.search(r'\d+', video_name)  # 숫자만 추출
    return int(match.group()) if match else None

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
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # print(f"Segment saved successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while saving segment: {e.stderr.decode('utf-8')}")

def process_videos_from_json(json_file, video_base_path, designated_path, start, end, segment_method="fixed", segment_duration=1):
    """JSON 파일에서 지정된 범위(start~end)의 비디오만 처리하여 세그먼트 데이터를 생성하는 함수

    Args:
        json_file (str): 비디오 메타데이터가 포함된 JSON 파일 경로
        video_base_path (str): 원본 비디오가 저장된 폴더 경로
        designated_path (str): 잘라낸 클립을 저장할 폴더 경로
        start (int): 처리할 비디오 시작 번호 (예: 1)
        end (int): 처리할 비디오 종료 번호 (예: 10)
        segment_method (str): 세그먼테이션 방법 ("fixed", "scene", "shot")
        segment_duration (int): 세그먼트 길이 (초 단위, "fixed" 방식일 경우)

    Returns:
        list: 지정된 범위의 비디오 세그먼트 데이터를 포함하는 JSON 리스트
    """
    # JSON 파일을 읽어옴
    with open(json_file, "r", encoding="utf-8") as f:
        video_metadata = json.load(f)

    all_scene_data = []  # 모든 비디오의 세그먼트 데이터를 저장할 리스트

    for video_info in tqdm(video_metadata):
        video_name = video_info["video_name"]
        video_number = extract_video_number(video_name)  # 파일 이름에서 숫자 추출

        # 범위를 벗어난 비디오는 건너뜀
        if video_number is None or not (start <= video_number <= end):
            continue

        video_path = os.path.join(video_base_path, video_name)  # 원본 비디오 파일 경로
        video_save_path = os.path.join(designated_path, f"video_{video_number}")  # 저장할 위치

        if not os.path.exists(video_path):
            print(f"⚠️ 경고: {video_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue  # 파일이 존재하지 않으면 처리하지 않고 넘어감

        # Ensure the directory for saving clips exists
        os.makedirs(video_save_path, exist_ok=True)

        try:
            # 지정된 방법으로 세그먼터 생성
            segmenter = create_segmenter(method=segment_method, segment_duration=segment_duration)
            segments = segmenter.get_segments(video_path)  # 세그먼트 리스트 가져오기

            # 🎯 clip_num을 1부터 시작하는 숫자로 설정
            clip_num = 1

            for start_time, end_time in segments:
                clip_file_name = f"{clip_num:05d}.mp4"  # 1부터 시작하는 숫자 형식 ("00001.mp4")
                output_clip_path = os.path.join(video_save_path, clip_file_name)

                # 비디오 클립을 저장
                save_segment(video_path, output_clip_path, start_time, end_time)

                # JSON 형식으로 저장할 데이터
                scene_data = {
                    "video_path": f"video_{video_number}/{clip_file_name}",
                    "video_id": video_info["video_id"],
                    "title": video_info["title"],
                    "url": video_info["url"],
                    "start_time": f"{start_time:.2f}",
                    "end_time": f"{end_time:.2f}"
                }

                all_scene_data.append(scene_data)  # 리스트에 추가
                clip_num += 1  # clip_num 증가

        except Exception as e:
            print(f"오류가 발생했습니다: {e} - 비디오: {video_path}")
            continue  # 오류 발생 시 다음 비디오로 넘어감

    return all_scene_data  # 모든 비디오 세그먼트 데이터를 반환

# ✅ 예제 실행
json_file = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/json/DB/annotations/Movieclips_annotation.json"  # 비디오 메타데이터가 포함된 JSON 파일 경로
video_base_path = "/hdd1/lim_data/YouTube-8M-video"  # 원본 비디오가 저장된 폴더 경로
designated_path = "/hdd1/lim_data/YouTube-8M-video-3sec_clips"  # 클립이 저장될 폴더 경로
start_video = 896  # 처리할 비디오 시작 번호
end_video = 1218 # 처리할 비디오 종료 번호

# 지정한 범위 내의 비디오를 처리하여 JSON 데이터 생성
output_json = process_videos_from_json(json_file, video_base_path, designated_path, start=start_video, end=end_video, segment_method="fixed", segment_duration=3)

# JSON 데이터를 파일로 저장
with open("./896_1218_3sec.json", "w", encoding="utf-8") as json_file:
    json.dump(output_json, json_file, indent=4)

# JSON 데이터를 출력
# print(json.dumps(output_json, indent=4))
