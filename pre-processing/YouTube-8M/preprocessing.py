import os
import cv2
import json
import logging
import shutil
import csv
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from scenedetect.scene_detector import FlashFilter
from tqdm import tqdm  # 진행 상황 표시를 위한 라이브러리

logging.basicConfig(filename="process.log", level=logging.INFO)

def load_json(json_path):
    """JSON 파일을 로드"""
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(annotation_path, file_name, scene_data):
    """Save scene data as a single JSON file."""
    os.makedirs(annotation_path, exist_ok=True)
    json_path = os.path.join(annotation_path, f"{file_name}.json")
    with open(json_path, "w") as json_file:
        json.dump(scene_data, json_file, indent=4)

def create_clip_videos(input_video_path, output_dir_path):
    """Create clip videos and return scene list."""
    os.makedirs(output_dir_path, exist_ok=True)

    # Get FPS
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        logging.error(f"Failed to retrieve FPS for {input_video_path}")
        return [], 0
    cap.release()

    min_scene_len_frames = int(1 * fps)

    scene_list = detect(input_video_path, ContentDetector(min_scene_len=min_scene_len_frames))
    return scene_list, fps

import subprocess
'''
        cmd = [
            "ffmpeg",
            "-i", input_video,  # Input file
            "-ss", str(start.get_seconds()),  # 정확한 시작 시간 설정
            "-to", str(end.get_seconds()),  # 종료 시간 설정
            '-c:v', 'mpeg4',  # Use mpeg4 codec for video
            '-c:a', 'aac',    # Use aac codec for audio
            '-strict', 'experimental',
            '-b:a', '192k',
            '-y',
            output_file         
        ]
'''
def split_video_ffmpeg(input_video, scene_list, output_dir):
    if not scene_list:
        print("No scenes detected, skipping video processing.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, (start, end) in enumerate(scene_list):
        output_file = os.path.join(output_dir, f"scene_{i+1:03}.mp4")
        cmd = [
            "ffmpeg",
            "-ss", str(start.get_seconds()),  # 시작 시간
            "-i", input_video,
            "-t", str(end.get_seconds() - start.get_seconds()),  # `-to` 대신 `-t` 사용하여 상대적 길이 설정
            "-c", "copy",
            output_file
        ]


        print(f"Running command: {' '.join(cmd)}")  # Debugging

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")
            break  # Stop if an error occurs
            
if __name__ == "__main__":
    category_name = "Movieclips"
    # feat : 전체 비디오 돌릴꺼면 수정 ㄱㄱ
    num_videos = 1 # 총 1218개

    # feat: 경로 수정
    root_video_path = "/data/ephemeral/home/data/YouTube-8M-video"
    annotation_path = "./data/YouTube-8M-clips-annatations"

    # feat : json 읽고 video_name 에 접근
    raw_json_data = load_json("/data/ephemeral/home/data/YouTube-8M-annatation/Movieclips_annotation.json")
    json_video_name = {item["video_name"]: item for item in raw_json_data}

    video_files = sorted(os.listdir(root_video_path))

    if num_videos != -1:
        if num_videos < -1:
            raise ValueError("--num_videos must be -1 (for all videos) or a positive integer.")
        video_files = video_files[:num_videos]

    scene_number = 1
    scene_data = []

    # tqdm으로 진행 상황 표시
    for video_idx, video_file in enumerate(tqdm(video_files, desc="Processing Videos")):
        video_name = os.path.splitext(video_file)[0]
        logging.info(f"Processing video ID: {video_name}")

        input_video_path = os.path.join(root_video_path, video_file)

        # feat: 경로 수정
        output_video_dir = os.path.join(f"./data/YouTube-8M-clips", video_name)

        # Detect scenes
        scene_list, fps = create_clip_videos(input_video_path, output_video_dir)

        # Split the video into clips
        if scene_list:
            split_video_ffmpeg(input_video_path, scene_list, output_dir=output_video_dir)

            # Process each scene
            for i, (start, end) in enumerate(tqdm(scene_list, desc=f"Processing Scenes for {video_name}", leave=False)):
                # feat: 저장 video 이름 변경 ex) default : scene-001.mp4 -> 00001.mp4
                clip_file_name = f"{scene_number:05}.mp4"
                old_filename = f"scene_{i+1:03}.mp4"
                old_path = os.path.join(output_video_dir, old_filename)
                new_video_path = os.path.join(output_video_dir, clip_file_name)

                # Rename video clip
                if os.path.exists(old_path):
                    shutil.move(old_path, new_video_path)
                    logging.info(f"Renamed {old_path} to {new_video_path}")

                # feat: raw json 접근
                raw_json = json_video_name[f"{video_name}.mp4"]
                title = raw_json["title"]
                url = raw_json["url"]
                video_id = raw_json["video_id"]

                # Append to JSON annotation
                scene_data.append({
                    "video_path": f"{video_name}/{clip_file_name}",
                    "video_id": video_id, # video1, video2로 할 지 영상 제목으로 할 지
                    
                    # feat: 수정 부분
                    # "clip_id": f"{scene_number:05}", # 고유번호로 할 지
                    "title" : title,
                    "url" : url,
            
                    "start_time": round(start.get_seconds(),2), # ms가 존재함 어찌 할까여
                    "end_time": round(end.get_seconds(),2),
                })
                scene_number += 1  # Increment global scene number
        else:
            logging.info(f"No scenes found for video ID: {video_id}")
            continue

    # Save global JSON and CSV
    save_json(annotation_path, f"{category_name}_clips_annotations", scene_data)
    logging.info(f"Saved annotations to {annotation_path}/{category_name}_annotations.json")