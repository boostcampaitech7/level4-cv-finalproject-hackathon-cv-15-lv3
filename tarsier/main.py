import os
import json
import time
from etc.split import process_videos_from_json  # 비디오 세그먼트 생성 함수
from etc.caption import generate_caption  # 캡션 생성 함수
from utils import load_model_and_processor  # 모델 로딩 함수
import decord  # 비디오 로딩 라이브러리
import ffmpeg  # 비디오 검사 라이브러리
import traceback

# 설정 값
LOG_FILE = "./data/error_log.txt"
JSON_FILE = "/data/ephemeral/home/data/YouTube-8M-annatation/Movieclips_annotation.json"
VIDEO_BASE_PATH = "/data/ephemeral/home/data/YouTube-8M-video"
DESIGNATED_PATH = "./data/videos"  # 클립이 저장될 위치
OUTPUT_JSON_FILE = "./data/video_segments.json"  # JSON이 저장될 위치
MODEL_PATH = "/data/ephemeral/home/Tarsier-7b"

START_VIDEO = 3  # 처리할 비디오 시작 번호
END_VIDEO = 4  # 처리할 비디오 종료 번호
SEGMENT_METHOD = "fixed"  # 비디오 세그먼트 방식
SEGMENT_DURATION = 60  # 세그먼트 길이 (초)

# 모델 및 프로세서 로드
print("📌 모델과 프로세서를 로드하는 중...")
model, processor = load_model_and_processor(MODEL_PATH, max_n_frames=8)

# STEP 1: 비디오 세그먼트 생성
print("🎬 비디오 세그먼트 생성 중...")
video_segments = process_videos_from_json(
    JSON_FILE, VIDEO_BASE_PATH, DESIGNATED_PATH,
    start=START_VIDEO, end=END_VIDEO,
    segment_method=SEGMENT_METHOD, segment_duration=SEGMENT_DURATION
)

# JSON 파일 저장 (비디오 세그먼트 정보)
with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(video_segments, f, indent=4)

print("✅ 비디오 세그먼트 생성 완료!")

# STEP 2: 캡션 생성
print("📝 캡션 생성 중...")
start_time = time.time()

def is_valid_video(video_path):
    """비디오가 정상적으로 재생 가능한지 검사"""
    try:
        probe = ffmpeg.probe(video_path)
        video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
        return len(video_streams) > 0
    except Exception:
        return False

for video in video_segments:
    video_path = os.path.join(DESIGNATED_PATH, video['video_path'])
    
    if not os.path.exists(video_path):
        print(f"❌ 파일이 존재하지 않습니다: {video_path}")
        continue

    if not is_valid_video(video_path):
        print(f"⚠️ 손상된 비디오 파일을 건너뜁니다: {video_path}")
        continue

    try:
        print(f"클립 처리 중: {video_path}")
        clip_start_time = time.time()

        caption = generate_caption(model, processor, video_path)

        clip_end_time = time.time()
        print(f"클립 {video['video_path']} 처리 시간: {clip_end_time - clip_start_time:.2f}초")

        # 생성된 캡션으로 JSON 구조 업데이트
        video['caption'] = caption

    except Exception as e:
        error_message = f"오류가 발생하여 비디오 처리를 건너뜁니다: {video_path}\n"

        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(error_message)

        print(error_message)

# STEP 3: 캡션이 포함된 JSON 파일 저장
with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(video_segments, f, indent=4)

end_time = time.time()
print(f"🚀 총 소요 시간: {end_time - start_time:.2f}초")
print("🎉 캡션 생성 및 JSON 업데이트 완료!")
