import argparse
import os
import time
import json
import logging
import warnings
import moviepy
import sys
from io import StringIO
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from tqdm import tqdm
from video_to_text.video_captioning import MPLUGVideoCaptioningPipeline, TarsierVideoCaptioningPipeline
from text_to_video.embedding import FaissSearch

# 모든 로깅과 경고 억제
logging.getLogger('imageio').setLevel(logging.ERROR)
logging.getLogger('moviepy').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)  # transformers 경고 억제
moviepy.config.VERBOSE = False
warnings.filterwarnings('ignore')  # 모든 경고 억제

def build_video_db(model_type="mplug", segmentation_method="fixed", segmentation_params=None):
    """비디오 DB 구축 파이프라인"""
    print("\n🚀 비디오 DB 구축 파이프라인 시작...")
    start_time = time.time()
    
    # 설정 값
    print("⚙️ 설정 로드 중...")
    VIDEOS_DIR = "/data/ephemeral/home/jaehuni/split_exp/videos"
    KEEP_CLIPS = False
    OUTPUT_DIR = "output/text2video"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    TARSIER_MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))
    
    # 메타데이터 로드
    print("📂 메타데이터 로드 중...")
    load_time = time.time()
    with open('/data/ephemeral/home/jaehuni/split_exp/matched_videos.json', 'r', encoding='utf-8') as f:
        video_metadata = {item['video_name']: item for item in json.load(f)}
    print(f"⏱️ 메타데이터 로드 완료 ({time.time() - load_time:.1f}초)")
    
    # 1. 비디오 캡셔닝
    print(f"\n🔧 {model_type.upper()} 모델 초기화 중...")
    model_init_time = time.time()
    
    if model_type == "mplug":
        pipeline = MPLUGVideoCaptioningPipeline(
            keep_clips=KEEP_CLIPS,
            segmentation_method=segmentation_method,
            segmentation_params=segmentation_params,
            mode="text2video",
            video_metadata=video_metadata
        )
    else:  # tarsier
        pipeline = TarsierVideoCaptioningPipeline(
            model_path=TARSIER_MODEL_PATH,
            keep_clips=KEEP_CLIPS,
            segmentation_method=segmentation_method,
            segmentation_params=segmentation_params,
            mode="text2video",
            video_metadata=video_metadata
        )
    print(f"⏱️ 모델 초기화 완료 ({time.time() - model_init_time:.1f}초)")
    
    # 비디오 처리 및 캡션 생성
    print("\n🎥 비디오 처리 중...")
    process_time = time.time()
    results = pipeline.process_directory(VIDEOS_DIR)
    print(f"⏱️ 비디오 처리 완료 ({time.time() - process_time:.1f}초)")
    
    # 결과 저장
    print("\n💾 캡션 결과 저장 중...")
    save_time = time.time()
    pipeline.save_results(results)
    print(f"⏱️ 캡션 저장 완료 ({time.time() - save_time:.1f}초)")
    
    # 2. 임베딩 생성 및 저장
    print("\n🔍 FAISS 임베딩 생성 중...")
    embedding_time = time.time()
    json_path = os.path.join(OUTPUT_DIR, "t2v_captions.json")
    faiss_search = FaissSearch(json_path=json_path)
    print(f"⏱️ 임베딩 생성 완료 ({time.time() - embedding_time:.1f}초)")
    
    total_time = time.time() - start_time
    print(f"\n✨ 전체 DB 구축 완료 (총 {total_time:.1f}초)")
    print(f"📊 처리된 비디오 수: {len(results)}")
    print(f"📁 결과 저장 경로: {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description='Build Video Database')
    parser.add_argument('--model', choices=['mplug', 'tarsier'], default='tarsier',
                      help='Choose model type: mplug or tarsier (default: tarsier)')
    parser.add_argument('--segmentation', choices=['fixed', 'scene', 'shot'], default='fixed',
                      help='Choose segmentation method (default: fixed)')
    parser.add_argument('--segment-duration', type=float, default=5.0,
                      help='Duration for fixed segmentation (default: 5.0)')
    
    args = parser.parse_args()
    
    # 세그멘테이션 파라미터 설정
    segmentation_params = {}
    if args.segmentation == 'fixed':
        segmentation_params = {"segment_duration": args.segment_duration}
    
    build_video_db(
        model_type=args.model,
        segmentation_method=args.segmentation,
        segmentation_params=segmentation_params
    )

if __name__ == "__main__":
    main()