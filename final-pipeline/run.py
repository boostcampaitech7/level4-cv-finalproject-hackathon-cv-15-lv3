import argparse
import yaml
import json
import os
import sys
import time
from tqdm import tqdm
from moviepy import VideoFileClip
from utils.translator import DeepGoogleTranslator, DeepLTranslator
from video_to_text.video_captioning import TarsierVideoCaptioningPipeline
from text_to_video.embedding import FaissSearch
from split_process.main_server.main_server_run import main as split_process_main
from split_process.main_server.config import Config as SplitConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_search_result_clip(video_path, start_time, end_time, output_dir, clip_name):
    """검색 결과 클립을 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        output_path = os.path.join(output_dir, f"{clip_name}.mp4")
        clip.write_videofile(output_path, codec='libx264', audio=False)
        clip.close()
        
        print(f"✅ 검색 결과 클립 저장 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 클립 저장 중 오류 발생: {str(e)}")
        return None

def video_to_text_process():
    """비디오를 텍스트로 변환하는 파이프라인"""
    print("\n🚀 비디오-텍스트 변환 파이프라인 시작...")
    process_start_time = time.time()
    
    # YAML 설정 파일 로드
    try:
        with open('video2text_input.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {str(e)}")
        return

    # 기본 설정값 (코드로 관리)
    KEEP_CLIPS = True  # 클립 저장을 위해 True로 변경
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))
    clips_dir = os.path.join(current_dir, "clips/video2text/")  # 클립 저장 경로
    
    # clips 디렉토리 생성
    os.makedirs(clips_dir, exist_ok=True)

    # 파이프라인 초기화
    pipeline = TarsierVideoCaptioningPipeline(
        model_path=model_path,
        keep_clips=KEEP_CLIPS,
        mode="video2text",
        video_metadata={},
        clips_dir=clips_dir  # 클립 저장 경로 지정
    )
    
    # 비디오 처리
    video_list = []
    for video_data in config.get('videos', []):
        video_path = video_data['video_id']
        
        if not os.path.exists(video_path):
            print(f"⚠️ 비디오를 찾을 수 없음: {video_path}")
            continue
        
        video_list.extend([
            (video_path, ts['start_time'], ts['end_time'])
            for ts in video_data['timestamps']
        ])
    
    if not video_list:
        print("❌ 처리할 비디오가 없습니다.")
        return
    
    # 비디오 처리 및 캡션 생성
    print(f"\n🎥 비디오 처리 중... (총 {len(video_list)}개 클립)")
    results = []
    for idx, (video_path, start_time, end_time) in enumerate(video_list, 1):
        print(f"\n처리 중: {idx}/{len(video_list)} - {os.path.basename(video_path)} ({start_time}초 ~ {end_time}초)")
        result = pipeline.process_video(video_path, start_time, end_time)
        if result:
            results.append(result)
            print(f"✅ 완료")
    
    # 결과 출력
    print("\n📝 생성된 캡션:")
    print("=" * 80)
    for i, ((original_path, start_time, end_time), result) in enumerate(zip(video_list, results), 1):
        # YouTube-8M 비디오인 경우 매핑 정보 활용
        if 'YouTube_8M/YouTube_8M_video' in original_path:
            video_name = os.path.basename(original_path)  # video_XXX.mp4
            mapping_path = './videos/YouTube_8M/YouTube_8M_annotation/Movieclips_annotation.json'
            
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    video_info = next(
                        (item for item in mapping_data if item['video_name'] == video_name),
                        None
                    )
                    if video_info:
                        video_title = video_info['title']
                        print(f"\n🎬 클립 {i}: {video_title} (ID: {video_name})")
                    else:
                        print(f"\n🎬 클립 {i}: {video_name}")
            except Exception as e:
                print(f"\n🎬 클립 {i}: {video_name}")
        else:
            # 외부 입력 비디오의 경우 파일명만 출력
            video_name = os.path.basename(original_path)
            print(f"\n🎬 클립 {i}: {video_name}")
        
        print(f"⏰ 구간: {result['start_time']}초 ~ {result['end_time']}초")
        print(f"결과: {result['caption_ko']}")
        print("-" * 80)
    
    # 결과 출력 후 시간 계산
    total_time = time.time() - process_start_time
    minutes, seconds = divmod(total_time, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        print(f"\n✨ 전체 처리 완료 (총 {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초)")
    else:
        print(f"\n✨ 전체 처리 완료 (총 {int(minutes)}분 {seconds:.1f}초)")
    
    print(f"📊 처리된 세그먼트: {len(results)}/{len(video_list)}")
    print(f"💾 클립 저장 위치: {clips_dir}")

def text_to_video_search(query_text, new_videos_dir=None):
    """텍스트로 비디오 검색하는 파이프라인"""
    print("\n🚀 텍스트-비디오 검색 파이프라인 시작...")
    start_time = time.time()
    
    # 새로운 비디오가 있는 경우 처리
    if new_videos_dir and os.path.exists(new_videos_dir):
        print(f"\n🎥 새로운 비디오 처리 중... ({new_videos_dir})")
    
        # 설정 업데이트
        SplitConfig.VIDEOS_DIR = new_videos_dir
        SplitConfig.SPLIT_VIDEOS_DIR = os.path.join(new_videos_dir, "split")
        
        # 분산 처리 실행
        print("📦 비디오 분할 및 분산 처리 시작...")
        process_start_time = time.time()
        split_process_main()
        
        # JSON 결과 취합
        print("\n📊 처리 결과 취합 중...")
        json_results = []
        json_dir = "/data/ephemeral/home/json"  # 메인 서버의 JSON 저장 경로
        
        for json_file in os.listdir(json_dir):
            if json_file.startswith("video_files_") and json_file.endswith(".json"):
                with open(os.path.join(json_dir, json_file), 'r') as f:
                    json_results.extend(json.load(f))
        
        # 새 결과를 DB에 추가
        new_db_path = "output/text2video/new_videos_captions.json"
        with open(new_db_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4, ensure_ascii=False)
        
        print(f"⏱️ 새 비디오 처리 완료 ({time.time() - process_start_time:.1f}초)")
    
    # FAISS 검색
    search_time = time.time()
    translator = DeepLTranslator()
    
    # DB 로드 및 통합
    main_db_path = "database/caption_embedding_tf.json"
    new_db_path = "output/text2video/new_videos_captions.json"

    combined_data = []
    with open(main_db_path, 'r', encoding='utf-8') as f:
        combined_data.extend(json.load(f))
    
    if os.path.exists(new_db_path):
        with open(new_db_path, 'r', encoding='utf-8') as f:
            combined_data.extend(json.load(f))
    
    temp_db_path = "output/text2video/temp_combined_db.json"
    with open(temp_db_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

    faiss_search = FaissSearch(json_path=temp_db_path)
    
    print(f"🔎 검색어: '{query_text}'")
    print(f"🔎 검색어 번역: '{translator.translate_ko_to_en(query_text)}'")
    similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=2)
    print(f"⏱️ 검색 완료 ({time.time() - search_time:.1f}초)")
    
    os.remove(temp_db_path)
    
    # 결과 출력
    for i, (caption, similarity, video_info) in enumerate(similar_captions):
        print(f"\n🎯 검색 결과 {i+1}")
        print(f"📊 유사도: {similarity:.4f}")
        print(f"🎬 비디오: {os.path.basename(video_info['video_path'])}")
        print(f"⏰ 구간: {video_info['start_time']}초 ~ {video_info['end_time']}초")
        print(f"📝 제목: {video_info['title']}")
        print(f"📝 캡션: {caption}")
    
    total_time = time.time() - start_time
    print(f"\n✨ 전체 처리 완료 (총 {total_time:.1f}초)")

    return similar_captions

def main():
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('mode', choices=['text2video', 'video2text'],
                      help='Choose pipeline mode: text2video or video2text')
    parser.add_argument('--new-videos', type=str, default=None,
                      help='Path to directory containing new videos to process')
    
    args = parser.parse_args()
    
    if args.mode == 'text2video':
        query_text = "초록색 옷을 입고있는 남자가 멈추라고 하는 장면" # 검색하고 싶은 쿼리 입력
        text_to_video_search(query_text, new_videos_dir=args.new_videos)
    else:
        video_to_text_process()

if __name__ == "__main__":
    main()