import argparse
import json
import os
import sys
import time
from tqdm import tqdm
from moviepy import VideoFileClip
from utils.translator import DeepLTranslator, DeepGoogleTranslator
from video_to_text.video_captioning import MPLUGVideoCaptioningPipeline, TarsierVideoCaptioningPipeline
from text_to_video.embedding import FaissSearch

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

def text_to_video_search(query_text, model_type="mplug", new_videos_dir=None):
    """텍스트로 비디오 검색하는 파이프라인"""
    print("\n🚀 텍스트-비디오 검색 파이프라인 시작...")
    start_time = time.time()
    
    # 설정 값 로드
    print("⚙️ 설정 로드 중...")
    KEEP_CLIPS = False
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))

    # 메타데이터 로드
    print("📂 메타데이터 로드 중...")
    with open('../videos/sample.json', 'r', encoding='utf-8') as f:
        video_metadata = {item['video_name']: item for item in json.load(f)}

    # 새로운 비디오가 있는 경우 처리
    if new_videos_dir and os.path.exists(new_videos_dir):
        print(f"\n🎥 새로운 비디오 처리 중... ({new_videos_dir})")
        
        # VideoCaptioningPipeline 초기화
        print(f"🔧 {model_type.upper()} 모델 초기화 중...")
        model_init_time = time.time()
        
        if model_type == "mplug":
            pipeline = MPLUGVideoCaptioningPipeline(
                keep_clips=KEEP_CLIPS,
                segmentation_method="fixed",
                segmentation_params={"segment_duration": 5},
                mode="text2video",
                video_metadata=video_metadata
            )
        else:
            pipeline = TarsierVideoCaptioningPipeline(
                model_path=model_path,
                keep_clips=KEEP_CLIPS,
                segmentation_method="fixed",
                segmentation_params={"segment_duration": 5},
                mode="text2video",
                video_metadata=video_metadata
            )
        
        print(f"⏱️ 모델 초기화 완료 ({time.time() - model_init_time:.1f}초)")
        
        # 새 비디오 처리
        print("\n🎬 새 비디오 처리 중...")
        process_time = time.time()
        new_results = pipeline.process_directory(new_videos_dir)
        
        if new_results:
            # 임베딩 모델 초기화
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # 새 결과에 임베딩 추가
            print("📊 새 캡션 임베딩 생성 중...")
            for item in new_results:
                caption_en = item['caption']
                embedding = embedding_model.encode([caption_en])[0]
                item['embedding'] = embedding.tolist()  # numpy array를 list로 변환
        
        # 새 결과를 별도 파일로 저장
        new_db_path = "output/text2video/new_videos_captions.json"
        if os.path.exists(new_db_path):
            # 기존 새 비디오 DB에 추가
            with open(new_db_path, 'r', encoding='utf-8') as f:
                existing_new_db = json.load(f)
            existing_new_db.extend(new_results)
            new_results = existing_new_db
            
        with open(new_db_path, 'w', encoding='utf-8') as f:
            json.dump(new_results, f, indent=4, ensure_ascii=False)
                
        print(f"⏱️ 새 비디오 처리 완료 ({time.time() - process_time:.1f}초)")
    
    # FAISS 검색 (두 DB 모두 사용)
    print("\n🔍 FAISS 검색 시스템 초기화 중...")
    search_time = time.time()
    translator = DeepGoogleTranslator()
    
    # 기존 DB와 새 DB를 모두 로드
    main_db_path = "/data/ephemeral/home/jaehuni/json_DB_v2/caption_embedding_tf.json"
    new_db_path = "output/text2video/new_videos_captions.json"
    
    combined_data = []
    
    # 기존 DB 로드
    with open(main_db_path, 'r', encoding='utf-8') as f:
        combined_data.extend(json.load(f))
    
    # 새 DB가 있으면 로드
    if os.path.exists(new_db_path):
        with open(new_db_path, 'r', encoding='utf-8') as f:
            combined_data.extend(json.load(f))
    
    # 임시 통합 DB 파일 생성
    temp_db_path = "output/text2video/temp_combined_db.json"
    with open(temp_db_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    
    # 통합 DB로 검색
    faiss_search = FaissSearch(json_path=temp_db_path)
    
    print(f"🔎 검색어: '{query_text}'")
    print(f"🔎 검색어 번역: '{translator.translate_ko_to_en(query_text)}'")
    similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=5)
    print(f"⏱️ 검색 완료 ({time.time() - search_time:.1f}초)")
    
    # 임시 파일 삭제
    os.remove(temp_db_path)
    
    # 결과 출력 및 클립 저장
    search_results_dir = "output/text2video/search_results"
    
    for i, (caption, similarity, video_info) in enumerate(similar_captions):
        print(f"\n🎯 검색 결과 {i+1}")
        print(f"📊 유사도: {similarity:.4f}")
        print(f"🎬 비디오: {os.path.basename(video_info['video_path'])}")
        print(f"⏰ 구간: {video_info['start_time']}초 ~ {video_info['end_time']}초")
        print(f"📝 제목: {video_info['title']}")
        print(f"📝 캡션: {caption}")
        
        # clip_name = f"search_result_{i+1}_{os.path.basename(video_info['video_path']).split('.')[0]}"
        # saved_path = save_search_result_clip(
        #     video_info['video_path'],
        #     video_info['start_time'],
        #     video_info['end_time'],
        #     search_results_dir,
        #     clip_name
        # )
    
    total_time = time.time() - start_time
    print(f"\n✨ 전체 처리 완료 (총 {total_time:.1f}초)")

def video_to_text_process(model_type="mplug"):
    """비디오를 텍스트로 변환하는 파이프라인"""
    print("\n🚀 비디오-텍스트 변환 파이프라인 시작...")
    start_time = time.time()
    
    # 설정 값
    print("⚙️ 설정 로드 중...")
    VIDEOS_DIR = "../videos"
    INPUT_JSON = "video_to_text/input_table.json"
    KEEP_CLIPS = False
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))

    # 메타데이터 로드
    print("📂 메타데이터 로드 중...")
    load_time = time.time()
    with open('../videos/sample.json', 'r', encoding='utf-8') as f:
        video_metadata = {item['video_name']: item for item in json.load(f)}
    print(f"⏱️ 메타데이터 로드 완료 ({time.time() - load_time:.1f}초)")

    # 파이프라인 초기화
    print(f"🔧 {model_type.upper()} 모델 초기화 중...")
    model_init_time = time.time()
    
    if model_type == "mplug":
        pipeline = MPLUGVideoCaptioningPipeline(
            keep_clips=KEEP_CLIPS,
            mode="video2text",
            video_metadata=video_metadata
        )
    else:  # tarsier
        pipeline = TarsierVideoCaptioningPipeline(
            model_path=model_path,
            keep_clips=KEEP_CLIPS,
            mode="video2text",
            video_metadata=video_metadata
        )
    print(f"⏱️ 모델 초기화 완료 ({time.time() - model_init_time:.1f}초)")
    
    # JSON 파일에서 세그먼트 정보 로드
    print("\n📂 입력 JSON 로드 중...")
    json_load_time = time.time()
    with open(INPUT_JSON, 'r') as f:
        input_data = json.load(f)
    print(f"⏱️ JSON 로드 완료 ({time.time() - json_load_time:.1f}초)")
    
    # 비디오 처리
    print("\n🎬 비디오 목록 생성 중...")
    video_list = []
    for video_data in tqdm(input_data['videos'], desc="비디오 처리 준비"):
        video_path = os.path.join(VIDEOS_DIR, f"{video_data['video_name']}.mp4")
        if not os.path.exists(video_path):
            print(f"⚠️ 비디오를 찾을 수 없음: {video_data['video_name']}")
            continue
        
        video_list.extend([
            (video_path, seg['start'], seg['end'])
            for seg in video_data['segments']
        ])
    
    if not video_list:
        print("❌ 처리할 비디오가 없습니다.")
        return
    
    # 비디오 처리 및 캡션 생성
    print(f"\n🎥 총 {len(video_list)}개 세그먼트 처리 중...")
    process_time = time.time()
    results = pipeline.process_videos(video_list)
    print(f"⏱️ 비디오 처리 완료 ({time.time() - process_time:.1f}초)")
    
    # 결과 저장
    print("\n💾 결과 저장 중...")
    save_time = time.time()
    pipeline.save_results(results)
    print(f"⏱️ 결과 저장 완료 ({time.time() - save_time:.1f}초)")
    
    total_time = time.time() - start_time
    print(f"\n✨ 전체 처리 완료 (총 {total_time:.1f}초)")
    print(f"📊 처리된 세그먼트: {len(results)}/{len(video_list)}")

def main():
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('mode', choices=['text2video', 'video2text'],
                      help='Choose pipeline mode: text2video or video2text')
    parser.add_argument('--model', choices=['mplug', 'tarsier'], default='tarsier',
                      help='Choose model type: mplug or tarsier (default: tarsier)')
    parser.add_argument('--new-videos', type=str, default=None,
                      help='Path to directory containing new videos to process')
    
    args = parser.parse_args()
    
    if args.mode == 'text2video':
        query_text = "한 남자가 멈추라고 외치는 장면"
        text_to_video_search(query_text, model_type=args.model, new_videos_dir=args.new_videos)
    else:
        video_to_text_process(model_type=args.model)

if __name__ == "__main__":
    main()