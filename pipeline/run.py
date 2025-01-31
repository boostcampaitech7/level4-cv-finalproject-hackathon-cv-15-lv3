import argparse
import json
import os
import sys
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

def text_to_video_search(query_text, model_type="mplug"):
    """텍스트로 비디오 검색하는 파이프라인"""
    # 설정 값
    VIDEOS_DIR = "../videos"
    KEEP_CLIPS = True
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))

    # 메타데이터 로드
    with open('../videos/sample.json', 'r') as f:
        video_metadata = {item['video_name']: item for item in json.load(f)}

    # 세그멘테이션 설정 (내부에서 변경 가능)
    segmentation_method = "fixed"  # "fixed", "scene", "shot" 중 선택
    segmentation_params = {"segment_duration": 5}

    # 1. VideoCaptioningPipeline으로 전체 비디오 처리
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
            model_path=model_path,
            keep_clips=KEEP_CLIPS,
            segmentation_method=segmentation_method,
            segmentation_params=segmentation_params,
            mode="text2video",
            video_metadata=video_metadata
        )
    
    # 전체 비디오 디렉토리 처리
    results = pipeline.process_directory(VIDEOS_DIR)
    if results:
        pipeline.save_results(results)
    
    # 2. FAISS 검색 시스템 및 번역기 초기화
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=f"output/text2video/t2v_captions.json")
    
    # 검색 실행
    similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=1)
    
    # 결과 출력 및 클립 저장
    search_results_dir = "output/text2video/search_results"
    
    for i, (caption, similarity, video_info) in enumerate(similar_captions):
        print(f"\n🎯 검색 결과 {i+1}")
        print(f"📊 유사도: {similarity:.4f}")
        print(f"🎬 비디오: {os.path.basename(video_info['video_path'])}")
        print(f"⏰ 구간: {video_info['start_time']}초 ~ {video_info['end_time']}초")
        print(f"📝 제목: {video_info['title']}")
        print(f"📝 캡션: {caption}")
        
        # 검색 결과 클립 저장
        clip_name = f"search_result_{i+1}_{os.path.basename(video_info['video_path']).split('.')[0]}"
        saved_path = save_search_result_clip(
            video_info['video_path'],
            video_info['start_time'],
            video_info['end_time'],
            search_results_dir,
            clip_name
        )

def video_to_text_process(model_type="mplug"):
    """비디오를 텍스트로 변환하는 파이프라인"""
    # 설정 값
    VIDEOS_DIR = "../videos"
    INPUT_JSON = "video_to_text/input_table.json"
    KEEP_CLIPS = False
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))

    # 메타데이터 로드
    with open('../videos/sample.json', 'r', encoding='utf-8') as f:
        video_metadata = {item['video_name']: item for item in json.load(f)}

    # 파이프라인 초기화 (모델 선택)
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
    
    # JSON 파일에서 세그먼트 정보 로드
    with open(INPUT_JSON, 'r') as f:
        input_data = json.load(f)
    
    # 비디오 처리
    video_list = []
    for video_data in input_data['videos']:
        video_path = os.path.join(VIDEOS_DIR, f"{video_data['video_name']}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_data['video_name']}")
            continue
        
        video_list.extend([
            (video_path, seg['start'], seg['end'])
            for seg in video_data['segments']
        ])
    
    if not video_list:
        print("Error: No valid videos found to process")
        return
    
    # 비디오 처리 및 결과 저장
    results = pipeline.process_videos(video_list)
    pipeline.save_results(results)

def main():
    parser = argparse.ArgumentParser(description='Video Processing Pipeline')
    parser.add_argument('mode', choices=['text2video', 'video2text'],
                      help='Choose pipeline mode: text2video or video2text')
    parser.add_argument('--model', choices=['mplug', 'tarsier'], default='tarsier',
                      help='Choose model type: mplug or tarsier (default: tarsier)')
    
    args = parser.parse_args()
    
    if args.mode == 'text2video':
        # 검색할 텍스트 직접 지정
        query_text = "남자 얼굴 위에 거미가 올라가서 남자가 놀라는 장면"
        text_to_video_search(query_text, model_type=args.model)
    else:
        video_to_text_process(model_type=args.model)

if __name__ == "__main__":
    main()