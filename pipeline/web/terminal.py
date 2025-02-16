import os
import shutil
from video_captioning import VideoCaptioningPipeline, find_video_file
from embedding import FaissSearch
from moviepy import VideoFileClip

def save_search_result_clip(video_path, start_time, end_time, output_dir, clip_name):
    """검색 결과 클립을 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 비디오에서 클립 추출
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        output_path = os.path.join(output_dir, f"{clip_name}.mp4")
        
        # 클립 저장
        clip.write_videofile(output_path, codec='libx264', audio=False)
        clip.close()
        
        print(f"✅ 검색 결과 클립 저장 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 클립 저장 중 오류 발생: {str(e)}")
        return None

if __name__ == "__main__":
    # 설정 값 직접 입력
    VIDEOS_DIR = "/data/ephemeral/home/jiwan/level4-cv-finalproject-hackathon-cv-15-lv3/pipeline/videos"
    KEEP_CLIPS = False
    SEGMENT_DURATION = 5

    # Initialize pipeline
    # pipeline = VideoCaptioningPipeline(
    #     keep_clips=KEEP_CLIPS,
    #     segment_duration=SEGMENT_DURATION
    # )
    
    # # Process all videos in directory
    # results = pipeline.process_directory(VIDEOS_DIR)
    
    # if results:
    #     # Save results
    #     pipeline.save_results(results)
    # -----------------------------------------------------------------------------------

    # ✅ DeepL API 키 설정
    DEEPL_API_KEY = "dabf2942-070c-47e2-94e1-b43cbef766e3:fx"

    # ✅ FAISS 검색 시스템 및 번역기 초기화
    json_path = "output/embedding.json"
    source_json_path = "output/captions.json"

    translator = DeepLTranslator(api_key=DEEPL_API_KEY)
    faiss_search = FaissSearch(json_path=json_path)

    # ✅ (선택) 새로운 임베딩 생성 필요 시 호출
    faiss_search.generate_and_save_embeddings(source_json_path)

    # ✅ 검색 실행 (한국어 입력)
    query_text = "남자 얼굴 위에 거미가 올라가서 남자가 놀라는 장면"
    similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=1)

    # ✅ 검색 결과 출력 및 클립 저장
    search_results_dir = "output/search_results"
    
    for i, (caption, similarity, video_info) in enumerate(similar_captions):
        print(f"\n🎯 검색 결과 {i+1}")
        print(f"📊 유사도: {similarity:.4f}")
        print(f"🎬 비디오: {os.path.basename(video_info['video_path'])}")
        print(f"⏰ 구간: {video_info['start_time']}초 ~ {video_info['end_time']}초")
        print(f"🎯 클립 ID: {video_info['clip_id']}")
        print(f"📝 캡션: {caption}")
        
        # 검색 결과 클립 저장
        clip_name = f"search_result_{i+1}_{video_info['clip_id']}"
        saved_path = save_search_result_clip(
            video_info['video_path'],
            video_info['start_time'],
            video_info['end_time'],
            search_results_dir,
            clip_name
        )

