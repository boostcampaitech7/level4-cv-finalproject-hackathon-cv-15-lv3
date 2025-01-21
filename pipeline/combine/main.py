import json
from video_captioning import VideoCaptioningPipeline, find_video_file
from embedding import FaissSearch, DeepLTranslator


if __name__ == "__main__":
    # 설정 값 직접 입력
    VIDEOS_DIR = "/path/to/videos"
    INPUT_JSON = "/path/to/input.json"
    KEEP_CLIPS = False

    # Initialize pipeline
    pipeline = VideoCaptioningPipeline(keep_clips=KEEP_CLIPS)
    
    # Load segments from JSON
    with open(INPUT_JSON, 'r') as f:
        input_data = json.load(f)
    
    # Create video list from all videos and their segments
    video_list = []
    for video_data in input_data['videos']:
        video_path = find_video_file(VIDEOS_DIR, video_data['video_name'])
        
        # Verify video exists
        if not video_path:
            print(f"Warning: Video not found: {video_data['video_name']}")
            continue
        
        # Add all segments for this video
        video_list.extend([
            (video_path, seg['start'], seg['end'])
            for seg in video_data['segments']
        ])
    
    if not video_list:
        print("Error: No valid videos found to process")
        exit(1)
    
    # Process videos
    results = pipeline.process_videos(video_list)
    
    # Save results
    pipeline.save_results(results)
    # -----------------------------------------------------------------------------------

    # ✅ DeepL API 키 설정
    DEEPL_API_KEY = "dabf2942-070c-47e2-94e1-b43cbef766e3:fx"

    # ✅ FAISS 검색 시스템 및 번역기 초기화
    json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/pipeline/combine/output/embedding.json"
    source_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/pipeline/combine/output/captions.json"

    translator = DeepLTranslator(api_key=DEEPL_API_KEY)
    faiss_search = FaissSearch(json_path=json_path)

    # ✅ (선택) 새로운 임베딩 생성 필요 시 호출
    faiss_search.generate_and_save_embeddings(source_json_path)

    # ✅ 검색 실행 (한국어 입력)
    query_text = "여성이 그룹에서 카메라를 보고 이야기하고 있다."
    similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=1)

    # ✅ 결과 출력
    for i, (caption, similarity) in enumerate(similar_captions):
        print(f"🔹 유사도 {i+1}: {similarity:.4f} (코사인 유사도 기반)")
        print(f"   캡션: {caption}\n")

