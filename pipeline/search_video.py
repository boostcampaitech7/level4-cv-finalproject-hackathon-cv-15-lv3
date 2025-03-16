import os
import json
from moviepy import VideoFileClip
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator

def search_and_save_videos(query_text, json_path="output/text2video/t2v_captions.json", top_k=5, save_clips=False):
    """텍스트로 비디오 검색하고 선택적으로 클립 저장"""
    # 출력 디렉토리 설정
    output_dir = "output/search_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if save_clips:
        clips_dir = os.path.join(output_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
    
    # FAISS 검색 초기화
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=json_path)
    
    # 검색 실행
    results = faiss_search.find_similar_captions(query_text, translator, top_k=top_k)
    
    # 결과 저장을 위한 리스트
    saved_results = []
    
    print("\n🔍 검색 결과:")
    for i, (caption_ko, similarity, video_info) in enumerate(results, 1):
        print(f"\n[{i}] 유사도: {similarity:.4f}")
        print(f"캡션: {caption_ko}")
        print(f"비디오: {video_info['video_path']}")
        print(f"시간: {video_info['start_time']:.1f}s - {video_info['end_time']:.1f}s")
        
        # 결과 정보 생성
        result_info = {
            "rank": i,
            "similarity": float(similarity),
            "caption_ko": caption_ko,
            "video_path": video_info['video_path'],
            "start_time": float(video_info['start_time']),
            "end_time": float(video_info['end_time'])
        }
        
        # 클립 저장이 활성화된 경우에만 클립 추출 및 저장
        if save_clips:
            try:
                video = VideoFileClip(video_info['video_path'])
                clip = video.subclipped(video_info['start_time'], video_info['end_time'])
                
                # 클립 파일명 생성
                base_name = os.path.splitext(os.path.basename(video_info['video_path']))[0]
                clip_name = f"{base_name}_{video_info['start_time']:.1f}_{video_info['end_time']:.1f}.mp4"
                clip_path = os.path.join(clips_dir, clip_name)
                
                # 클립 저장
                clip.write_videofile(clip_path, codec='libx264', audio=False)
                result_info["clip_path"] = clip_path
                
                # 리소스 정리
                clip.close()
                video.close()
                
            except Exception as e:
                print(f"⚠️ 클립 저장 중 오류 발생: {str(e)}")
        
        saved_results.append(result_info)
    
    # 검색 결과 JSON 저장
    results_file = os.path.join(output_dir, "search_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "query": query_text,
            "results": saved_results
        }, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ 검색 결과 저장 완료 → {output_dir}")
    if save_clips:
        print(f"✅ 클립 저장 완료 → {clips_dir}")
    return saved_results

def main():
    # 검색 쿼리 설정
    query = "남자 얼굴 위에 거미가 올라가서 남자가 놀라는 장면"
    
    # 검색 및 결과 저장
    search_and_save_videos(
        query_text=query,
        json_path="output/text2video/t2v_captions.json",
        top_k=1,
        save_clips=False  # 클립 저장 비활성화
    )

if __name__ == "__main__":
    main()