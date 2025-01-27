import argparse
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator

def search_videos(query_text, json_path="output/text2video/t2v_captions.json"):
    """텍스트로 비디오 검색"""
    # FAISS 검색 초기화
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=json_path)
    
    # 검색 실행
    results = faiss_search.find_similar_captions(query_text, translator, top_k=1)
    
    # 결과 출력
    print("\n🔍 검색 결과:")
    for caption_ko, similarity, video_info in results:
        print(f"\n유사도: {similarity:.4f}")
        print(f"캡션: {caption_ko}")
        print(f"비디오: {video_info['video_path']}")
        print(f"시간: {video_info['start_time']:.1f}s - {video_info['end_time']:.1f}s")

def main():
    parser = argparse.ArgumentParser(description='Search Videos by Text')
    parser.add_argument('query', type=str, help='Search query text')
    parser.add_argument('--db', type=str, default="output/text2video/t2v_captions.json",
                      help='Path to video database JSON (default: output/text2video/t2v_captions.json)')
    
    args = parser.parse_args()
    search_videos(args.query, args.db)

if __name__ == "__main__":
    main()