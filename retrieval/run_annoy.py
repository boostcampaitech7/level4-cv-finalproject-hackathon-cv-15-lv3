#!/usr/bin/env python
import time
import argparse
from search.annoy_search import AnnoySearch  # AnnoySearch 클래스가 정의된 파일

def main():
    parser = argparse.ArgumentParser(description="Annoy 기반 비디오 캡션 검색")
    parser.add_argument("--json", type=str, default="json/caption_embedding_tf.json", help="검색할 JSON 데이터 파일 경로")
    parser.add_argument("--top_k", type=int, default=3, help="검색 결과 개수 (top-k)")
    args = parser.parse_args()
    
    searcher = AnnoySearch(json_path=args.json)
    
    count = 0
    sum_time = 0

    while True:
        query = input("Annoy 검색할 텍스트를 입력 (종료하려면 'exit' 입력): ")
        if query.lower() == "exit":
            print("프로그램 종료!")
            break

        start_time = time.perf_counter()
        count += 1
        results = searcher.find_similar_captions(query, top_k=args.top_k)
        end_time = time.perf_counter()
        sum_time += end_time - start_time

        print(f"검색 수행 시간: {end_time - start_time:.6f}초")

        if results:
            for i, (caption, similarity, video_info) in enumerate(results, start=1):
                print(f"\n결과 {i}:")
                print(f"캡션: {caption}")
                print(f"유사도: {similarity:.4f}")
        else:
            print("검색 결과가 없음")       
        
        print("%d번의 결과: 총 %.6f초" % (count, sum_time))

if __name__ == "__main__":
    main()
