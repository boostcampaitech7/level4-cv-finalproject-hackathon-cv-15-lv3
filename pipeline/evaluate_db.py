import pandas as pd
import json
import os
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator

def evaluate_search_performance(excel_path, db_path, top_k=5):
    """검색 성능 평가"""
    # Excel 파일 읽기
    df = pd.read_excel(excel_path)
    
    # FAISS 검색 초기화
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=db_path)
    
    # 평가 지표
    metrics = {
        'total_queries': len(df),
        'found_in_topk': 0,
        'mean_rank': 0,
        'mean_similarity': 0,
        'detailed_results': []
    }
    
    print(f"\n📊 검색 성능 평가 시작 (top-{top_k})")
    
    for _, row in df.iterrows():
        query = row['Query']
        video_id = row['VideoURL'].split('=')[-1]
        gt_start = row['StartTime']
        gt_end = row['EndTime']
        
        # 쿼리 번역
        query_en = translator.translate_ko_to_en(query)
        print(f"\n🔍 쿼리 평가 중:")
        print(f"   원본: {query}")
        print(f"   번역: {query_en}")
        
        # 전체 DB에서 검색
        results = faiss_search.find_similar_captions(query, translator, top_k=top_k)
        
        # 결과 분석
        found = False
        rank = -1
        max_similarity = 0
        
        for i, (caption_ko, similarity, video_info) in enumerate(results, 1):
            # video_path에서 video_id 추출
            result_video_id = video_info.get('video_id')
            start_time = float(video_info['start_time'])
            end_time = float(video_info['end_time'])
            
            # 정답 비디오이고 시간이 겹치는지 확인
            if result_video_id == video_id:
                time_overlap = (
                    (start_time <= gt_start <= end_time) or
                    (start_time <= gt_end <= end_time) or
                    (gt_start <= start_time <= gt_end)
                )
                if time_overlap:
                    found = True
                    rank = i
                    max_similarity = similarity
                    break
        
        # 결과 저장 - similarity를 float로 변환
        result_info = {
            'query': query,
            'video_id': video_id,
            'found': found,
            'rank': rank,
            'similarity': float(max_similarity),  # float32를 float로 변환
            'gt_start': float(gt_start),         # 혹시 모를 다른 float32 값들도 변환
            'gt_end': float(gt_end)
        }
        metrics['detailed_results'].append(result_info)
        
        # 통계 업데이트 - similarity를 float로 변환
        if found:
            metrics['found_in_topk'] += 1
            metrics['mean_rank'] += rank
            metrics['mean_similarity'] += float(max_similarity)
        
        # 결과 출력
        status = "✅ 발견" if found else "❌ 미발견"
        print(f"{status} (순위: {rank if found else 'N/A'}, 유사도: {max_similarity:.4f})")
    
    # 최종 통계 계산 - 결과를 float로 변환
    if metrics['found_in_topk'] > 0:
        metrics['mean_rank'] = float(metrics['mean_rank'] / metrics['found_in_topk'])
        metrics['mean_similarity'] = float(metrics['mean_similarity'] / metrics['found_in_topk'])
    
    # 결과 출력
    print("\n📊 최종 평가 결과:")
    print(f"총 쿼리 수: {metrics['total_queries']}")
    print(f"Top-{top_k} 내 발견: {metrics['found_in_topk']} ({metrics['found_in_topk']/metrics['total_queries']*100:.1f}%)")
    if metrics['found_in_topk'] > 0:
        print(f"평균 순위: {metrics['mean_rank']:.2f}")
        print(f"평균 유사도: {metrics['mean_similarity']:.4f}")
    
    # 결과 저장
    output_dir = "results/search_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"search_evaluation_top{top_k}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ 평가 결과 저장 완료: {output_path}")
    return metrics

def main():
    excel_path = "csv/evaluation_dataset_v2.xlsx"
    db_configs = [
        "output/text2video/test2_db_d5_t2v_captions.json",
        "output/text2video/test2_db_s_t2v_captions.json",
        "output/text2video/test2_db_pya_t2v_captions.json",
        "output/text2video/test2_db_pyc_t2v_captions.json"
    ]
    
    for db_path in db_configs:
        print(f"\n🎯 DB 평가 중: {db_path}")
        evaluate_search_performance(excel_path, db_path, top_k=10)

if __name__ == "__main__":
    main()