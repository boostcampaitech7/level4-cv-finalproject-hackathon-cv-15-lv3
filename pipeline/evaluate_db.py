import pandas as pd
import json
import os
import numpy as np
import time
import argparse
from tqdm import tqdm
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator

def evaluate_recall_at_k(excel_path, db_path, top_k=5):
    """Recall@k 평가"""
    df = pd.read_excel(excel_path)
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=db_path)
    
    metrics = {
        'total_queries': len(df),
        'recall_at_k': 0,
        'mean_similarity': 0,
        'detailed_results': []
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Recall@{top_k} 평가"):
        query = row['Query']
        video_id = row['VideoURL'].split('=')[-1]
        gt_start = row['StartTime']
        gt_end = row['EndTime']
        
        query_en = translator.translate_ko_to_en(query)
        results = faiss_search.find_similar_captions(query, translator, top_k=top_k)
        
        found = False
        rank = top_k + 1
        max_similarity = 0
        
        for i, (caption_ko, similarity, video_info) in enumerate(results, 1):
            if video_info.get('video_id') == video_id:
                time_overlap = (
                    (float(video_info['start_time']) <= gt_start <= float(video_info['end_time'])) or
                    (float(video_info['start_time']) <= gt_end <= float(video_info['end_time'])) or
                    (gt_start <= float(video_info['start_time']) <= gt_end)
                )
                if time_overlap:
                    found = True
                    rank = i
                    max_similarity = similarity
                    break
        
        if found and rank <= top_k:
            metrics['recall_at_k'] += 1
            metrics['mean_similarity'] += float(max_similarity)
    
    metrics['recall_at_k'] = float(metrics['recall_at_k'] / metrics['total_queries'])
    if metrics['recall_at_k'] > 0:
        metrics['mean_similarity'] = float(metrics['mean_similarity'] / 
                                        (metrics['recall_at_k'] * metrics['total_queries']))
    
    return metrics

def evaluate_median_rank(excel_path, db_path, batch_size=2048):
    """Median Rank 평가 (FAISS GPU 제한 고려)"""
    df = pd.read_excel(excel_path)
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=db_path)
    
    metrics = {
        'total_queries': len(df),
        'all_ranks': [],
        'detailed_results': []
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Median Rank 평가"):
        query = row['Query']
        video_id = row['VideoURL'].split('=')[-1]
        gt_start = row['StartTime']
        gt_end = row['EndTime']
        
        query_en = translator.translate_ko_to_en(query)
        
        found = False
        rank = len(faiss_search.data)
        current_offset = 0
        
        while current_offset < len(faiss_search.data) and not found:
            batch_k = min(batch_size, len(faiss_search.data) - current_offset)
            results = faiss_search.find_similar_captions(query, translator, top_k=batch_k)
            
            for i, (caption_ko, similarity, video_info) in enumerate(results, 1):
                if video_info.get('video_id') == video_id:
                    time_overlap = (
                        (float(video_info['start_time']) <= gt_start <= float(video_info['end_time'])) or
                        (float(video_info['start_time']) <= gt_end <= float(video_info['end_time'])) or
                        (gt_start <= float(video_info['start_time']) <= gt_end)
                    )
                    if time_overlap:
                        found = True
                        rank = current_offset + i
                        break
            
            current_offset += batch_k
        
        metrics['all_ranks'].append(rank)
    
    metrics['median_rank'] = float(np.median(metrics['all_ranks']))
    return metrics

def save_summary_results(all_results, metric_type, top_k=None):
    """종합 결과를 파일로 저장"""
    output_dir = "results/search_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 설정
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if metric_type == "recall":
        filename = f"summary_recall{top_k}_{timestamp}.txt"
    else:
        filename = f"summary_median_rank_{timestamp}.txt"
    
    output_path = os.path.join(output_dir, filename)
    
    # 결과 포맷팅
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("📊 DB 별 성능 비교:\n")
        f.write("="*50 + "\n")
        
        if metric_type == "recall":
            f.write(f"{'DB 이름':20} {'Recall@'+str(top_k):10} {'평균유사도':10}\n")
            f.write("-"*50 + "\n")
            for db_name, results in all_results.items():
                f.write(f"{db_name:20} {results['recall_at_k']*100:8.2f}% {results['mean_similarity']:10.4f}\n")
        else:
            f.write(f"{'DB 이름':20} {'MedianRank':12}\n")
            f.write("-"*50 + "\n")
            for db_name, results in all_results.items():
                f.write(f"{db_name:20} {results['median_rank']:10.1f}\n")
    
    print(f"\n✨ 종합 결과 저장 완료: {output_path}")
    
    # 콘솔에도 출력
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())

def main():
    parser = argparse.ArgumentParser(description='검색 성능 평가')
    parser.add_argument('--metric', type=str, choices=['recall', 'median', 'both'], 
                       default='both', help='평가할 메트릭 (recall, median, both)')
    parser.add_argument('--top-k', type=int, default=10, 
                       help='Recall@k의 k값 (기본값: 10)')
    args = parser.parse_args()

    excel_path = "csv/evaluation_dataset_v2.xlsx"
    db_configs = [
        "output/text2video/test2_db_d3_t2v_captions.json",
        "output/text2video/test2_db_d5_t2v_captions.json",
        "output/text2video/test2_db_d7_t2v_captions.json",
        "output/text2video/test2_db_s_t2v_captions.json",
        "output/text2video/test2_db_pya_t2v_captions.json",
        "output/text2video/test2_db_pyc_t2v_captions.json"
    ]
    
    # Recall@k 평가
    if args.metric in ['recall', 'both']:
        recall_results = {}
        print(f"\n=== Recall@{args.top_k} 평가 시작 ===")
        for db_path in tqdm(db_configs, desc="DB 평가 진행률"):
            db_name = os.path.basename(db_path).split('_captions.json')[0]
            results = evaluate_recall_at_k(excel_path, db_path, top_k=args.top_k)
            recall_results[db_name] = results
        save_summary_results(recall_results, "recall", args.top_k)
    
    # Median Rank 평가
    if args.metric in ['median', 'both']:
        median_rank_results = {}
        print("\n=== Median Rank 평가 시작 ===")
        for db_path in tqdm(db_configs, desc="DB 평가 진행률"):
            db_name = os.path.basename(db_path).split('_captions.json')[0]
            results = evaluate_median_rank(excel_path, db_path)
            median_rank_results[db_name] = results
        save_summary_results(median_rank_results, "median_rank")

if __name__ == "__main__":
    main()