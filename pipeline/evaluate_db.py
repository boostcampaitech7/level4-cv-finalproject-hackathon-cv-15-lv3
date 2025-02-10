import pandas as pd
import json
import os
import numpy as np
import time
import argparse
from tqdm import tqdm
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator, DeepLTranslator

def evaluate_metrics(excel_path, db_path, top_k=5):
    """Recall@k와 Median Rank 동시 평가"""
    df = pd.read_excel(excel_path)
    translator = DeepGoogleTranslator()
    # translator = DeepLTranslator()
    faiss_search = FaissSearch(json_path=db_path)
    
    metrics = {
        'total_queries': len(df),
        'recall_at_k': 0,
        'mean_similarity': 0,
        'found_ranks': []  # Median Rank 계산용 (top_k 내 발견된 결과만)
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"메트릭 평가"):
        query = row['Query']
        video_id = row['VideoURL'].split('=')[-1]
        gt_start = row['StartTime']
        gt_end = row['EndTime']
        
        query_en = translator.translate_ko_to_en(query)
        results = faiss_search.find_similar_captions(query, translator, top_k=top_k)
        
        found = False
        rank = None
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
                    # top_k 내에서 발견된 경우에만 순위 기록
                    metrics['found_ranks'].append(rank)
                    metrics['recall_at_k'] += 1
                    metrics['mean_similarity'] += float(max_similarity)
                    break
    
    # Recall@k 계산
    metrics['recall_at_k'] = float(metrics['recall_at_k'] / metrics['total_queries'])
    if metrics['recall_at_k'] > 0:
        metrics['mean_similarity'] = float(metrics['mean_similarity'] / 
                                        (metrics['recall_at_k'] * metrics['total_queries']))
    
    # Median Rank 계산 (top_k 내 발견된 결과만 사용)
    if metrics['found_ranks']:
        metrics['median_rank'] = float(np.median(metrics['found_ranks']))
    else:
        metrics['median_rank'] = float('nan')  # 또는 None
    
    return metrics

def save_summary_results(all_results, top_k, excel_path):
    """종합 결과를 파일로 저장"""
    output_dir = "results/search_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"summary_metrics_{timestamp}.txt"
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("📊 DB 별 성능 비교: " + excel_path + "\n")
        f.write("="*70 + "\n")
        f.write(f"{'DB 이름':20} {'Recall@'+str(top_k):10} {'평균유사도':10} {'MedianRank':12}\n")
        f.write("-"*70 + "\n")
        for db_name, results in all_results.items():
            f.write(f"{db_name:20} {results['recall_at_k']*100:8.2f}% {results['mean_similarity']:10.4f} {results['median_rank']:10.1f}\n")
    
    print(f"\n✨ 종합 결과 저장 완료: {output_path}")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())

def main():
    parser = argparse.ArgumentParser(description='검색 성능 평가')
    parser.add_argument('--top-k', type=int, default=1, 
                       help='Recall@k의 k값 (기본값: 1)')
    args = parser.parse_args()

    excel_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/pipeline/evaluation_dataset_v2.xlsx"
    # excel_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/evaluation/GT.xlsx"
    db_configs = [
        "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/json/DB/annotations/caption_embedding_tf_35_mpnet_new_embeddings.json"
    ]
    
    results = {}
    print(f"\n=== 검색 성능 평가 시작 ===")
    for db_path in tqdm(db_configs, desc="DB 평가 진행률"):
        db_name = os.path.basename(db_path).split('_captions.json')[0]
        metrics = evaluate_metrics(excel_path, db_path, top_k=args.top_k)
        results[db_name] = metrics
    
    save_summary_results(results, args.top_k, excel_path)

if __name__ == "__main__":
    main()