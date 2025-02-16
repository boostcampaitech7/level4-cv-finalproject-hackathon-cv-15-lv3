import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator

def plot_similarity_graph(video_name, query, timestamps, similarities, gt_start=None, gt_end=None, db_path=None, style="advanced"):
    """유사도 그래프 생성 및 저장"""
    plt.figure(figsize=(12, 6))
    
    if style == "simple":
        plt.plot(timestamps, similarities, marker='o')
        if gt_start is not None and gt_end is not None:
            plt.axvspan(gt_start, gt_end, color='red', alpha=0.3, label='Ground Truth')
    else:
        plt.plot(timestamps, similarities, marker='o', alpha=0.5, linestyle='--', label='Point-wise similarity')
        
        if len(timestamps) > 1:
            segment_duration = timestamps[1] - timestamps[0]
        else:
            segment_duration = 5
        end_times = timestamps[1:] + [timestamps[-1] + segment_duration]
        
        plt.hlines(y=similarities, xmin=timestamps, xmax=end_times, 
                  colors='red', linewidth=2, label='Segment similarity')
        
        if gt_start is not None and gt_end is not None:
            plt.axvspan(gt_start, gt_end, color='green', alpha=0.2, label='Ground Truth')
    
    plt.title(f'Similarity Graph\nVideo: {video_name}\nQuery: "{query}"')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    
    if style == "advanced":
        plt.legend()
    
    output_dir = f"results/similarity_graphs/{db_path.split('/')[-1].split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    safe_query = "".join(x for x in query if x.isalnum() or x in [' ', '_'])[:50]
    safe_query = safe_query.replace(' ', '_')
    
    output_path = os.path.join(output_dir, f"{video_name}__{safe_query}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_video_similarities(excel_path, db_path, graph_style="advanced"):
    """비디오별 유사도 분석 및 그래프 생성"""
    df = pd.read_excel(excel_path)
    
    with open(db_path, 'r', encoding='utf-8') as f:
        db_data = json.load(f)
    
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=db_path)
    
    for _, row in df.iterrows():
        video_url = row['VideoURL']
        video_id = video_url.split('=')[-1]
        query = row['Query']
        # 쿼리 번역
        query_en = translator.translate_ko_to_en(query)
        gt_start = row['StartTime']
        gt_end = row['EndTime']
        
        video_clips = []
        timestamps = []
        similarities = []
        
        for clip in db_data:
            if clip['video_id'] == video_id:
                video_clips.append(clip)
                
        if not video_clips:
            print(f"⚠️ No clips found for video_id: {video_id}")
            continue
                
        video_clips.sort(key=lambda x: float(x['start_time']))
        video_name = video_clips[0]['video_path'].split('/')[0]
        
        for clip in video_clips:
            similarity = faiss_search.compute_similarity(query, clip['caption'], translator)
            timestamps.append(float(clip['start_time']))
            similarities.append(similarity)
        
        if timestamps:
            plot_similarity_graph(
                video_name=video_name,
                query=query_en,  # 영어로 번역된 쿼리 사용
                timestamps=timestamps,
                similarities=similarities,
                gt_start=gt_start,
                gt_end=gt_end,
                db_path=db_path,
                style=graph_style
            )
            print(f"✅ Graph generated: {video_name} (Query: {query} -> {query_en})")

def main():
    # 파일 경로 설정
    excel_path = "csv/evaluation_dataset_v2.xlsx"
    db_path = "output/text2video/test2_db_pyc_t2v_captions.json"
    
    # 그래프 스타일 선택 ("simple" 또는 "advanced")
    graph_style = "simple"  # 또는 "advanced"
    
    # 분석 실행
    analyze_video_similarities(excel_path, db_path, graph_style)
    print("\n✅ 모든 그래프 생성 완료")

if __name__ == "__main__":
    main()