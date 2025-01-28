import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from text_to_video.embedding import FaissSearch
from utils.translator import DeepGoogleTranslator

def plot_similarity_graph(video_name, query, timestamps, similarities, gt_start=None, gt_end=None, db_path=None):
    """유사도 그래프 생성 및 저장"""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, similarities, marker='o')
    
    # 정답 구간이 있으면 하이라이트
    if gt_start is not None and gt_end is not None:
        plt.axvspan(gt_start, gt_end, color='red', alpha=0.3, label='Ground Truth')
    
    plt.title(f'Similarity Graph for "{video_name}"')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Similarity Score')
    plt.grid(True)
    
    # 파일 이름에 쿼리 포함하여 저장
    output_dir = f"output/similarity_graphs/{db_path.split('/')[-1].split('.')[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 이름으로 사용할 수 있게 쿼리 문자열 처리
    safe_query = "".join(x for x in query if x.isalnum() or x in [' ', '_'])[:50]  # 길이 제한
    safe_query = safe_query.replace(' ', '_')
    
    plt.savefig(os.path.join(output_dir, f"{video_name}__{safe_query}.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

def analyze_video_similarities(excel_path, db_path):
    """비디오별 유사도 분석 및 그래프 생성"""
    # Excel 파일 읽기
    df = pd.read_excel(excel_path)
    
    # DB 파일 읽기
    with open(db_path, 'r', encoding='utf-8') as f:
        db_data = json.load(f)
    
    # FAISS 검색 초기화
    translator = DeepGoogleTranslator()
    faiss_search = FaissSearch(json_path=db_path)
    
    # 각 비디오에 대해 처리
    for _, row in df.iterrows():
        video_name = row['MatchedName']
        query = row['Query1']
        gt_start = row['StartTime']
        gt_end = row['EndTime']
        
        # DB에서 해당 비디오의 모든 클립 찾기
        video_clips = []
        timestamps = []
        similarities = []
        
        for clip in db_data:
            if video_name in clip['video_path']:
                video_clips.append(clip)
                
        # 시간순 정렬
        video_clips.sort(key=lambda x: x['start_time'])
        
        # 각 클립에 대한 유사도 계산
        for clip in video_clips:
            similarity = faiss_search.compute_similarity(query, clip['caption'], translator)
            timestamps.append(clip['start_time'])
            similarities.append(similarity)
        
        if timestamps:  # 데이터가 있는 경우에만 그래프 생성
            plot_similarity_graph(
                video_name=video_name,
                query=query,
                timestamps=timestamps,
                similarities=similarities,
                gt_start=gt_start,
                gt_end=gt_end,
                db_path=db_path
            )
            print(f"✅ 그래프 생성 완료: {video_name}")

def main():
    # 파일 경로 설정
    excel_path = "evaluation_dataset_jhuni_test_with_matches.xlsx"
    db_path = "output/text2video/test_db_d1_t2v_captions.json"
    
    # 분석 실행
    analyze_video_similarities(excel_path, db_path)
    print("\n✅ 모든 그래프 생성 완료")

if __name__ == "__main__":
    main()