import pandas as pd
import torch
from faiss_search import FaissSearch, DeepLTranslator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')

# 파일 경로 설정
query_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/json/gt_5.json"
db_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/json/caption_embedding_tf_mpnet.json"

# Excel 파일 읽기
df = pd.read_json(query_path)

# FAISS 검색 초기화
translator = DeepLTranslator()
faiss_search = FaissSearch(json_path=db_path, use_gpu=True, model_name="all-mpnet-base-v2")  # CPU 사용

# 평가 지표 초기화
metrics = {
    'total_queries': len(df),
    'found_in_topk': 0,
    'mean_rank': 0,
    'mean_similarity': 0,
    'detailed_results': []
}
top_k = 15
print(f"\n📊 검색 성능 평가 시작 (top-{top_k})")

df = df.iloc[:6]

for _, row in df.iterrows():
    query = row['query']
    video_id = row['video_id']
    gt_start = row['start_time']
    gt_end = row['end_time']
    
    # 쿼리 번역
    query_en = translator.translate_ko_to_en(query)
    print(f"\n🔍 쿼리 평가 중:")
    print(f"   원본: {query}")
    print(f"   번역: {query_en}")
    
    # 전체 DB에서 검색
    results = faiss_search.find_similar_captions(query_en, top_k=top_k)
    
    # 결과 분석
    found = False
    rank = -1
    max_similarity = 0
    
    en_caption_query = []
    
    for i, (caption_en, caption_ko, similarity, video_info) in enumerate(results, 1):
        en_caption_query.append([query_en, caption_en])

    with torch.no_grad():
        inputs = tokenizer(en_caption_query, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # print(scores)
    
    for i in range(len(results)):
        print(results[i][2], scores[i])
        results[i] = list(results[i])
        results[i][2] = scores[i]
        
    results.sort(key=lambda x: x[2], reverse=True)
    
    for i, (caption_en, caption_ko, similarity, video_info) in enumerate(results, 1):
        # video_path에서 video_id 추출
        result_video_id = video_info.get('video_id')
        start_time = float(video_info['start_time'])
        end_time = float(video_info['end_time'])

        gt_start = gt_start.timestamp() if isinstance(gt_start, pd.Timestamp) else float(gt_start)
        gt_end = gt_end.timestamp() if isinstance(gt_end, pd.Timestamp) else float(gt_end)
        
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
    print(f"   가장 유사하다고 판단한 캡션: {results[0][0]}")

# 평균 계산
if metrics['found_in_topk'] > 0:
    metrics['mean_rank'] /= metrics['found_in_topk']
    metrics['mean_similarity'] /= metrics['found_in_topk']

print("\n📈 평가 결과:")
print(f"총 쿼리 수: {metrics['total_queries']}")
print(f"발견된 쿼리 수: {metrics['found_in_topk']}")
print(f"평균 순위: {metrics['mean_rank']:.2f}")
print(f"평균 유사도: {metrics['mean_similarity']:.4f}")
