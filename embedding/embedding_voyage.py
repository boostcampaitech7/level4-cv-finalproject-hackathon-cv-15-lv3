import json
import cohere
import numpy as np
import os
import time

# Cohere API 키 설정 (환경 변수에서 가져오기)
cohere_key = os.getenv("COHERE_API_KEY", "EmLgshjThMnpOpl14HlMpY4eiwWLhRLtxUxbws8x")  # API 키 입력
co = cohere.Client(cohere_key)

# JSON 파일 불러오기
file_path = "../json/DB_v1.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# caption 내용 추출
captions = [entry["caption"] for entry in data]

# 최적 배치 크기 설정 (안전한 범위에서)
BATCH_SIZE = 100  # 적절한 배치 크기로 증가

# 결과 저장할 리스트
all_embeddings = []

# 진행도 출력 함수
def print_progress(processed, total):
    progress = (processed / total) * 100
    print(f"\rProgress: [{processed}/{total}] {progress:.2f}%", end="", flush=True)

# 배치 처리
total_captions = len(captions)
for i in range(0, len(captions), BATCH_SIZE):
    batch = captions[i:i + BATCH_SIZE]
    try:
        result = co.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")
        embeddings = np.asarray(result.embeddings)
        all_embeddings.extend(embeddings)
        time.sleep(5)  # API 제한 초과 방지를 위해 3초 대기
    except cohere.errors.RateLimitError:
        print("\n🚨 API Rate Limit Exceeded! 10초 대기 후 재시도...")
        time.sleep(10)  # API Rate Limit 초과 시 10초 대기 후 재시도
    
    # 진행도 업데이트
    print_progress(len(all_embeddings), total_captions)

print("\n✅ Embedding process completed!")

# JSON에 embedding 추가
for entry, embedding in zip(data, all_embeddings):
    entry["embedding"] = embedding.tolist()

# JSON 저장
output_file = "caption_embedding_voyage.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"✅ Updated JSON saved to {output_file}")