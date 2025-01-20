import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 1. JSON 파일 로드
with open("/data/ephemeral/home/embedding/embedding.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔹 2. caption과 embedding 리스트 추출
captions = [entry["caption"] for entry in data]
embeddings = np.array([entry["embedding"] for entry in data])  # NumPy 배열 변환

# 🔹 3. 임베딩 모델 로드 (sentence-transformers 사용)
model = SentenceTransformer("all-MiniLM-L6-v2")

def find_similar_caption(input_text, top_k=1):
    # 🔹 4. 입력 텍스트 임베딩 변환
    query_embedding = model.encode([input_text])

    # 🔹 5. Cosine Similarity 계산
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # 🔹 6. 가장 유사한 caption 찾기 (Top-k)
    top_indices = similarities.argsort()[::-1][:top_k]  # 유사도가 높은 순 정렬
    results = [(captions[i], similarities[i]) for i in top_indices]

    return results

# 🔹 7. 테스트 실행
query_text = "사람들이 책상에 앉아있습니다."
similar_captions = find_similar_caption(query_text, top_k=3)

# 🔹 8. 결과 출력
for i, (caption, score) in enumerate(similar_captions):
    print(f"🔹 유사도 {i+1}: {score:.4f}")
    print(f"   Caption: {caption}\n")
