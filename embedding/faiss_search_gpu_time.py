import faiss
import json
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# ✅ 1. JSON 파일 로드 & 시간 측정
start_time = time.time()
with open("/data/ephemeral/home/embedding/embedding.json", "r", encoding="utf-8") as f:
    data = json.load(f)
json_load_time = time.time() - start_time
print(f"⏳ JSON 로드 시간: {json_load_time:.4f} 초")

# ✅ 2. 캡션 및 임베딩 추출 & 시간 측정
start_time = time.time()
captions = [entry["caption"] for entry in data]  # 캡션 리스트 추출
embeddings = np.array([entry["embedding"] for entry in data], dtype=np.float32)  # 임베딩을 NumPy 배열로 변환
data_extraction_time = time.time() - start_time
print(f"⏳ 데이터 추출 시간: {data_extraction_time:.4f} 초")

# ✅ 3. 벡터 정규화 (Cosine Similarity 계산을 위한 사전 처리)
start_time = time.time()
faiss.normalize_L2(embeddings)  # L2 정규화를 수행하여 코사인 유사도 기반 검색 가능하도록 변환
normalize_time = time.time() - start_time
print(f"⏳ 임베딩 정규화 시간: {normalize_time:.4f} 초")

# ✅ 4. FAISS-GPU 인덱스 생성
start_time = time.time()
dimension = embeddings.shape[1]  # 임베딩 차원 가져오기
res = faiss.StandardGpuResources()  # GPU 자원 관리 객체 생성
index = faiss.IndexFlatIP(dimension)  # 내적(Inner Product) 기반 유사도 검색을 위한 인덱스 생성
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # CPU 인덱스를 GPU로 이동
faiss_index_creation_time = time.time() - start_time
print(f"⏳ FAISS 인덱스 생성 시간: {faiss_index_creation_time:.4f} 초")

# ✅ 5. FAISS 인덱스에 임베딩 추가
start_time = time.time()
gpu_index.add(embeddings)  # FAISS 인덱스에 임베딩 추가
faiss_index_add_time = time.time() - start_time
print(f"⏳ FAISS 인덱스 추가 시간: {faiss_index_add_time:.4f} 초")

# ✅ 6. 문장 임베딩 모델 로드
start_time = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")  # 사전 훈련된 Sentence Transformer 모델 로드
model_load_time = time.time() - start_time
print(f"⏳ 문장 변환 모델 로드 시간: {model_load_time:.4f} 초")

# ✅ 7. FAISS에서 유사한 문장 찾는 함수 정의
def find_similar_caption_faiss_gpu(input_text, top_k=3):
    # ✅ 7-1. 입력 문장을 임베딩 벡터로 변환 & 시간 측정
    start_time = time.time()
    query_embedding = model.encode([input_text]).astype(np.float32)
    embedding_time = time.time() - start_time
    print(f"⏳ 입력 문장 임베딩 변환 시간: {embedding_time:.4f} 초")

    # ✅ 7-2. 쿼리 임베딩 정규화
    start_time = time.time()
    faiss.normalize_L2(query_embedding)  # 검색을 위해 L2 정규화 적용
    normalize_query_time = time.time() - start_time
    print(f"⏳ 입력 문장 정규화 시간: {normalize_query_time:.4f} 초")

    # ✅ 7-3. FAISS에서 검색 실행
    start_time = time.time()
    D, I = gpu_index.search(query_embedding, top_k)  # 가장 유사한 top_k개 벡터 검색
    faiss_search_time = time.time() - start_time
    print(f"⏳ FAISS 검색 시간: {faiss_search_time:.4f} 초")

    # ✅ 7-4. 검색된 결과(캡션) 반환
    start_time = time.time()
    results = [(captions[i], D[0][idx]) for idx, i in enumerate(I[0])]
    retrieval_time = time.time() - start_time
    print(f"⏳ 결과 조회 시간: {retrieval_time:.4f} 초")

    return results

# ✅ 8. 테스트 검색 실행
query_text = "A woman is talking to the camera in a group."
similar_captions = find_similar_caption_faiss_gpu(query_text, top_k=3)

# ✅ 9. 검색 결과 출력
for i, (caption, similarity) in enumerate(similar_captions):
    print(f"🔹 유사도 {i+1}: {similarity:.4f} (코사인 유사도)")
    print(f"   캡션: {caption}\n")
