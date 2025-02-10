import json
import os
import numpy as np
import faiss
import requests
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}\n")
    return device

# ✅ 설정값
QUERY_JSON = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/contrastive/caption-query.json"  # 입력 JSON (쿼리)
DB_JSON = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/json/DB/annotations/caption_embedding_tf_35_mpnet.json"  # 전체 데이터베이스 JSON (캡션 + 임베딩 포함)
OUTPUT_JSON = "contrastive_caption_query.json"  # 최종 Contrastive 데이터 저장 파일
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 임베딩 모델
TOP_K = 5  # 부정 샘플 개수

# ✅ DeepL API 설정 (API KEY 입력 필수)
DEEPL_API_KEY = ""  # <<<< DeepL API 키를 입력하세요
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

# ✅ DeepL API를 이용한 번역 함수
def translate_text(text, source_lang="KO", target_lang="EN"):
    params = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    response = requests.post(DEEPL_API_URL, data=params)
    if response.status_code == 200:
        return response.json()["translations"][0]["text"]
    else:
        print(f"🚨 번역 실패! 상태 코드: {response.status_code}, 응답: {response.text}")
        return None

# ✅ DB JSON 데이터 로드 (FAISS 검색을 위해 사용)
print("📂 DB JSON 파일 로드 중...")
with open(DB_JSON, "r", encoding="utf-8") as f:
    db_data = json.load(f)

db_captions = [entry["caption"] for entry in db_data]  # 전체 DB 캡션 리스트
db_embeddings = np.array([entry["embedding"] for entry in db_data], dtype=np.float32)  # 캡션 임베딩

# ✅ Sentence Transformer 모델 로드 (GPU 사용)
print("📥 Sentence Embedding 모델 로드 중...")
device = set_cuda(0)
model = SentenceTransformer(MODEL_NAME).to(device)
model.eval()

# ✅ FAISS 인덱스 구축 (GPU 사용)
print("🔍 FAISS 인덱스 GPU 모드로 구축 중...")
dimension = db_embeddings.shape[1]
res = faiss.StandardGpuResources()  # GPU 리소스 초기화
cpu_index = faiss.IndexFlatL2(dimension)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # GPU로 변환
gpu_index.add(db_embeddings)

# ✅ Query JSON 데이터 로드
print("📂 Query JSON 파일 로드 중...")
with open(QUERY_JSON, "r", encoding="utf-8") as f:
    query_data = json.load(f)

contrastive_data = []

# ✅ Contrastive Dataset 생성
print("🔄 Contrastive Learning 데이터 생성 중...")
for i, entry in tqdm(enumerate(query_data), total=len(query_data)):
    query_text = entry["query"]
    positive_caption = entry["caption"]  # Ground Truth

    # ✅ DeepL API를 이용하여 Query 번역
    translated_query = translate_text(query_text)
    if not translated_query:
        print(f"⚠️ 번역 실패 - 원본 사용: {query_text}")
        translated_query = query_text  # 번역 실패 시 원본 사용

    # Query Encoding (GPU에서 수행)
    query_embedding = model.encode([translated_query], convert_to_numpy=True)

    # ✅ FAISS 검색 (DB JSON을 대상으로 수행)
    _, retrieved_indices = gpu_index.search(query_embedding, TOP_K + 1)

    # 부정 샘플 선정 (자기 자신 제외)
    negative_captions = [
        db_captions[idx] for idx in retrieved_indices[0] if db_captions[idx] != positive_caption
    ][:TOP_K]

    # JSON 저장
    contrastive_data.append({
        "query": query_text,
        "translated_query": translated_query,
        "positive": positive_caption,
        "negatives": negative_captions
    })

# ✅ JSON 저장
print("💾 Contrastive 데이터 저장 중...")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(contrastive_data, f, indent=4, ensure_ascii=False)

print(f"✅ Contrastive Learning 데이터셋이 저장되었습니다: {OUTPUT_JSON}")