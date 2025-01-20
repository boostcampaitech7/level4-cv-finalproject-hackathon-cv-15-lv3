import json
from sentence_transformers import SentenceTransformer

# 임베딩 모델 로드 (작고 빠른 모델 선택)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 기존 JSON 데이터 로드
with open("/data/ephemeral/home/embedding/updated_Movieclips_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# caption을 임베딩하고 JSON에 추가
for entry in data:
    caption_text = entry["caption"]
    embedding = model.encode(caption_text).tolist()  # NumPy 배열을 리스트로 변환
    entry["embedding"] = embedding  # JSON에 추가

# 수정된 JSON 저장
with open("/data/ephemeral/home/embedding/embedding.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Embedding 추가 완료! data_with_embeddings.json 파일에 저장됨.")
