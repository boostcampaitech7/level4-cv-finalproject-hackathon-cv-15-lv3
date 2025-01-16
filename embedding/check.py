import numpy as np
import json
# JSON 파일 다시 불러오기
with open("/data/ephemeral/home/embedding/embedding.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

# 임베딩 필드가 존재하는지 확인
for i, entry in enumerate(loaded_data):
    if "embedding" in entry and isinstance(entry["embedding"], list):
        print(f"✅ Clip {entry['clip_id']} embedding loaded successfully. First 5 values: {entry['embedding'][:5]}")
    else:
        print(f"❌ Embedding missing for Clip {entry['clip_id']}!")

    if i >= 1:  # 처음 2개만 확인
        break


# 모든 임베딩 벡터의 차원 확인
embedding_lengths = [len(entry["embedding"]) for entry in loaded_data]
print("Embedding dimensions:", set(embedding_lengths))  # 모든 임베딩이 같은 크기인지 확인

from sklearn.metrics.pairwise import cosine_similarity

# 두 개의 caption 임베딩 유사도 비교
embedding_1 = np.array(loaded_data[0]["embedding"]).reshape(1, -1)
embedding_2 = np.array(loaded_data[1]["embedding"]).reshape(1, -1)

similarity = cosine_similarity(embedding_1, embedding_2)[0][0]
print(f"Cosine similarity between first two captions: {similarity:.4f}")
