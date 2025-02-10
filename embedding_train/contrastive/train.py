import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import evaluation
from datasets import Dataset

def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}\n")
    return device

# ✅ 1. 데이터 로드
with open("/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/contrastive/contrastive_caption_query.json", "r", encoding="utf-8") as f:
    contrastive_data = json.load(f)

# ✅ 2. 모델 로드 (기본: all-mpnet-base-v2)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
device = set_cuda(0)
model.to(device)

# ✅ 3. Contrastive Learning 데이터셋 구축
train_samples = []
for entry in contrastive_data:
    query = entry["translated_query"]
    positive = entry["positive"]
    negatives = entry["negatives"]

    # Positive Pair 추가
    train_samples.append(InputExample(texts=[query, positive], label=1.0))  # 1.0 = 유사도가 높아야 함

    # Negative Pairs 추가
    for neg in negatives:
        train_samples.append(InputExample(texts=[query, neg], label=0.5))  # 0.0 = 유사도가 낮아야 함

# ✅ 4. DataLoader 생성
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)

# ✅ 5. Contrastive Loss 정의
train_loss = losses.CosineSimilarityLoss(model)

# ✅ 6. 모델 학습
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=15,
    warmup_steps=100
)

# ✅ 7. 학습된 모델 저장
model.save("fine_tuned_contrastive_model")

print("✅ Fine-Tuning 완료! 모델이 'fine_tuned_contrastive_model'에 저장되었습니다.")