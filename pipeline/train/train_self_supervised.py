import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import random
import faiss
import os

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 3  # Self-Supervised 학습은 조금 더 짧게
LEARNING_RATE = 1e-6  # 작은 학습률
TEMPERATURE = 0.07  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

# Supervised Learning에서 저장된 모델 불러오기
model = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)
model.load_state_dict(torch.load("saved_model/supervised_model.pth"))
model.eval()  # 평가 모드로 로드 후 학습모드 전환

# FAISS 검색 시스템 초기화
faiss_search = FaissSearch("GT-DB.json")

# Self-Supervised Learning Query (실제 사용자 쿼리 활용)
query_list = [
    "남자가 자동차를 타고 이동하는 장면",
    "비가 오는 도심 속 행인들이 걷는 모습",
    "강가에서 노을을 바라보는 커플",
    "도서관에서 학생들이 책을 읽고 있는 모습",
    "산 정상에서 드론이 촬영하는 장면"
]

# Self-Supervised Dataset
class SelfSupervisedDataset(Dataset):
    def __init__(self, query_list, faiss_search):
        self.query_list = query_list
        self.faiss_search = faiss_search

    def __len__(self):
        return len(self.query_list)

    def __getitem__(self, idx):
        query = self.query_list[idx]

        # Top-K 검색 결과 가져오기
        top_k_results = self.faiss_search.find_similar_captions(query)

        # Positive Sample: Top-1
        positive_sample = top_k_results[0][0]

        # Negative Sample: Top-K 중 Top-1을 제외한 나머지
        negative_sample = random.choice([res[0] for res in top_k_results[1:]])

        return query, positive_sample, negative_sample

# Self-Supervised Learning 데이터 로드
dataset = SelfSupervisedDataset(query_list, faiss_search)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 옵티마이저 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Self-Supervised Learning 수행
print("Self-Supervised Learning 시작...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        queries, positive_samples, negative_samples = zip(*batch)

        query_embeddings = model.encode(queries, convert_to_tensor=True).to(DEVICE)
        positive_embeddings = model.encode(positive_samples, convert_to_tensor=True).to(DEVICE)
        negative_embeddings = model.encode(negative_samples, convert_to_tensor=True).to(DEVICE)

        loss = contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Self-Supervised Learning Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

print("Self-Supervised Learning 완료!")