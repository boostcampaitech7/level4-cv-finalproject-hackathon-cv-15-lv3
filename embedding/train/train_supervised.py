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
EPOCHS = 5
LEARNING_RATE = 1e-5
TEMPERATURE = 0.07  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

# Sentence Transformer 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

# GT-caption 데이터 로드
with open("GT-DB.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

# FAISS 검색 시스템 초기화
class FaissSearch:
    def __init__(self, json_path, model_name="all-MiniLM-L6-v2", use_gpu=True):
        self.json_path = json_path
        self.model = SentenceTransformer(model_name)

        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        self.captions = [entry["caption"] for entry in self.data]
        self.embeddings = [self.model.encode(entry["caption"]).tolist() for entry in self.data]
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)

        # FAISS 인덱스 생성
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings.numpy())

    def find_similar_captions(self, input_text, top_k=TOP_K):
        query_embedding = self.model.encode([input_text]).astype("float32")
        D, I = self.index.search(query_embedding, top_k)
        return [(self.captions[i], D[0][idx]) for idx, i in enumerate(I[0])]

# Supervised Dataset
class SupervisedDataset(Dataset):
    def __init__(self, gt_data, faiss_search):
        self.data = gt_data
        self.faiss_search = faiss_search

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        gt_caption = entry["caption"]

        # Top-K 검색
        top_k_results = self.faiss_search.find_similar_captions(gt_caption)
        
        positive_sample = top_k_results[0][0]  # Top-1
        negative_sample = random.choice([res[0] for res in top_k_results[1:]])  # Top-K 중 1개

        return gt_caption, positive_sample, negative_sample

# Contrastive Loss
def contrastive_loss(query_emb, pos_emb, neg_emb, temperature=TEMPERATURE):
    pos_sim = F.cosine_similarity(query_emb, pos_emb)
    neg_sim = F.cosine_similarity(query_emb, neg_emb)
    
    loss = -torch.log(torch.exp(pos_sim / temperature) / 
                      (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))
    return loss.mean()

# 옵티마이저 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# FAISS 검색 시스템 생성
faiss_search = FaissSearch("GT-DB.json")

# Supervised Learning 데이터셋 로드
dataset = SupervisedDataset(gt_data, faiss_search)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Supervised Learning 수행
print("Supervised Learning 시작...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        gt_captions, positive_samples, negative_samples = zip(*batch)

        gt_embeddings = model.encode(gt_captions, convert_to_tensor=True).to(DEVICE)
        positive_embeddings = model.encode(positive_samples, convert_to_tensor=True).to(DEVICE)
        negative_embeddings = model.encode(negative_samples, convert_to_tensor=True).to(DEVICE)

        loss = contrastive_loss(gt_embeddings, positive_embeddings, negative_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Supervised Learning Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# 모델 가중치 저장
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/supervised_model.pth")
print("Supervised Learning 완료 및 가중치 저장 완료!")