import json
import torch
import os
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from text_to_video.embedding import FaissSearch
from contra_dataloader import SupervisedDataset
from loss import ContrastiveLoss

def supervised_learning(gt_data_path, save_path, batch_size, epochs, learning_rate, top_k):
    """Supervised Learning을 수행하고 모델 가중치를 저장"""
    
    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 로드
    print("모델 로드 중...")
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion_loss = ContrastiveLoss()

    # GT-caption 데이터 로드
    print("GT 데이터 로드 중...")
    with open(gt_data_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    print(f"GT 데이터 로드 완료: {len(gt_data)}개의 항목")

    # FAISS 검색 시스템 초기화
    print("FAISS 검색 시스템 초기화 중...")
    faiss_search = FaissSearch(gt_data_path)

    # Supervised Learning 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = SupervisedDataset(gt_data, faiss_search, top_k=top_k)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"데이터셋 준비 완료: {len(dataloader)} 배치")

    # 학습 루프
    print("Supervised Learning 시작...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            gt_captions, positive_samples, negative_samples = zip(*batch)

            # 임베딩 생성
            gt_embeddings = model.encode(gt_captions, convert_to_tensor=True).to(device)
            positive_embeddings = model.encode(positive_samples, convert_to_tensor=True).to(device)
            negative_embeddings = model.encode(negative_samples, convert_to_tensor=True).to(device)

            # 손실 계산 및 최적화
            loss = criterion_loss.contrastive_loss(gt_embeddings, positive_embeddings, negative_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # 모델 가중치 저장
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "supervised_model.pth"))
    print("Supervised Learning 완료 및 가중치 저장 완료!")


if __name__ == "__main__":
    # 하이퍼파라미터 및 설정값 정의
    GT_DATA_PATH = "GT-DB.json"  # GT 데이터 경로
    SAVE_PATH = "saved_model"    # 모델 가중치 저장 경로
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    TOP_K = 5

    # Supervised Learning 실행
    supervised_learning(
        gt_data_path=GT_DATA_PATH,
        save_path=SAVE_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        top_k=TOP_K
    )