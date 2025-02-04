import json
import torch
import os
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from text_to_video.embedding import FaissSearch
from contra_dataloader import SupervisedDataset
from loss import ContrastiveLoss

def supervised_learning(sampling_db_path, save_path, batch_size, epochs, learning_rate, top_k):
    """Supervised Learning을 수행하고 모델 가중치를 저장"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sentence Transformer 모델 및 옵티마이저 초기화
    print("모델 로드 중...")
    model = SentenceTransformer("all-MiniLM-L6-v2").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion_loss = ContrastiveLoss()

    # GT-DB 로드
    print("GT 데이터 로드 중...")
    with open(sampling_db_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    print(f"GT 데이터 로드 완료: {len(database)}개의 항목")

    caption_dict = {}
    print("Caption 추출 중...")
    for entry in database:
        video_path = entry["video_path"]
        caption = entry["caption"]
        caption_dict[video_path] = caption

    captions = list(caption_dict.values())
    # # FAISS 검색 시스템 초기화
    # print("FAISS 검색 시스템 초기화 중...")
    # faiss_search = FaissSearch(sampling_db_path)

    # Supervised Learning 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = SupervisedDataset(database)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"데이터셋 준비 완료: {len(dataloader)} 배치")

    # 학습 루프
    print("Supervised Learning 시작...")
    embeddings = {}
    for epoch in range(epochs):
        caption_tensors = model.encode(captions, convert_to_tensor=True).to(DEVICE)

        # Caption과 임베딩을 매칭하여 딕셔너리 생성
        caption_embeddings = {
            video_path: (caption, caption_tensors[i])
            for i, (video_path, caption) in enumerate(caption_dict.items())
        }

        total_loss = 0
        for batch in dataloader:
            video_path, query, gt_caption = batch

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
    sampling_db_path = "GT-DB.json"  # GT 데이터 경로
    SAVE_PATH = "saved_model"    # 모델 가중치 저장 경로
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-5
    TOP_K = 5

    # Supervised Learning 실행
    supervised_learning(
        sampling_db_path=sampling_db_path,
        save_path=SAVE_PATH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        top_k=TOP_K
    )