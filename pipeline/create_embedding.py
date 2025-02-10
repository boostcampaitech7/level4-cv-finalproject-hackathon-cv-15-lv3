import json
import os
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from tqdm import tqdm

def create_embeddings(input_json_path):
    """JSON 파일의 캡션에 대한 임베딩을 새로 생성하는 함수"""
    print(f"\n🚀 임베딩 생성 시작: {input_json_path}")
    
    # 모델 로드
    print("📦 임베딩 모델 로딩 중...")
    #model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    #model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer('/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/contrastive/fine_tuned_contrastive_model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 원본 파일 읽기
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"💫 총 {len(data)}개 캡션에 대한 임베딩 생성 중...")
    
    # 배치 처리를 위한 설정
    batch_size = 32
    
    # 캡션 수집
    captions = [item['caption'] for item in data]
    
    # 배치 단위로 임베딩 생성
    embeddings = []
    for i in tqdm(range(0, len(captions), batch_size)):
        batch_captions = captions[i:i + batch_size]
        batch_embeddings = model.encode(batch_captions, convert_to_tensor=True)
        embeddings.extend(batch_embeddings.cpu().numpy())
    
    # 임베딩 추가
    for item, embedding in zip(data, embeddings):
        item['embedding'] = embedding.tolist()
    
    # 결과 저장
    output_path = input_json_path.replace('.json', '_new_embeddings.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 새로운 임베딩이 추가된 파일 저장됨: {output_path}")
    print(f"📊 처리된 항목 수: {len(data)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        create_embeddings(json_path)
    else:
        print("❌ JSON 파일 경로를 입력해주세요.")