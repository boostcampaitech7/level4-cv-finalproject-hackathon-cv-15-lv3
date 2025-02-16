import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

# 상수 설정
MAX_LENGTH = 1024  # 입력 길이 제한 (기존 32768 → 1024로 줄여 OOM 방지)
BATCH_SIZE = 8  # 배치 크기 (적절한 값으로 조정 가능)

class Embedding:
    def __init__(self, model_name="dunzhang/stella_en_1.5B_v5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True).to(self.device)

    def get_embedding(self, text, prompt_name="s2p_query"):
        # SentenceTransformer의 encode 메서드를 사용하여 임베딩 생성
        return self.model.encode(text, prompt_name=prompt_name)

    @staticmethod
    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(data, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def process_embeddings(self, input_json_path, output_json_path):
        data = self.load_json(input_json_path)
        
        # for debugging
        # data = data[:100]

        # Filter out None captions
        captions = [entry["caption"] for entry in data if "caption" in entry and entry["caption"] is not None]
        
        # Reduce batch size and number of workers
        caption_loader = DataLoader(captions, batch_size=2, shuffle=False, num_workers=1)

        print(f"총 {len(captions)}개의 caption 임베딩을 생성합니다...")

        embeddings = []
        for batch in tqdm(caption_loader, desc="Processing Captions", unit="batch"):
            batch_embeddings = self.get_embedding(batch)
            embeddings.extend(batch_embeddings)

        idx = 0
        for entry in data:
            if "caption" in entry and entry["caption"] is not None:
                # Convert numpy array to list
                entry["embedding"] = embeddings[idx].tolist()
                idx += 1

        self.save_json(data, output_json_path)
        print(f"\n✅ 모든 임베딩이 {output_json_path}에 저장되었습니다.")

# Example usage
input_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/evaluation/DB_v1_no_embedding.json"
output_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/evaluation/DB_v1_embedding.json"

embedding = Embedding()
embedding.process_embeddings(input_json_path, output_json_path)

