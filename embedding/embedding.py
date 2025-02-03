import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

# 상수 설정
MAX_LENGTH = 1024  # 입력 길이 제한 (기존 32768 → 1024로 줄여 OOM 방지)
BATCH_SIZE = 1  # 배치 크기 (적절한 값으로 조정 가능)

class Embedding:
    def __init__(self, model_name="dunzhang/stella_en_1.5B_v5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def get_embedding(self, text, instruction=""):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        # Debug: Print available keys in the output
        # print("Model output keys:", output.keys())

        if "sentence_embeddings" in output:
            embedding = output["sentence_embeddings"]
        elif "last_hidden_state" in output:
            embedding = output['last_hidden_state']
        else:
            raise ValueError("모델 출력에서 sentence_embeddings 또는 last_hidden_state를 찾을 수 없습니다.")

        return F.normalize(embedding.clone().detach(), p=2, dim=1).cpu().tolist()[0]

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
        captions = [entry["caption"] for entry in data if "caption" in entry]
        caption_loader = DataLoader(captions, batch_size=BATCH_SIZE, shuffle=False)

        print(f"총 {len(data)}개의 caption 임베딩을 생성합니다...")

        embeddings = []
        for batch in tqdm(caption_loader, desc="Processing Captions", unit="batch"):
            batch_embeddings = [self.get_embedding(text) for text in batch]
            embeddings.extend(batch_embeddings)
            torch.cuda.empty_cache()

        idx = 0
        for entry in data:
            if "caption" in entry:
                entry["embedding"] = embeddings[idx]
                idx += 1

        self.save_json(data, output_json_path)
        print(f"\n✅ Embedding 추가 완료! 결과가 {output_json_path}에 저장되었습니다.")

# Example usage
input_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/evaluation/DB_v1.json"
output_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/evaluation/DB_v1_embedding.json"

embedding = Embedding()
embedding.process_embeddings(input_json_path, output_json_path)