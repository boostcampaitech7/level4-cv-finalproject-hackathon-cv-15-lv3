import os
import json
from sentence_transformers import SentenceTransformer

class CaptionEmbedder:
    def __init__(self, json_path, model_path):
        """임베딩을 저장할 JSON 파일 경로와 사용할 모델을 설정"""
        self.json_path = json_path
        self.model = SentenceTransformer(model_path)  # 로컬 모델 로드

    def generate_and_save_embeddings(self, source_json_path):
        """새로운 임베딩을 생성하여 JSON 파일로 저장"""
        if not os.path.exists(source_json_path):
            print(f"🚨 오류: {source_json_path} 파일을 찾을 수 없습니다!")
            return
        
        print("🔄 캡션을 임베딩하고 JSON에 저장 중...")
        
        with open(source_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            caption_text = entry.get("caption")
            if caption_text:
                embedding = self.model.encode(caption_text).tolist()  # NumPy 배열을 리스트로 변환
                entry["embedding"] = embedding

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ 새로운 임베딩 저장 완료! → {self.json_path}")

# 사용 예시
if __name__ == "__main__":
    source_json = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/json/caption_embedding_tf_mpnet.json"  # 원본 JSON 파일
    output_json = "/data/ephemeral/home/jiwan/level4-cv-finalproject-hackathon-cv-15-lv3/embedding/json/captions_embedding_tf.json"  # 임베딩이 포함된 JSON 파일
    local_model_path = "/data/ephemeral/home/min/all-MiniLM-L6-v2 copy"  # 미리 다운로드된 모델 경로

    embedder = CaptionEmbedder(output_json, local_model_path)
    embedder.generate_and_save_embeddings(source_json)
