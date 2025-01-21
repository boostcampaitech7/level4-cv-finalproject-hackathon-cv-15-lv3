import faiss
import json
import os
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

class DeepLTranslator:
    """DeepL API를 사용한 한국어 ↔ 영어 번역기 클래스"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api-free.deepl.com/v2/translate"

    def translate(self, text, source_lang, target_lang):
        """DeepL API를 사용하여 번역 수행"""
        params = {
            "auth_key": self.api_key,
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        response = requests.post(self.url, data=params)
        
        if response.status_code != 200:
            print(f"🚨 번역 API 오류: {response.status_code} - {response.text}")
            return None
        
        return response.json().get("translations", [{}])[0].get("text", "")

    def translate_ko_to_en(self, text):
        return self.translate(text, "KO", "EN")

    def translate_en_to_ko(self, text):
        return self.translate(text, "EN", "KO")


class FaissSearch:
    """FAISS 기반 검색 시스템 클래스"""

    def __init__(self, json_path, model_name="all-MiniLM-L6-v2", use_gpu=True):
        self.json_path = json_path
        self.model = SentenceTransformer(model_name)

        # ✅ JSON 데이터 로드 또는 생성
        if os.path.exists(self.json_path):
            self._load_json_data()
        else:
            print("📂 JSON 파일이 존재하지 않음. 새로운 임베딩을 생성합니다...")
            self.generate_and_save_embeddings("output/captions.json")
            self._load_json_data()

        # ✅ FAISS 인덱스 초기화
        self._initialize_faiss(use_gpu)

    def _load_json_data(self):
        """JSON 파일을 로드하여 캡션 및 임베딩을 가져옴"""
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.captions = [entry["caption"] for entry in self.data]
        self.embeddings = np.array([entry["embedding"] for entry in self.data], dtype=np.float32)
        faiss.normalize_L2(self.embeddings)

    def _initialize_faiss(self, use_gpu):
        """FAISS 인덱스를 생성 및 초기화"""
        self.dimension = self.embeddings.shape[1]
        self.res = faiss.StandardGpuResources() if use_gpu else None
        self.index = faiss.IndexFlatIP(self.dimension)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index) if use_gpu else self.index
        self.gpu_index.add(self.embeddings)

    def generate_and_save_embeddings(self, source_json_path):
        """새로운 임베딩을 생성하여 JSON 파일로 저장"""
        if not os.path.exists(source_json_path):
            print(f"🚨 오류: {source_json_path} 파일을 찾을 수 없습니다!")
            return
        
        print("🔄 캡션을 임베딩하고 JSON에 저장 중...")

        with open(source_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            caption_text = entry["caption"]
            embedding = self.model.encode(caption_text).tolist()  # NumPy 배열을 리스트로 변환
            entry["embedding"] = embedding

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✅ 새로운 임베딩 저장 완료! → {self.json_path}")

    def find_similar_captions(self, input_text, translator, top_k=3):
        """한국어 입력 → 영어 변환 → FAISS 검색 → 한국어 변환 후 결과 반환"""
        translated_query = translator.translate_ko_to_en(input_text)
        if not translated_query:
            print("🚨 번역 실패! 입력 텍스트를 확인하세요.")
            return []

        query_embedding = self.model.encode([translated_query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        D, I = self.gpu_index.search(query_embedding, top_k)
        
        results = []
        for idx, i in enumerate(I[0]):
            caption_ko = translator.translate_en_to_ko(self.captions[i])
            video_info = {
                'video_path': self.data[i]['video_path'],
                'video_id': self.data[i]['video_id'],
                'clip_id': self.data[i]['clip_id'],
                'start_time': self.data[i]['start_time'],
                'end_time': self.data[i]['end_time']
            }
            results.append((caption_ko, D[0][idx], video_info))

        return results
