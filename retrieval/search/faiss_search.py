import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

class DeepGoogleTranslator:
    """deep-translator 라이브러리를 사용한 한국어 ↔️ 영어 번역기 클래스"""
    
    def __init__(self):
        self.ko_to_en = GoogleTranslator(source='ko', target='en')
        self.en_to_ko = GoogleTranslator(source='en', target='ko')
    
    def translate_ko_to_en(self, text):
        """한국어 → 영어 번역"""
        try:
            return self.ko_to_en.translate(text)
        except Exception as e:
            print(f"🚨 번역 오류: {str(e)}")
            return None
    
    def translate_en_to_ko(self, text):
        """영어 → 한국어 번역"""
        try:
            return self.en_to_ko.translate(text)
        except Exception as e:
            print(f"🚨 번역 오류: {str(e)}")
            return None

# all-mpnet-base-v2
# all-MiniLM-L6-v2
class FaissSearch:
    """FAISS 기반 검색 시스템 클래스"""
    
    def __init__(self, json_path, model_name="all-MiniLM-L6-v2", use_gpu=True):
        self.json_path = json_path
        self.translator = DeepGoogleTranslator()
        self.model = SentenceTransformer(model_name)  # SentenceTransformer 모델 설정

        # JSON 데이터 로드
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            print(f"🚨 오류: {self.json_path} 파일을 찾을 수 없습니다!")
            return

        # `embedding`이 없는 항목 건너뛰기
        self.filtered_data = [entry for entry in self.data if "embedding" in entry and entry["embedding"]]
        
        if not self.filtered_data:
            print("🚨 `embedding`이 포함된 데이터가 없습니다!")
            return
        
        # 필터링된 데이터로 임베딩 배열 생성
        self.captions = [entry["caption"] for entry in self.filtered_data]
        self.embeddings = np.array([entry["embedding"] for entry in self.filtered_data], dtype=np.float32)
        faiss.normalize_L2(self.embeddings)

        # FAISS 인덱스 초기화
        self._initialize_faiss(use_gpu)
    
    def _initialize_faiss(self, use_gpu):
        """FAISS 인덱스를 생성 및 초기화"""
        self.dimension = self.embeddings.shape[1]
        self.res = faiss.StandardGpuResources() if use_gpu else None
        self.index = faiss.IndexFlatIP(self.dimension)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index) if use_gpu else self.index
        self.gpu_index.add(self.embeddings)
    
    def find_similar_captions(self, input_text, top_k=3):
        """한국어 입력 → 영어 변환 → FAISS 검색 → 한국어 변환 후 결과 반환"""
        translated_query = self.translator.translate_ko_to_en(input_text)
        # translated_query = input_text
        if not translated_query:
            print("🚨 번역 실패! 입력 텍스트를 확인하세요.")
            return []

        # SentenceTransformer로 임베딩 생성
        query_embedding = self.model.encode([translated_query], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"🚨 임베딩 차원 불일치! Query: {query_embedding.shape[1]}, FAISS Index: {self.dimension}")

        D, I = self.gpu_index.search(query_embedding, top_k)

        results = []
        for idx, i in enumerate(I[0]):
            caption_ko = self.translator.translate_en_to_ko(self.captions[i])
            video_info = {
                'video_path': self.filtered_data[i].get('video_path', 'N/A'),
                'video_id': self.filtered_data[i].get('video_id', 'N/A'),
                'title': self.filtered_data[i].get('title', 'Unknown Title'),
                'url': self.filtered_data[i].get('url', 'N/A'),
                'start_time': self.filtered_data[i].get('start_time', 0),
                'end_time': self.filtered_data[i].get('end_time', 0)
            }
            results.append((caption_ko, D[0][idx], video_info))

        return results