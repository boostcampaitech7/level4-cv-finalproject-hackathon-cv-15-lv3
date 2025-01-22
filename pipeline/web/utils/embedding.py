import faiss
import json
import os
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

class FaissSearch:
    """FAISS 기반 검색 시스템 클래스"""

    def __init__(self, json_path, model_name="all-MiniLM-L6-v2", use_gpu=True):
        self.json_path = json_path
        self.model = SentenceTransformer(model_name)

        # ✅ JSON 데이터 로드 또는 생성
        start_time = time.time()
        if os.path.exists(self.json_path):
            self._load_json_data()
        else:
            print("📂 JSON 파일이 존재하지 않음. 새로운 임베딩을 생성합니다...")
            self.generate_and_save_embeddings("output/captions.json")
            self._load_json_data()
        print(f"🕒 JSON 데이터 로드 완료: {time.time() - start_time:.4f} 초")

        # ✅ FAISS 인덱스 초기화
        start_time = time.time()
        self._initialize_faiss(use_gpu)
        print(f"🕒 FAISS 인덱스 초기화 완료: {time.time() - start_time:.4f} 초")

    def _load_json_data(self):
        """JSON 파일을 로드하여 캡션 및 임베딩을 가져옴"""
        start_time = time.time()
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.captions = [entry["caption"] for entry in self.data]
        self.embeddings = np.array([entry["embedding"] for entry in self.data], dtype=np.float32)
        faiss.normalize_L2(self.embeddings)
        print(f"🕒 JSON 로드 및 데이터 정규화 완료: {time.time() - start_time:.4f} 초")

    def _initialize_faiss(self, use_gpu):
        """FAISS 인덱스를 생성 및 초기화"""
        start_time = time.time()
        self.dimension = self.embeddings.shape[1]
        self.res = faiss.StandardGpuResources() if use_gpu else None
        self.index = faiss.IndexFlatIP(self.dimension)
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index) if use_gpu else self.index
        self.gpu_index.add(self.embeddings)
        print(f"🕒 FAISS 인덱스 생성 완료: {time.time() - start_time:.4f} 초")

    def generate_and_save_embeddings(self, source_json_path):
        """새로운 임베딩을 생성하여 JSON 파일로 저장"""
        if not os.path.exists(source_json_path):
            print(f"🚨 오류: {source_json_path} 파일을 찾을 수 없습니다!")
            return

        print("🔄 캡션을 임베딩하고 JSON에 저장 중...")
        start_time = time.time()

        with open(source_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        encode_start_time = time.time()
        for entry in data:
            caption_text = entry["caption"]
            embedding = self.model.encode(caption_text).tolist()  # NumPy 배열을 리스트로 변환
            entry["embedding"] = embedding
        print(f"🕒 모든 캡션 임베딩 완료: {time.time() - encode_start_time:.4f} 초")

        save_start_time = time.time()
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"🕒 JSON 저장 완료: {time.time() - save_start_time:.4f} 초")

        print(f"✅ 새로운 임베딩 저장 완료! → {self.json_path}")
        print(f"🕒 전체 임베딩 처리 완료: {time.time() - start_time:.4f} 초")

    def find_similar_captions(self, input_text, translator, top_k=3):

        """한국어 입력 → 영어 변환 → FAISS 검색 → 한국어 변환 후 결과 반환 (병렬 번역 최적화)"""
        # ✅ Step 1: 입력 텍스트 번역
        start_time = time.time()
        
        if hasattr(translator, "batch_translate"):
            translated_query = translator.batch_translate([input_text], direction="ko_to_en")[0]
        else:
            translated_query = translator.translate_ko_to_en(input_text)

        print(f"🕒 번역 (KO→EN) 완료: {time.time() - start_time:.4f} 초")

        if not translated_query:
            print("🚨 번역 실패! 입력 텍스트를 확인하세요.")
            return []

        # ✅ Step 2: 쿼리 임베딩 생성
        start_time = time.time()
        query_embedding = self.model.encode([translated_query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        print(f"🕒 쿼리 임베딩 생성 완료: {time.time() - start_time:.4f} 초")

        # ✅ Step 3: FAISS 검색
        start_time = time.time()
        D, I = self.gpu_index.search(query_embedding, top_k)
        print(f"🕒 FAISS 검색 완료: {time.time() - start_time:.4f} 초")

        # ✅ Step 4: 검색 결과 번역 (EN → KO) 병렬 최적화
        start_time = time.time()
        
        captions_en = [self.captions[i] for i in I[0]]
        
        if hasattr(translator, "batch_translate"):
            captions_ko = translator.batch_translate(captions_en, direction="en_to_ko")
        else:
            captions_ko = [translator.translate_en_to_ko(caption) for caption in captions_en]

        results = []
        for idx, i in enumerate(I[0]):
            video_info = {
                'video_path': self.data[i]['video_path'],
                'video_id': self.data[i]['video_id'],
                'clip_id': self.data[i]['clip_id'],
                'start_time': self.data[i]['start_time'],
                'end_time': self.data[i]['end_time']
            }
            results.append((captions_ko[idx], D[0][idx], video_info))

        print(f"🕒 검색 결과 번역 완료: {time.time() - start_time:.4f} 초")
        
        return results