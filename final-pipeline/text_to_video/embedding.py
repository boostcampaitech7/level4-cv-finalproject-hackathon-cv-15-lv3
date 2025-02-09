import faiss
import json
import os
import numpy as np
import requests
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class FaissSearch:
    """FAISS 기반 검색 시스템 클래스"""
    # all-MiniLM-L6-v2, all-mpnet-base-v2
    def __init__(self, json_path, model_name="all-MiniLM-L6-v2", use_gpu=True):
        init_start = time.time()
        print("\n🔧 FAISS 검색 시스템 초기화 중...")
        
        self.json_path = json_path
        
        # 1. 모델 로드
        model_start = time.time()
        print("📥 임베딩 모델 로드 중...")
        self.model = SentenceTransformer(model_name)
        self.model.to("cuda")
        self.model.eval()
        print(f"✓ 모델 로드 완료 ({time.time() - model_start:.1f}초)")

        # 2. JSON 데이터 로드
        json_start = time.time()
        print("📂 JSON 데이터 로드 중...")
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ JSON 로드 완료 ({time.time() - json_start:.1f}초)")
            
            # 3. 임베딩 생성 또는 로드
            embedding_start = time.time()
            if "embedding" not in self.data[0]:
                print("🔄 임베딩 벡터 생성 중...")
                for entry in tqdm(self.data, desc="임베딩 생성"):
                    entry["embedding"] = self.model.encode(entry["caption"]).tolist()
                
                # 임베딩 저장
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=4, ensure_ascii=False)
            print(f"✓ 임베딩 처리 완료 ({time.time() - embedding_start:.1f}초)")
        else:
            print(f"🚨 오류: {self.json_path} 파일을 찾을 수 없습니다!")
            return

        # 4. FAISS 인덱스 생성
        faiss_start = time.time()
        print("🔍 FAISS 인덱스 생성 중...")
        self.captions = [entry["caption"] for entry in self.data]
        self.embeddings = np.array([entry["embedding"] for entry in self.data], dtype=np.float32)
        faiss.normalize_L2(self.embeddings)
        self._initialize_faiss(use_gpu)
        print(f"✓ FAISS 인덱스 생성 완료 ({time.time() - faiss_start:.1f}초)")
        
        print(f"\n✨ 초기화 완료 (총 {time.time() - init_start:.1f}초)")
        print(f"• 총 캡션 수: {len(self.data)}개")

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

    def find_similar_captions(self, input_text, translator, top_k=3):
        """검색 시간 모니터링이 추가된 검색 함수"""
        search_start = time.time()
        
        # 1. 번역
        translate_start = time.time()
        translated_query = translator.translate_ko_to_en(input_text)
        translate_time = time.time() - translate_start
        
        print(f"🔎 검색어: '{input_text}'")
        print(f"🔎 번역된 검색어: '{translated_query}'")
        
        if not translated_query:
            print("🚨 검색어 번역 실패!")
            return []
        
        # 2. 임베딩 생성
        embed_start = time.time()
        query_embedding = self.model.encode([translated_query]).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        embed_time = time.time() - embed_start
        
        # 3. FAISS 검색
        search_start_time = time.time()
        D, I = self.gpu_index.search(query_embedding, top_k)
        search_time = time.time() - search_start_time
        
        # 4. 결과 처리
        results = []
        process_start = time.time()
        for idx, i in enumerate(I[0]):
            # try:
            #     caption_ko = ''
            #     if not caption_ko:  # 번역 실패 시 영어 캡션 사용
            #         print(f"⚠️ 캡션 번역 실패 - 영어 캡션 사용: {self.captions[i][:100]}...")
            #         caption_ko = self.captions[i]
            # except Exception as e:
            #     print(f"⚠️ 캡션 번역 중 오류 - 영어 캡션 사용: {str(e)}")
            #     caption_ko = self.captions[i]
                
            video_folder = self.data[i]['video_path'].split('/')[0]
            video_name = f"{video_folder}.mp4"
            real_video_path = os.path.join("../videos", video_name)
            
            video_info = {
                'video_path': real_video_path,
                'video_id': self.data[i]['video_id'],
                'title': self.data[i]['title'],
                'url': self.data[i]['url'],
                'start_time': float(self.data[i]['start_time']),
                'end_time': float(self.data[i]['end_time'])
            }
            results.append((D[0][idx], video_info))
        
        process_time = time.time() - process_start
        total_time = time.time() - search_start
        
        print("\n⏱️ 검색 성능:")
        print(f"• 번역 시간: {translate_time:.3f}초")
        print(f"• 임베딩 시간: {embed_time:.3f}초")
        print(f"• 검색 시간: {search_time:.3f}초")
        print(f"• 결과 처리 시간: {process_time:.3f}초")
        print(f"• 총 소요 시간: {total_time:.3f}초")
        
        return results
