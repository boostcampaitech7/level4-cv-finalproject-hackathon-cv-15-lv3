import json
import os
import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

class DeepGoogleTranslator:
    def __init__(self):
        self.ko_to_en = GoogleTranslator(source='ko', target='en')
        self.en_to_ko = GoogleTranslator(source='en', target='ko')

    def translate_ko_to_en(self, text):
        try:
            return self.ko_to_en.translate(text)
        except Exception as e:
            print(f"🚨 번역 오류: {str(e)}")
            return None

    def translate_en_to_ko(self, text):
        try:
            return self.en_to_ko.translate(text)
        except Exception as e:
            print(f"🚨 번역 오류: {str(e)}")
            return None

class AnnoySearch:
    def __init__(self, json_path, model_name="all-mpnet-base-v2"):
        self.json_path = json_path
        self.translator = DeepGoogleTranslator()
        self.model = SentenceTransformer(model_name)

        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            print(f"🚨 오류: {self.json_path} 파일을 찾을 수 없습니다!")
            return

        self.filtered_data = [entry for entry in self.data if "embedding" in entry and entry["embedding"]]

        if not self.filtered_data:
            print("🚨 `embedding`이 포함된 데이터가 없습니다!")
            return
        
        self.captions = [entry["caption"] for entry in self.filtered_data]
        self.embeddings = np.array([entry["embedding"] for entry in self.filtered_data], dtype=np.float32)

        self._initialize_annoy()

    def _initialize_annoy(self):
        self.index = AnnoyIndex(self.embeddings.shape[1], 'angular')
        for i, emb in enumerate(self.embeddings):
            self.index.add_item(i, emb)
        self.index.build(10)

    def find_similar_captions(self, input_text, top_k=3):
        translated_query = self.translator.translate_ko_to_en(input_text)
        if not translated_query:
            print("🚨 번역 실패! 입력 텍스트를 확인하세요.")
            return []

        query_embedding = self.model.encode([translated_query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding[0], dtype=np.float32)

        results = self.index.get_nns_by_vector(query_embedding.tolist(), top_k, include_distances=True)

        return [
            (self.translator.translate_en_to_ko(self.captions[i]), d, self.filtered_data[i])
            for i, d in zip(results[0], results[1])
        ]
