import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class CaptionMatcher:
    def __init__(self, model_path, test_json_path):
        """
        모델과 테스트 데이터 로드
        :param model_path: SentenceTransformer 로컬 모델 경로
        :param test_json_path: 테스트 데이터 JSON 파일 경로
        """
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 로컬 모델 로드
        #self.model = SentenceTransformer(model_path)  # 로컬 모델 로드
        self.test_json_path = test_json_path
        self.test_data = self.load_test_data()

    def load_test_data(self):
        """JSON 파일에서 Test 데이터 로드"""
        if not os.path.exists(self.test_json_path):
            print(f"🚨 오류: {self.test_json_path} 파일을 찾을 수 없습니다!")
            return []

        with open(self.test_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"📂 Loaded {len(data)} samples from {self.test_json_path}")
        print(f"🔎 First entry preview: {data[0] if len(data) > 0 else 'No data'}")

        return data

    def generate_embeddings(self):
        """쿼리(1번)와 캡션(0번)의 임베딩 생성"""
        captions = [entry[0] for entry in self.test_data]  # 캡션
        queries = [entry[1] for entry in self.test_data]  # 쿼리

        caption_embeddings = self.model.encode(captions, convert_to_numpy=True)
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)

        return captions, queries, caption_embeddings, query_embeddings

    def build_faiss_index(self, caption_embeddings):
        """Faiss 인덱스 생성 (캡션 임베딩 저장)"""
        d = caption_embeddings.shape[1]  # 벡터 차원
        index = faiss.IndexFlatIP(d)  # 내적을 사용한 Index 생성 (Cosine 유사도에 활용)
        faiss.normalize_L2(caption_embeddings)  # L2 정규화 → 코사인 유사도와 동일한 효과
        index.add(caption_embeddings)

        return index

    def test_matching(self, top_k=5):
        """쿼리와 캡션 간의 코사인 유사도 기반 매칭 수행"""
        captions, queries, caption_embeddings, query_embeddings = self.generate_embeddings()
        index = self.build_faiss_index(caption_embeddings)

        # L2 정규화 (Cosine Similarity 활용)
        faiss.normalize_L2(query_embeddings)

        # Faiss 인덱스를 사용하여 Top-K 검색
        distances, retrieved_indices = index.search(query_embeddings, top_k)

        # 결과 저장
        results = []
        for i, query in enumerate(queries):
            retrieved_captions = [captions[idx] for idx in retrieved_indices[i]]
            retrieved_scores = distances[i].tolist()  # 유사도 점수

            results.append({
                "query": query,
                "ground_truth_caption": captions[i],  # 원래 매칭된 캡션
                "retrieved_captions": retrieved_captions,  # 유사한 캡션 리스트
                "retrieved_scores": retrieved_scores  # 유사도 점수
            })

        return results

    def compute_accuracy(self, results):
        """Top-1 및 Top-5 기준 정답률 계산"""
        total_samples = len(results)
        top1_correct = 0
        top5_correct = 0

        for result in results:
            ground_truth = result["ground_truth_caption"]
            retrieved_captions = result["retrieved_captions"]

            if ground_truth == retrieved_captions[0]:  # Top-1 매칭 확인
                top1_correct += 1
            if ground_truth in retrieved_captions:  # Top-5 매칭 확인
                top5_correct += 1

        top1_accuracy = top1_correct / total_samples * 100
        top5_accuracy = top5_correct / total_samples * 100

        print(f"🎯 Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_samples})")
        print(f"🎯 Top-5 Accuracy: {top5_accuracy:.2f}% ({top5_correct}/{total_samples})")

        return top1_accuracy, top5_accuracy

    def save_results(self, output_path, top_k=5):
        """매칭 결과를 JSON 파일로 저장 + Accuracy 출력"""
        results = self.test_matching(top_k)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"✅ Test 매칭 결과 저장 완료 → {output_path}")

        # 🔹 정답률 계산
        self.compute_accuracy(results)

# ✅ 실행 코드
if __name__ == "__main__":
    model_path = "/data/ephemeral/home/finetune/final"  # 미리 훈련된 모델 경로
    test_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data_train_test/test_sentence_pairs.json"  # Test 데이터셋
    output_json_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/embedding_train/data_train_test/test_results.json"  # 결과 저장 경로

    matcher = CaptionMatcher(model_path, test_json_path)
    matcher.save_results(output_json_path, top_k=5)  # Top-5 유사 문장 저장
