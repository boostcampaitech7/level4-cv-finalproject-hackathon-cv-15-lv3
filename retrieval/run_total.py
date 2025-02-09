#!/usr/bin/env python
import time
import numpy as np
import hnswlib
import faiss
import scann
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
# import ngt  # 필요하면 활성화

# 테스트 데이터 설정
dim = 128  # 벡터 차원
num_elements = 10000  # 인덱스에 추가할 데이터 개수
top_k = 5  # 검색할 상위 k개
num_trials = 100  # 검색 실행 횟수

# 랜덤 데이터 생성 (정규화 & 노이즈 추가)
data = np.random.rand(num_elements, dim).astype(np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)  # 정규화
data += np.random.normal(0, 0.01, data.shape).astype(np.float32)  # 노이즈 추가

query = np.random.rand(1, dim).astype(np.float32)
query /= np.linalg.norm(query)  # 정규화
query += np.random.normal(0, 0.01, query.shape).astype(np.float32)  # 노이즈 추가

# 결과 저장
timing_results = {
    "HNSW": [],
    "FAISS": [],
    "ScaNN": [],
    "Annoy": []
}

### 1️⃣ HNSW (hnswlib)
hnsw_index = hnswlib.Index(space='l2', dim=dim)
hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)
hnsw_index.add_items(data, np.arange(num_elements))
hnsw_index.set_ef(50)

for _ in range(num_trials):
    start_time = time.perf_counter()
    hnsw_index.knn_query(query, k=top_k)
    timing_results["HNSW"].append(time.perf_counter() - start_time)

### 2️⃣ FAISS
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(data)

for _ in range(num_trials):
    start_time = time.perf_counter()
    faiss_index.search(query, top_k)
    timing_results["FAISS"].append(time.perf_counter() - start_time)

### 3️⃣ ScaNN (query를 1D로 변환)
scann_index = scann.scann_ops_pybind.builder(data, 10, "dot_product").score_brute_force().build()

for _ in range(num_trials):
    start_time = time.perf_counter()
    scann_index.search(query.flatten(), top_k)  # ✅ 1D 변환 필수
    timing_results["ScaNN"].append(time.perf_counter() - start_time)

### 4️⃣ Annoy
annoy_index = AnnoyIndex(dim, 'angular')
for i in range(num_elements):
    annoy_index.add_item(i, data[i])
annoy_index.build(10)

for _ in range(num_trials):
    start_time = time.perf_counter()
    annoy_index.get_nns_by_vector(query.flatten().tolist(), top_k, include_distances=True)
    timing_results["Annoy"].append(time.perf_counter() - start_time)

# ### 5️⃣ NGT (필요하면 활성화)
# ngt_index = ngt.Index.create(b"ngt_index", dim)
# for i in range(num_elements):
#     ngt_index.insert(data[i])
# ngt_index.build_index()

# for _ in range(num_trials):
#     start_time = time.perf_counter()
#     ngt_index.search(query.flatten().tolist(), top_k)
#     timing_results["NGT"].append(time.perf_counter() - start_time)

# 평균 실행 시간 계산
avg_times = {method: np.mean(times) for method, times in timing_results.items()}

# 결과 출력
print("\n🔹 검색 속도 비교 결과 (100회 평균, 초)")
for method, time_taken in avg_times.items():
    print(f"{method}: {time_taken:.6f} 초")

# 결과 시각화
plt.figure(figsize=(8, 5))
plt.bar(avg_times.keys(), avg_times.values(), color=['blue', 'red', 'green', 'purple'])
plt.xlabel("Search Algorithm")
plt.ylabel("Average Search Time (seconds)")
plt.title("Comparison of Search Algorithms (100 Trials)")
plt.show()
