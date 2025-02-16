from embedding import FaissSearch, DeepLTranslator

# ✅ DeepL API 키 설정
DEEPL_API_KEY = "dabf2942-070c-47e2-94e1-b43cbef766e3:fx"

# ✅ FAISS 검색 시스템 및 번역기 초기화
json_path = "/data/ephemeral/home/embedding/modual/embedding.json"
source_json_path = "/data/ephemeral/home/embedding/modual/updated_Movieclips_annotations.json"

translator = DeepLTranslator(api_key=DEEPL_API_KEY)
faiss_search = FaissSearch(json_path=json_path)

# ✅ (선택) 새로운 임베딩 생성 필요 시 호출
faiss_search.generate_and_save_embeddings(source_json_path)

# ✅ 검색 실행 (한국어 입력)
query_text = "여성이 그룹에서 카메라를 보고 이야기하고 있다."
similar_captions = faiss_search.find_similar_captions(query_text, translator, top_k=3)

# ✅ 결과 출력
for i, (caption, similarity) in enumerate(similar_captions):
    print(f"🔹 유사도 {i+1}: {similarity:.4f} (코사인 유사도 기반)")
    print(f"   캡션: {caption}\n")
