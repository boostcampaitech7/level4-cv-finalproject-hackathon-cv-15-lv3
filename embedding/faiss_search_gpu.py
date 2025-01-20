import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ 1. Load JSON Data
with open("/data/ephemeral/home/embedding/embedding.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ 2. Extract captions & embeddings
captions = [entry["caption"] for entry in data]
embeddings = np.array([entry["embedding"] for entry in data], dtype=np.float32)

# ✅ 3. Normalize embeddings for Cosine Similarity
faiss.normalize_L2(embeddings)  # Normalizing all embeddings for cosine similarity

# ✅ 4. Create FAISS-GPU Index
dimension = embeddings.shape[1]  # Get the embedding dimension
res = faiss.StandardGpuResources()  # GPU resource manager
index = faiss.IndexFlatIP(dimension)  # Use Inner Product (Cosine Similarity)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU

# ✅ 5. Add embeddings to FAISS Index
gpu_index.add(embeddings)

# ✅ 6. Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def find_similar_caption_faiss_gpu(input_text, top_k=3):
    # ✅ 7. Convert query text to embedding
    query_embedding = model.encode([input_text]).astype(np.float32)

    # ✅ 8. Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # ✅ 9. Perform FAISS Search on GPU
    D, I = gpu_index.search(query_embedding, top_k)  # D = Similarity Score, I = Indices

    # ✅ 10. Retrieve top similar captions
    results = [(captions[i], D[0][idx]) for idx, i in enumerate(I[0])]

    return results

# ✅ 11. Run a test search
query_text = "A woman is talking to the camera in a group."
similar_captions = find_similar_caption_faiss_gpu(query_text, top_k=3)

# ✅ 12. Display results
for i, (caption, similarity) in enumerate(similar_captions):
    print(f"🔹 Similarity {i+1}: {similarity:.4f} (Cosine Similarity)")
    print(f"   Caption: {caption}\n")
