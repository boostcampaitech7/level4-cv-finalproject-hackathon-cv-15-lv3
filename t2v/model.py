import torch
import numpy as np
import pandas as pd
import faiss
import json
import os
import time
import av
import multiprocessing.pool  # ✅ 정식 ThreadPool 사용
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import clip
from tqdm import tqdm

def set_cuda(gpu):
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}\n")
    return device

# GPU 설정 (RTX 3090 최적화)
device = set_cuda(0)

# CLIP 모델 로드 (멀티 GPU 지원)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model = torch.nn.DataParallel(clip_model)

# BLIP-2 모델 로드 (FP16 최적화)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).half()

# ✅ GPU 메모리 정리 함수
def clear_cuda_cache():
    torch.cuda.empty_cache()

# 1. PyAV(FFmpeg 기반) 사용하여 프레임 저장 없이 메모리에서 직접 로드 (FPS 4 적용, 최대 300 프레임 제한)
def extract_frames_memory_pyav(video_path, fps=4):
    start_time_exec = time.time()

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"

    frames = []
    timestamps = []
    frame_interval = max(1, int(video_stream.time_base * fps))  

    print(f"🎥 [프레임 추출] {video_path} - FPS: {fps}")
    for frame in tqdm(container.decode(video=0), desc="Extracting frames"):
        if frame.pts % frame_interval == 0:
            img = frame.to_ndarray(format="rgb24")  # RGB 변환
            timestamp = float(frame.pts * video_stream.time_base)  # 초 단위 시간 변환
            frames.append(img)
            timestamps.append((timestamp, timestamp + (1 / fps)))

    container.close()

    # ✅ 최대 300개 프레임까지만 사용 (메모리 최적화)
    frames = frames[:300]
    timestamps = timestamps[:300]

    end_time_exec = time.time()
    print(f"✅ [프레임 추출 완료] {video_path} - {len(frames)}개 프레임, 실행 시간: {end_time_exec - start_time_exec:.2f}s")
    return frames, timestamps

# 2. BLIP-2 배치 처리 적용 (batch_size=1로 낮춤)
def generate_captions(frames, relevant_timestamps, batch_size=1):
    start_time_exec = time.time()
    
    captions = {}
    num_batches = len(frames) // batch_size + 1

    print("📝 [캡션 생성 중]...")
    for i in tqdm(range(num_batches), desc="Generating captions"):
        batch_frames = frames[i * batch_size:(i + 1) * batch_size]
        batch_images = [Image.fromarray(frame) for frame in batch_frames]

        if len(batch_images) == 0:
            continue

        inputs = blip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        caption_ids = blip_model.generate(**inputs, max_length=50, num_beams=3)
        caption_texts = blip_processor.batch_decode(caption_ids, skip_special_tokens=True)

        for j, caption in enumerate(caption_texts):
            captions[relevant_timestamps[i * batch_size + j]] = caption

    end_time_exec = time.time()
    print(f"✅ [캡셔닝 완료] 실행 시간: {end_time_exec - start_time_exec:.2f}s")

    return captions

# ✅ 3. 개별 영상 분석 함수
def process_single_video(video_path, query_list, relevance_threshold=0.2, fps=4):
    total_start_time = time.time()
    video_name = os.path.basename(video_path).split(".")[0]
    print(f"▶ Processing video: {video_name}")

    frames, timestamps = extract_frames_memory_pyav(video_path, fps=fps)
    
    # ✅ GPU 메모리 정리
    clear_cuda_cache()

    results_json = {"video": video_name, "results": []}

    for query in query_list:
        print(f"🔹 Query: {query}")
        query_embedding = clip_model.module.encode_text(clip.tokenize([query]).to(device)).detach().cpu().numpy()

        frame_embeddings = []
        print("🎯 [CLIP 임베딩 추출 중]...")
        for frame in tqdm(frames, desc="Extracting CLIP embeddings"):
            image = clip_preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            image_embedding = clip_model.module.encode_image(image).detach().cpu().numpy()
            frame_embeddings.append(image_embedding)

        frame_embeddings = np.vstack(frame_embeddings)

        # FAISS 기반 유사도 검색
        index = faiss.IndexFlatL2(frame_embeddings.shape[1])
        index.add(frame_embeddings)
        distances, indices = index.search(query_embedding, len(frames))

        relevant_timestamps = []
        for i, dist in zip(indices[0], distances[0]):
            if dist < relevance_threshold:
                relevant_timestamps.append(timestamps[i])

        captions = generate_captions(frames, relevant_timestamps)

        for (start_time, end_time), caption in captions.items():
            result_entry = {
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "caption_en": caption
            }
            results_json["results"].append(result_entry)

    total_end_time = time.time()
    print(f"✅ [전체 프로세스 완료] {video_name} - 총 실행 시간: {total_end_time - total_start_time:.2f}s")

# ✅ 4. 단일 프로세스로 실행 (멀티프로세싱 OFF)
if __name__ == "__main__":
    video_list = [
        "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/json/videos/new_video_9.mp4",
    ]
    query_list = ["A woman in pink clothes wearing glasses is teaching art to students in school uniforms in the classroom."]

    # ✅ 단일 프로세스로 실행하여 메모리 이슈 확인
    for video in tqdm(video_list, desc="Processing videos"):
        process_single_video(video, query_list, 0.2, 4)