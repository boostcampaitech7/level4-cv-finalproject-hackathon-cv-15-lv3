import os
import json
import time
from utils import load_model_and_processor
import torch
from tqdm import tqdm
import threading

def generate_captions_batch(model, processor, video_paths, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
    modified_prompt = "<video>\n Describe the video in detail."
    
    # 각 비디오에 대해 입력을 준비
    inputs_list = []
    for video_path in video_paths:
        inputs = processor(modified_prompt, video_path, edit_prompt=True, return_prompt=True)
        inputs.pop('prompt', None)
        inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
        inputs_list.append(inputs)
    
    # 배치 입력 생성 (각 입력을 개별적으로 처리)
    batch_inputs = {k: torch.cat([inputs[k] for inputs in inputs_list], dim=0) for k in inputs_list[0]}
    
    outputs = model.generate(
        **batch_inputs,
        do_sample=temperature > 0,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        use_cache=True
    )
    
    captions = []
    for i, inputs in enumerate(inputs_list):
        output_text = processor.tokenizer.decode(outputs[i][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        captions.append(output_text)
    
    return captions

# 경로 설정
video_base_path = "/data/ephemeral/home/split_process/split_process_videos"
json_file_path = "/data/ephemeral/home/split_process/split_process_json/video_files_198.18.12.111.json"
captioned_json_file_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/dataset/video_segments_with_caption.json"

model_path = "/data/ephemeral/home/Tarsier-7b"
error_log_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/error_log.txt"

# 모델 및 프로세서 로드
print("모델과 프로세서를 로드하는 중...")
model, processor = load_model_and_processor(model_path, max_n_frames=8)
model.half()
model.eval()

# 기존 JSON 파일 로드
with open(json_file_path, 'r') as f:
    video_metadata = json.load(f)

# 캡션 초기화
for video in video_metadata: 
    if 'caption' not in video:
        video['caption'] = None

# 전체 프로세스 타이머 시작
start_time = time.time()

# video_metadata = video_metadata[3871:]

def timeout_handler():
    raise TimeoutError("Processing took too long, moving to the next video.")

# JSON 메타데이터를 순회하며 각 클립을 처리
batch_size = 2  # 배치 크기 설정
print(len(video_metadata))
for i in range(0, len(video_metadata), batch_size):
    batch_videos = video_metadata[i:i + batch_size]
    
    # 디버깅: 비디오 경로와 존재 여부 출력
    for video in batch_videos:
        video_path = os.path.join(video_base_path, video['video_path'])
        # print(f"Checking video path: {video_path}, Exists: {os.path.exists(video_path)}, Caption: {video.get('caption')}")
    
    batch_video_paths = [
        os.path.join(video_base_path, video['video_path'])
        for video in batch_videos
        if os.path.exists(os.path.join(video_base_path, video['video_path']))
    ]
    
    # print(f"Batch video paths: {batch_video_paths}")  # 디버깅: 배치 비디오 경로 출력
    
    if not batch_video_paths:
        continue
    
    try:
        captions = generate_captions_batch(model, processor, batch_video_paths)
        # print(captions)
        for video, caption in zip(batch_videos, captions):
            video['caption'] = caption
        
        if i % 100 == 0:
            with open(captioned_json_file_path, "w") as f:
                json.dump(video_metadata, f, indent=4)

    except Exception as e:
        print(e)
        with open(captioned_json_file_path, "w") as f:
            json.dump(video_metadata, f, indent=4)
        
        error_message = f"오류 발생! 비디오 경로: {batch_video_paths}\n"
        with open(error_log_path, "a") as error_log:
            error_log.write(error_message)
        continue

# 업데이트된 JSON 파일 저장
with open(captioned_json_file_path, 'w') as f:
    json.dump(video_metadata, f, indent=4)

# 전체 프로세스 타이머 종료
end_time = time.time()
print(f"총 소요 시간: {end_time - start_time:.2f}초")