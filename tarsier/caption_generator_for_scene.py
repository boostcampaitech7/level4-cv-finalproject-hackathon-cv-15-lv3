import os
import json
import time
from utils import load_model_and_processor
import torch

# 비디오 클립에 대한 캡션을 생성하는 함수
def generate_caption(model, processor, video_path, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
    modified_prompt = "<video>\n Describe the video in detail."
    
    # 수정된 프롬프트를 처리
    inputs = processor(modified_prompt, video_path, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        inputs.pop('prompt')
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
    
    outputs = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        use_cache=True
    )
    
    output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text

# 경로 설정
video_base_path = "/data/ephemeral/home/level4-cv-finalproject-hackathon-cv-15-lv3/dataset/videos"
json_file_path = "/data/ephemeral/home/jiwan/level4-cv-finalproject-hackathon-cv-15-lv3/dataset/video_segments.json"
model_path = "/data/ephemeral/home/Tarsier-7b"

# 모델 및 프로세서 로드
print("모델과 프로세서를 로드하는 중...")
model, processor = load_model_and_processor(model_path, max_n_frames=8)

# 기존 JSON 파일 로드
with open(json_file_path, 'r') as f:
    video_metadata = json.load(f)

# 전체 프로세스 타이머 시작
start_time = time.time()

# JSON 메타데이터를 순회하며 각 클립을 처리
for video in video_metadata:
    video_path = os.path.join(video_base_path, video['video_path'])  # video_path 포함하도록 조정
    if os.path.exists(video_path):
        print(f"클립 처리 중: {video_path}")
        
        # 해당 비디오의 타이머 시작
        clip_start_time = time.time()
        
        caption = generate_caption(model, processor, video_path)
        
        # 해당 비디오의 타이머 종료
        clip_end_time = time.time()
        print(f"클립 {video['video_path']} 처리 시간: {clip_end_time - clip_start_time:.2f}초")
        
        # 생성된 캡션으로 JSON 구조 업데이트
        video['caption'] = caption  # 프롬프트 없이 캡션만 저장

# 업데이트된 JSON 파일 저장
with open(json_file_path, 'w') as f:
    json.dump(video_metadata, f, indent=4)

# 전체 프로세스 타이머 종료
end_time = time.time()
print(f"총 소요 시간: {end_time - start_time:.2f}초")
print("캡션이 생성되고 JSON이 성공적으로 업데이트되었습니다!")