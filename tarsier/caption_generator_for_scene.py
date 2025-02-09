import os
import json
import time
from utils import load_model_and_processor
import torch
from tqdm import tqdm
import threading

# 비디오 클립에 대한 캡션을 생성하는 함수
# def generate_caption(model, processor, video_path, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0.8):
#     modified_prompt = "<video>\n Describe the video in detail."
    
#     # 수정된 프롬프트를 처리
#     inputs = processor(modified_prompt, video_path, edit_prompt=True, return_prompt=True)
#     inputs.pop('prompt', None)  # 'prompt' 키가 존재할 경우 제거
#     inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
    
#     outputs = model.generate(
#         **inputs,
#         do_sample=True,
#         max_new_tokens=max_new_tokens,
#         top_p=top_p,
#         temperature=temperature,
#         use_cache=True
#     )
    
#     output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
#     return output_text

# 경로 설정
video_base_path = "/hdd1/lim_data/YouTube-8M-video-7sec_clips"
json_file_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/tarsier/7sec.json"
captioned_json_file_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/tarsier/7sec.json"

model_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/Tarsier-7b"
error_log_path = "error_log.txt"

# 모델 및 프로세서 로드
print("모델과 프로세서를 로드하는 중...")
model, processor = load_model_and_processor(model_path, max_n_frames=8)

# 기존 JSON 파일 로드
with open(json_file_path, 'r') as f:
    video_metadata = json.load(f)

for video in video_metadata:
    if 'caption' not in video:
        video['caption'] = None
        
# 전체 프로세스 타이머 시작
start_time = time.time()

# video_metadata = video_metadata[3871:]

def timeout_handler():
    raise TimeoutError("Processing took too long, moving to the next video.")

# JSON 메타데이터를 순회하며 각 클립을 처리

# error_index = [55, 399, 2640, 2641, 2654, 2655, 2658, 2663, 2669, 2670, 2673, 2674, 2678, 2679, 2681, 2682, 2687, 2688, 2691, 2692, 3647, 3883]
error_index = []
# start_index = error_index[-1] + 1  

for i in tqdm(range(len(video_metadata))):  # 시작 인덱스부터 반복
    video = video_metadata[i]
    video_path = os.path.join(video_base_path, video['video_path'])  # video_path 포함하도록 조정
    print(f" {i}/{len(video_metadata)} {video['video_path']}")
    if i in error_index:
        continue
    if not os.path.exists(video_path):
        # print(f"비디오 파일이 존재하지 않습니다: {video_path}")
        continue  # 비디오 파일이 없으면 다음으로 넘어감
    
    timer = threading.Timer(35, timeout_handler)  # 10초 타이머 설정
    timer.start()
    
    try:
        if video['caption'] is not None:
            print(f"이미 캡션이 있습니다. {video_path}")
            continue
        
        prompt = "<video>\n Describe the video in detail."
    
        inputs = processor(prompt, video_path, edit_prompt=True, return_prompt=True)
        # continue
        inputs.pop('prompt', None)  # 'prompt' 키가 존재할 경우 제거
        inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
        
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=512,
            top_p=1,
            temperature=0.8,
            use_cache=True
        )
        
        caption = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        
        video['caption'] = caption 
        
        if i % 20 == 1:
            with open(captioned_json_file_path, "w") as f:
                json.dump(video_metadata, f, indent=4)
            print(f"저장완료 {i}/{len(video_metadata)} : {video['video_path']}")

    except TimeoutError as te:
        pritn(te)
        with open(captioned_json_file_path, "w") as f:
            json.dump(video_metadata, f, indent=4)
        
        error_message = f"오류 발생! 비디오 경로: {video_path}\n"
        with open(error_log_path, "a") as error_log:
            error_log.write(error_message)
        continue  # 타임아웃 발생 시 다음 루프로 넘어감
    except Exception as e:
        print(e)
        with open(captioned_json_file_path, "w") as f:
            json.dump(video_metadata, f, indent=4)
        
        error_message = f"오류 발생! 비디오 경로: {video_path}\n"
        with open(error_log_path, "a") as error_log:
            error_log.write(error_message)
        continue
    finally:
        timer.cancel()  # 타이머 취소

# 업데이트된 JSON 파일 저장
with open(captioned_json_file_path, 'w') as f:
    json.dump(video_metadata, f, indent=4)

# 전체 프로세스 타이머 종료
end_time = time.time()
print(f"총 소요 시간: {end_time - start_time:.2f}초")