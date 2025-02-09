import os
import json
import time
import torch
from utils import load_model_and_processor
from tqdm import tqdm
import threading

# ✅ 배치 캡션 생성 함수 (batch_size=2)
def generate_captions_batch(model, processor, video_paths, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0.8):
    """Batch 크기만큼 동영상에서 캡션을 생성하는 함수"""
    
    modified_prompt = "<video>\n Describe the video in detail."
    
    inputs_list = []
    for video_path in video_paths:
        inputs = processor(modified_prompt, video_path, edit_prompt=True, return_prompt=True)
        inputs.pop('prompt', None)  # 필요 없는 prompt 제거
        inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
        inputs_list.append(inputs)
    
    # 배치 입력 생성 (각 입력을 개별적으로 처리)
    batch_inputs = {k: torch.cat([inputs[k] for inputs in inputs_list], dim=0) for k in inputs_list[0]}

    # ✅ 모델 실행 (Batch 처리)
    outputs = model.generate(
        **batch_inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        use_cache=True
    )

    # 결과 디코딩
    captions = []
    for i, inputs in enumerate(inputs_list):
        output_text = processor.tokenizer.decode(outputs[i][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
        captions.append(output_text)

    return captions

# ✅ 경로 설정
video_base_path = "/hdd1/lim_data/YouTube-8M-video-7sec_clips"
json_file_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/tarsier/7sec.json"
captioned_json_file_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/tarsier/7sec.json"

model_path = "/home/hwang/leem/level4-cv-finalproject-hackathon-cv-15-lv3/Tarsier-7b"
error_log_path = "error_log.txt"

# ✅ 모델 및 프로세서 로드
print("🚀 모델과 프로세서를 로드하는 중...")
model, processor = load_model_and_processor(model_path, max_n_frames=8)
model.half()
model.eval()

# ✅ 기존 JSON 파일 로드
with open(json_file_path, 'r', encoding="utf-8") as f:
    video_metadata = json.load(f)

# ✅ 기존 캡션 확인 후, 없는 경우만 처리
for video in video_metadata:
    if 'caption' not in video:
        video['caption'] = None

# ✅ 전체 프로세스 타이머 시작
start_time = time.time()

# ✅ 배치 크기 설정
batch_size = 2

# ✅ 오류 제외된 리스트 (필요 시 수정 가능)
error_index = []

# ✅ Batch-wise 진행
print(f"📊 총 {len(video_metadata)} 개의 비디오를 처리합니다. (Batch Size: {batch_size})")

for i in tqdm(range(0, len(video_metadata), batch_size), desc="Processing Videos"):
    batch_videos = video_metadata[i:i + batch_size]

    # ✅ 기존처럼 각 영상 개별 출력 유지
    for video in batch_videos:
        print(f" {i}/{len(video_metadata)} {video['video_path']}")

    # ✅ 비디오 경로 필터링
    batch_video_paths = [
        os.path.join(video_base_path, video["video_path"])
        for video in batch_videos
        if os.path.exists(os.path.join(video_base_path, video["video_path"]))
    ]

    if not batch_video_paths:
        continue  # 처리할 비디오가 없으면 넘어감

    timer = threading.Timer(20, lambda: print(f"⏳ Timeout: {batch_video_paths}"))
    timer.start()

    try:
        # ✅ 배치 캡션 생성 수행
        captions = generate_captions_batch(model, processor, batch_video_paths, temperature=0.8)

        # ✅ 캡션 추가
        for video, caption in zip(batch_videos, captions):
            video["caption"] = caption
        
        # ✅ 주기적으로 JSON 저장 (20번째 루프마다)
        if i % 20 == 1:
            with open(captioned_json_file_path, "w", encoding="utf-8") as f:
                json.dump(video_metadata, f, indent=4)
            print(f"💾 저장 완료 {i}/{len(video_metadata)} : {video['video_path']}")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

        # 오류 발생 시 JSON 저장
        with open(captioned_json_file_path, "w", encoding="utf-8") as f:
            json.dump(video_metadata, f, indent=4)

        # 오류 로그 저장
        with open(error_log_path, "a", encoding="utf-8") as error_log:
            error_log.write(f"오류 발생! 비디오 경로: {batch_video_paths}\n")

        continue  # 다음 배치로 이동
    finally:
        timer.cancel()  # 타이머 취소

# ✅ 최종 JSON 파일 저장
with open(captioned_json_file_path, "w", encoding="utf-8") as f:
    json.dump(video_metadata, f, indent=4)

# ✅ 전체 프로세스 타이머 종료
end_time = time.time()
print(f"✅ 총 소요 시간: {end_time - start_time:.2f}초")