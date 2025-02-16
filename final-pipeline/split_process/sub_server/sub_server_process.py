import os
import json
import torch
from config import Config
from sentence_transformers import SentenceTransformer
from tarsier_utils import load_model_and_processor

def process():
    print("🤖 Tarsier 모델 로딩 중...")
    model_path = "/data/ephemeral/home/Tarsier-7b"
    model, processor = load_model_and_processor(model_path, max_n_frames=4)
    
    print("🔤 임베딩 모델 로딩 중...")
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    video_files = os.listdir(Config.video_dir)
    results = []
    
    # 배치 크기 설정
    batch_size = 4
    print(f"총 {len(video_files)}개의 비디오 처리 시작... (배치 크기: {batch_size})")
    
    for i in range(0, len(video_files), batch_size):
        batch_video_files = video_files[i:i + batch_size]
        batch_video_paths = [
            os.path.join(Config.video_dir, video_file)
            for video_file in batch_video_files
            if os.path.exists(os.path.join(Config.video_dir, video_file))
        ]
        
        try:
            # 캡션 생성을 위한 배치 입력 준비
            instruction = "<video>\nDescribe the video in detail."
            inputs_list = []
            
            for video_path in batch_video_paths:
                inputs = processor(instruction, video_path, edit_prompt=True, return_prompt=True)
                inputs.pop('prompt', None)
                inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
                inputs_list.append(inputs)
            
            # 배치 입력 생성
            batch_inputs = {k: torch.cat([inputs[k] for inputs in inputs_list], dim=0) for k in inputs_list[0]}
            
            # 배치 추론
            outputs = model.generate(
                **batch_inputs,
                do_sample=True,
                max_new_tokens=512,
                top_p=0.9,
                temperature=0.8,
                use_cache=True
            )
            
            # 각 비디오 결과 처리
            for idx, video_file in enumerate(batch_video_files):
                try:
                    caption = processor.tokenizer.decode(
                        outputs[idx][inputs_list[idx]['input_ids'][0].shape[0]:],
                        skip_special_tokens=True
                    )
                    
                    if not caption:
                        continue
                    
                    # 파일명에서 정보 추출
                    name_parts = os.path.splitext(video_file)[0].split('_')
                    video_name = '_'.join(name_parts[:-2])
                    start_time = float(name_parts[-2])
                    end_time = float(name_parts[-1])
                    
                    # 임베딩 생성
                    embedding = embedding_model.encode([caption])[0]
                    
                    # 결과 저장
                    result = {
                        "video_path": f"{video_name}.mp4",  # 원본 비디오 이름
                        "video_id": "",  # 외부 비디오는 빈 문자열
                        "title": video_name,
                        "url": "",
                        "start_time": str(start_time),  # 문자열로 변환
                        "end_time": str(end_time),  # 문자열로 변환
                        "caption": caption,
                        "embedding": embedding.tolist()
                    }
                    results.append(result)
                    print(f"✓ {video_file} 처리 완료")
                    
                except Exception as e:
                    print(f"✗ {video_file} 개별 처리 실패: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"✗ 배치 처리 실패: {str(e)}")
            continue

    # JSON 파일로 저장
    with open(Config.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n총 {len(results)}/{len(video_files)}개의 비디오 처리 완료")
    print(f"결과가 {Config.output_file}에 저장되었습니다.")

