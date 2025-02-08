import os
import json
from config import Config
from sentence_transformers import SentenceTransformer
from tarsier_utils import load_model_and_processor

def process():
    # 모델 한 번만 로드
    print("🤖 Tarsier 모델 로딩 중...")
    model_path = "/data/ephemeral/home/Tarsier-7b"
    model, processor = load_model_and_processor(model_path)
    
    # 임베딩 모델 초기화
    print("🔤 임베딩 모델 로딩 중...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    video_files = os.listdir(Config.video_dir)
    results = []
    
    print(f"총 {len(video_files)}개의 비디오 처리 시작...")
    
    for video_file in video_files:
        video_path = os.path.join(Config.video_dir, video_file)
        
        # 파일명에서 정보 추출
        name_parts = os.path.splitext(video_file)[0].split('_')
        video_name = '_'.join(name_parts[:-2])
        start_time = float(name_parts[-2])
        end_time = float(name_parts[-1])
        
        try:
            # 캡션 생성 (미리 로드한 모델과 프로세서 전달)
            instruction = "<video>\nDescribe the video in detail."
            inputs = processor(instruction, video_path, edit_prompt=True, return_prompt=True)
            if 'prompt' in inputs:
                inputs.pop('prompt')
            
            inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
            
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=512,
                top_p=0.9,
                temperature=0.8,
                use_cache=True
            )
            
            caption = processor.tokenizer.decode(
                outputs[0][inputs['input_ids'][0].shape[0]:], 
                skip_special_tokens=True
            )
            
            if not caption:
                continue
                
            # 임베딩 생성
            embedding = embedding_model.encode([caption])[0]
            
            # 결과 저장
            result = {
                "video_path": f"{video_name}.mp4",  # 원본 비디오 이름으로 저장
                "video_id": "",
                "title": video_name,
                "url": "",
                "start_time": str(start_time),
                "end_time": str(end_time),
                "caption": caption,
                "embedding": embedding.tolist()
            }
            results.append(result)
            print(f"✓ {video_file} 처리 완료")
            
        except Exception as e:
            print(f"✗ {video_file} 처리 실패: {str(e)}")
            continue

    # JSON 파일로 저장
    with open(Config.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n총 {len(results)}/{len(video_files)}개의 비디오 처리 완료")
    print(f"결과가 {Config.output_file}에 저장되었습니다.")

