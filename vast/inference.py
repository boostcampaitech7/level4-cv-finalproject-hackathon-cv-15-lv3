import os 
import torch
import torch.distributed as dist
import json
from utils.args import get_args
from utils.initialize import initialize
from utils.build_model import build_model
from utils.build_dataloader import create_val_dataloaders
from tqdm import tqdm

def create_vast_annotations(movieclips_json_path, output_path):
    # Movieclips JSON 읽기
    with open(movieclips_json_path, 'r') as f:
        movieclips_data = json.load(f)
    
    # VAST 형식으로 변환
    vast_data = []
    
    for item in movieclips_data:
        clip_id = item["clip_id"]
        vast_data.append({
            "video_id": clip_id,
            "desc": [""],  # 빈 description 배열
            "split": "test"
        })
    
    # VAST 형식의 JSON 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(vast_data, f, indent=4)
    print(f"Created VAST annotations at: {output_path}")

def main():
    # 1. 환경 변수 설정
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 2. 기존 args 가져오기
    args = get_args()
    
    # 3. Movieclips annotations에서 VAST 형식 생성
    create_vast_annotations(args.run_cfg.input_json, args.run_cfg.vast_json)
    
    # 4. 초기화
    args.local_rank = 0
    initialize(args)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=0
        )
    
    # 5. 모델 로드
    model, _, _ = build_model(args)
    model.eval()
    
    # 6. 데이터로더 생성
    val_loaders = create_val_dataloaders(args)
    
    # 7. 캡션 생성
    with open(args.run_cfg.input_json, 'r') as f:
        original_data = json.load(f)

    caption_idx = 0
    for task_name, loader in val_loaders.items():
        print(f"\nProcessing task: {task_name}")
        torch.cuda.empty_cache()
        
        for batch in tqdm(loader, desc="Generating captions"):
            with torch.no_grad():
                outputs = model(batch, 'cap%tva', compute_loss=False)
                captions = outputs['generated_captions_tva']
                
                if 'ids' in batch:
                    for vid_id, cap in zip(batch['ids'], captions):
                        # video_id를 기반으로 original_data에서 올바른 인덱스 찾기
                        for idx, item in enumerate(original_data):
                            if str(item["clip_id"]) == str(vid_id):
                                original_data[idx]["caption"] = cap
                                print(f"\nVideo ID: {vid_id}")
                                print(f"Original path: {item['video_path']}")
                                print(f"Generated caption: {cap}")
                                break
    
    # 8. 결과 저장
    if dist.get_rank() == 0:
        output_dir = os.path.join(args.run_cfg.output_dir, 'inference_results')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'movieclips_captions.json')
        
        with open(output_file, 'w') as f:
            json.dump(original_data, f, indent=4)
        print(f"\nResults saved to {output_file}")
    
    # 9. 정리
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()