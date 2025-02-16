import os 
import torch
import torch.distributed as dist
import json
from utils.args import get_args
from utils.initialize import initialize
from utils.build_model import build_model
from utils.build_dataloader import create_val_dataloaders
from tqdm import tqdm


def main():
    # 1. 환경 변수 설정
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 2. 기존 args 가져오기
    args = get_args()
    
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
    results = []
    for task_name, loader in val_loaders.items():
        print(f"\nProcessing task: {task_name}")
        for batch in tqdm(loader, desc="Generating captions"):
            with torch.no_grad():
                outputs = model(batch, 'cap%tva', compute_loss=False)
                captions = outputs['generated_captions_tva']
                
                # 딕셔너리에서 직접 'ids' 접근
                if 'ids' in batch:  # hasattr 대신 in 연산자 사용
                    for i, (vid_id, cap) in enumerate(zip(batch['ids'], captions)):
                        results.append({
                            'video_id': vid_id,
                            'caption': cap
                        })
                        print(f"\nVideo {vid_id}: {cap}")
                else:
                    print("\nWarning: batch has no ids key")
    
    # 8. 결과 저장
    if dist.get_rank() == 0:
        output_dir = os.path.join(args.run_cfg.output_dir, 'inference_results')
        print(f"\nSaving results to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'video_captions.json')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    # 9. 정리
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()