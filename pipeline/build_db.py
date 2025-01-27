import argparse
import os
from video_to_text.video_captioning import MPLUGVideoCaptioningPipeline, TarsierVideoCaptioningPipeline
from text_to_video.embedding import FaissSearch

def build_video_db(model_type="mplug", segmentation_method="fixed", segmentation_params=None):
    """비디오 DB 구축 파이프라인"""
    # 설정 값
    VIDEOS_DIR = "/data/ephemeral/home/jaehuni/selected_videos"
    KEEP_CLIPS = False
    OUTPUT_DIR = "output/text2video"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    TARSIER_MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "..", "Tarsier-7b"))
    
    # 1. 비디오 캡셔닝
    if model_type == "mplug":
        pipeline = MPLUGVideoCaptioningPipeline(
            keep_clips=KEEP_CLIPS,
            segmentation_method=segmentation_method,
            segmentation_params=segmentation_params,
            mode="text2video"
        )
    else:  # tarsier
        pipeline = TarsierVideoCaptioningPipeline(
            model_path=TARSIER_MODEL_PATH,
            keep_clips=KEEP_CLIPS,
            segmentation_method=segmentation_method,
            segmentation_params=segmentation_params,
            mode="text2video"
        )
    
    # 비디오 처리 및 캡션 생성
    results = pipeline.process_directory(VIDEOS_DIR)
    
    # 결과 저장
    pipeline.save_results(results)
    
    # 2. 임베딩 생성 및 저장
    json_path = os.path.join(OUTPUT_DIR, "t2v_captions.json")
    faiss_search = FaissSearch(json_path=json_path)  # 이 과정에서 임베딩이 생성되고 JSON에 저장됨
    
    print(f"✅ DB 구축 완료 → {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description='Build Video Database')
    parser.add_argument('--model', choices=['mplug', 'tarsier'], default='tarsier',
                      help='Choose model type: mplug or tarsier (default: tarsier)')
    parser.add_argument('--segmentation', choices=['fixed', 'scene', 'shot'], default='fixed',
                      help='Choose segmentation method (default: fixed)')
    parser.add_argument('--segment-duration', type=float, default=5.0,
                      help='Duration for fixed segmentation (default: 5.0)')
    
    args = parser.parse_args()
    
    # 세그멘테이션 파라미터 설정
    segmentation_params = {}
    if args.segmentation == 'fixed':
        segmentation_params = {"segment_duration": args.segment_duration}
    
    build_video_db(
        model_type=args.model,
        segmentation_method=args.segmentation,
        segmentation_params=segmentation_params
    )

if __name__ == "__main__":
    main()