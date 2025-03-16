import os
import json
import torch
import time
from tqdm import tqdm
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from decord import VideoReader, cpu
from moviepy import VideoFileClip
from utils.translator import DeepLTranslator, DeepGoogleTranslator
from utils.tarsier_utils import load_model_and_processor
from utils.video_split import create_segmenter

@contextmanager
def suppress_output():
    """모든 출력을 억제하는 컨텍스트 매니저"""
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

def find_video_file(videos_dir, video_name):
    """Find video file in directory by name (ignoring extension)"""
    for file in os.listdir(videos_dir):
        if file.rsplit('.', 1)[0] == video_name:
            return os.path.join(videos_dir, file)
    return None


class TarsierVideoCaptioningPipeline:
    def __init__(self, model_path, keep_clips=False, segmentation_method="fixed", 
                 segmentation_params=None, mode='video2text', video_metadata=None, clips_dir=None):
        # Model initialization
        self.model, self.processor = load_model_and_processor(model_path, max_n_frames=8)
        self.model.eval()
        
        self.keep_clips = keep_clips
        self.mode = mode
        
        # 세그멘터 초기화
        segmentation_params = segmentation_params or {}
        self.segmenter = create_segmenter(
            method=segmentation_method, 
            **segmentation_params
        )
        
        # Create output and clips directories
        self.output_dir = f"output/{mode}"
        self.clips_dir = clips_dir or f"clips/{mode}"  # 외부에서 지정한 clips_dir 사용
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        
        # Initialize counters and mappings
        self.clip_counter = 0
        self.video_mapping = {}
        
        self.translator = DeepLTranslator()

        self.video_metadata = video_metadata

    def generate_segments(self, video_path):
        """Generate segments for a video using the selected segmentation method"""
        return self.segmenter.get_segments(video_path)

    def generate_caption(self, video_path):
        """Generate caption for a video clip using Tarsier"""
        instructions = [
            "<video>\nDescribe the video in detail."
        ]
        
        captions = []
        for instruction in instructions:
            try:
                caption = self._generate_single_caption(video_path, instruction)
                captions.append(caption)
            except Exception as e:
                print(f"🚨 캡션 생성 오류: {str(e)}")
                continue
        
        # Combine captions
        final_caption = " ".join(captions)
        return final_caption

    def _generate_single_caption(self, video_path, instruction):
        """Generate a single caption with given instruction"""
        inputs = self.processor(instruction, video_path, edit_prompt=True, return_prompt=True)
        if 'prompt' in inputs:
            inputs.pop('prompt')
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if v is not None}
        
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=512,
            top_p=0.9,
            temperature=0.8,
            use_cache=True
        )
        
        output_text = self.processor.tokenizer.decode(
            outputs[0][inputs['input_ids'][0].shape[0]:], 
            skip_special_tokens=True
        )
        return output_text

    def process_video(self, video_path, start_time, end_time):
        """Process a video segment and generate caption"""
        video_name = os.path.basename(video_path)  # video_XXX.mp4
        if video_name.startswith('video_'):
            video_id = video_name.split('.')[0]  # video_XXX 부분 추출 (확장자 제거)
        else:
            video_id = os.path.splitext(video_name)[0]  # 확장자 제거

        clip_id = f"{video_id}_{int(start_time)}_{int(end_time)}"  # 클립 ID 형식 변경
        
        # Extract clip
        clip_path = os.path.join(self.clips_dir, f"{clip_id}.mp4")  # 여기서 클립 파일명 결정
        try:
            with suppress_output(), VideoFileClip(video_path) as video:
                clip = video.subclipped(start_time, end_time)
                clip.write_videofile(clip_path, codec='libx264', audio=False)
        except Exception as e:
            print(f"🚨 클립 추출 오류: {str(e)}")
            return None
        
        # 메타데이터 가져오기
        metadata = self.video_metadata.get(video_name, {})

        # Generate caption
        with suppress_output():  # 캡션 생성 로그 억제
            caption = self.generate_caption(clip_path)
        if not caption:
            return None
            
        # Translate caption to Korean if in video2text mode
        caption_ko = None
        if self.mode == "video2text":
            caption_ko = self.translator.translate_en_to_ko(caption)
        
        # Create result entry with metadata
        result = {
            "video_path": f"video_{video_id}/{self.clip_counter:05d}.mp4",
            "video_id": metadata.get('video_id', ''),
            "title": metadata.get('title', video_id),
            "url": metadata.get('url', ''),
            "start_time": f"{start_time:.2f}",
            "end_time": f"{end_time:.2f}",
            "caption": caption,
            "clip_path": clip_path  # 실제 저장된 클립 경로 추가
        }
        
        if caption_ko:
            result["caption_ko"] = caption_ko
        
        # Update video mapping
        if video_id not in self.video_mapping:
            self.video_mapping[video_id] = {
                "video_path": video_path,
                "clips": []
            }
        
        self.video_mapping[video_id]["clips"].append({
            "clip_id": clip_id,
            "start_time": start_time,
            "end_time": end_time,
            "clip_path": clip_path
        })
        
        self.clip_counter += 1
        
        # keep_clips가 False인 경우에만 삭제
        if not self.keep_clips and os.path.exists(clip_path):
            os.remove(clip_path)
            
        return result

    def process_videos(self, video_list):
        """Process list of videos"""
        results = []
        for video_path, start_time, end_time in video_list:
            result = self.process_video(video_path, start_time, end_time)
            if result:
                results.append(result)
        return results

    def process_directory(self, videos_dir):
        """Process all videos in directory with detailed monitoring"""
        video_stats = {
            'total_videos': 0,
            'total_segments': 0,
            'total_duration': 0,
            'clip_extraction_time': 0,
            'caption_generation_time': 0,
            'total_success': 0,
            'total_failed': 0
        }
        
        # 1. 비디오 분석
        print("\n📊 비디오 분석 중...")
        analysis_start = time.time()
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        for file in tqdm(video_files, desc="비디오 정보 수집"):
            try:
                with suppress_output():  # VideoFileClip 로그 억제
                    with VideoFileClip(os.path.join(videos_dir, file)) as video:
                        video_stats['total_duration'] += video.duration
                        video_stats['total_videos'] += 1
            except Exception as e:
                print(f"⚠️ {file} 분석 실패: {str(e)}")
        
        analysis_time = time.time() - analysis_start
        print(f"✓ 비디오 분석 완료 ({analysis_time:.1f}초)")
        print(f"• 총 비디오: {video_stats['total_videos']}개")
        print(f"• 총 길이: {video_stats['total_duration']/60:.1f}분")

        # 2. 세그먼트 생성
        print("\n🔄 세그먼트 분할 중...")
        segment_start = time.time()
        video_list = []
        
        for file in tqdm(video_files, desc="세그먼트 생성"):
            video_path = os.path.join(videos_dir, file)
            try:
                with suppress_output():  # 세그먼터 로그 억제
                    segments = self.segmenter.get_segments(video_path)
                    video_stats['total_segments'] += len(segments)
                    video_list.extend([(video_path, start, end) for start, end in segments])
            except Exception as e:
                print(f"⚠️ {file} 세그먼트 생성 실패: {str(e)}")
        
        segment_time = time.time() - segment_start
        print(f"✓ 세그먼트 생성 완료 ({segment_time:.1f}초)")
        print(f"• 총 세그먼트: {video_stats['total_segments']}개")
        print(f"• 평균 세그먼트/비디오: {video_stats['total_segments']/video_stats['total_videos']:.1f}개")

        # 3. 비디오 처리
        results = []
        print("\n🎬 비디오 처리 중...")
        process_start = time.time()
        
        pbar = tqdm(total=len(video_list), desc="세그먼트 처리")
        for video_path, start_time, end_time in video_list:
            clip_start = time.time()
            with suppress_output():  # 비디오 처리 로그 억제
                result = self.process_video(video_path, start_time, end_time)
            
            if result:
                results.append(result)
                video_stats['total_success'] += 1
            else:
                video_stats['total_failed'] += 1
                
            video_stats['clip_extraction_time'] += time.time() - clip_start
            pbar.update(1)
        pbar.close()
        
        process_time = time.time() - process_start
        
        # 최종 통계
        print("\n📊 처리 결과:")
        print(f"• 총 소요 시간: {process_time:.1f}초")
        print(f"• 클립 처리 시간: {video_stats['clip_extraction_time']:.1f}초")
        print(f"• 성공: {video_stats['total_success']}/{video_stats['total_segments']}개")
        print(f"• 실패: {video_stats['total_failed']}개")
        print(f"• 평균 처리 속도: {video_stats['total_duration']/process_time:.1f}초/초")
        
        return results

    def save_results(self, results):
        """Save results to JSON files"""
        if self.mode == "video2text":
            captions_file = "v2t_captions.json"
            mapping_file = "v2t_mapping.json"
        else:
            captions_file = "t2v_captions.json"
            mapping_file = "t2v_mapping.json"

        output_path = os.path.join(self.output_dir, captions_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        mapping_path = os.path.join(self.output_dir, mapping_file)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.video_mapping, f, indent=4, ensure_ascii=False)