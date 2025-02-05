import os
import json
import torch
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
                 segmentation_params=None, mode='video2text', video_metadata=None):
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
        self.clips_dir = f"clips/{mode}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        
        # Initialize counters and mappings
        self.clip_counter = 0
        self.video_mapping = {}
        
        self.translator = DeepGoogleTranslator()

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
            video_id = video_name.split('_')[1]  # XXX 부분 추출
        else:
            video_id = video_name

        clip_id = f"clip_{self.clip_counter + 1}"
        
        # 메타데이터 가져오기
        metadata = self.video_metadata.get(video_name, {})
        
        # Extract clip
        clip_path = os.path.join(self.clips_dir, f"{video_id}_{clip_id}.mp4")
        try:
            with VideoFileClip(video_path) as video:
                clip = video.subclipped(start_time, end_time)
                clip.write_videofile(clip_path, codec='libx264', audio=False)
        except Exception as e:
            print(f"🚨 클립 추출 오류: {str(e)}")
            return None
        
        # Generate caption
        caption = self.generate_caption(clip_path)
        if not caption:
            return None
            
        # Translate caption to Korean if in video2text mode
        caption_ko = None
        if self.mode == "video2text":
            caption_ko = self.translator.translate_en_to_ko(caption)
        
        # Create result entry with metadata
        result = {
            "video_path": f"video_{video_id}/{self.clip_counter:05d}.mp4",  # video_XXX/00001.mp4 형식
            "video_id": metadata.get('video_id', ''),
            "title": metadata.get('title', video_id),
            "url": metadata.get('url', ''),
            "start_time": f"{start_time:.2f}",
            "end_time": f"{end_time:.2f}",
            "caption": caption
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
            "clip_id": f"{self.clip_counter:05d}",
            "start_time": start_time,
            "end_time": end_time,
            "clip_path": clip_path
        })
        
        self.clip_counter += 1
        
        # Clean up if needed
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
        """Process all videos in directory"""
        video_list = []
        print("📂 비디오 목록 생성 중...")
        
        # 비디오 파일 목록 생성
        video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        for file in tqdm(video_files, desc="세그먼트 분할"):
            video_path = os.path.join(videos_dir, file)
            segments = self.segmenter.get_segments(video_path)
            for start_time, end_time in segments:
                video_list.append((video_path, start_time, end_time))
        
        print(f"총 {len(video_files)}개 비디오, {len(video_list)}개 세그먼트 발견")
        
        # 비디오 처리
        results = []
        pbar = tqdm(total=len(video_list), desc="비디오 처리")
        for video_path, start_time, end_time in video_list:
            with suppress_output():
                result = self.process_video(video_path, start_time, end_time)
                if result:
                    results.append(result)
            pbar.update(1)
        pbar.close()
        
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