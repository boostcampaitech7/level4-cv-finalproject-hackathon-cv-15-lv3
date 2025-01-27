import os
import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from decord import VideoReader, cpu
from moviepy import VideoFileClip
from utils.translator import DeepLTranslator, DeepGoogleTranslator
from utils.tarsier_utils import load_model_and_processor
from utils.video_split import create_segmenter

class MPLUGVideoCaptioningPipeline:
    def __init__(self, model_path='mPLUG/mPLUG-Owl3-7B-240728', keep_clips=False, 
                 segmentation_method="fixed", segmentation_params=None, mode='video2text'):
        # 기존 초기화 코드는 그대로 유지
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation='sdpa',
            torch_dtype=torch.half,
            trust_remote_code=True
        )
        self.model.eval().cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = self.model.init_processor(self.tokenizer)
        self.MAX_NUM_FRAMES = 16
        
        self.keep_clips = keep_clips
        self.mode = mode
        
        # 세그멘터 초기화 추가
        segmentation_params = segmentation_params or {"segment_duration": 5}
        self.segmenter = create_segmenter(
            method=segmentation_method, 
            **segmentation_params
        )
        
        # 나머지 초기화 코드는 그대로 유지
        self.output_dir = f"output/{mode}"
        self.clips_dir = f"clips/{mode}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        
        self.video_mapping = {}
        self.video_counter = 1
        self.clip_counter = 1
        self.video_name_to_id = {}
        self.translator = DeepGoogleTranslator()

    def generate_segments(self, video_path):
        """Generate segments for a video using the selected segmentation method"""
        return self.segmenter.get_segments(video_path)


    def _extract_clip(self, video_path, start_time, end_time):
        """Extract clip from video using timestamp"""
        clip = VideoFileClip(video_path).subclipped(start_time, end_time)
        clip_name = f"{os.path.basename(video_path).rsplit('.', 1)[0]}_clip{self.clip_counter}.mp4"
        clip_path = os.path.join(self.clips_dir, clip_name)
        clip.write_videofile(clip_path, codec='libx264', audio=False)
        clip.close()
        return clip_path

    def _encode_video(self, video_path):
        """Encode video frames for processing"""
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > self.MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, self.MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames

    def _generate_caption(self, video_frames):
        """Generate caption for video frames"""
        messages = [
            {"role": "user", "content": "<|video|> Describe this video in detail."},
            {"role": "assistant", "content": ""}
        ]

        inputs = self.processor(messages, images=None, videos=[video_frames])
        inputs.to('cuda')
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': 200,
            'decode_text': True,
        })

        return self.model.generate(**inputs)[0]

    def process_videos(self, video_list):
        """Process list of videos and generate captions"""
        results = []

        for video_path, start_time, end_time in video_list:
            if video_path not in self.video_name_to_id:
                self.video_name_to_id[video_path] = f"video{self.video_counter}"
                self.video_mapping[self.video_name_to_id[video_path]] = {
                    "title": os.path.basename(video_path),
                    "full_path": video_path,
                    "clips": []
                }
                self.video_counter += 1

            video_id = self.video_name_to_id[video_path]
            clip_id = f"clip{self.clip_counter}"

            # Extract clip
            clip_path = self._extract_clip(video_path, start_time, end_time)

            try:
                # Encode and generate caption
                video_frames = self._encode_video(clip_path)
                caption = self._generate_caption(video_frames)
                
                # video2text 모드일 때만 한글 번역 추가
                if self.mode == "video2text":
                    caption_ko = self.translator.translate_en_to_ko(caption)
                else:
                    caption_ko = None

                # Add clip info to mapping with clip path
                self.video_mapping[video_id]["clips"].append({
                    "clip_id": clip_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "clip_path": clip_path
                })

                # Create result entry
                entry = {
                    "video_path": video_path,
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "caption": caption
                }
                
                # video2text 모드일 때만 한글 캡션 추가
                if caption_ko:
                    entry["caption_ko"] = caption_ko
                
                self.clip_counter += 1
                results.append(entry)

            finally:
                # Only remove clip if keep_clips is False
                if not self.keep_clips and os.path.exists(clip_path):
                    os.remove(clip_path)

        return results
    
    def get_video_duration(self, video_path):
        """Get video duration using moviepy"""
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close()
        return duration

    def generate_segments(self, video_path):
        """Generate segments for a video with fixed duration"""
        duration = self.get_video_duration(video_path)
        segments = []
        start_time = 0
        
        while start_time < duration:
            end_time = min(start_time + self.segment_duration, duration)
            if end_time - start_time >= 1:  # 최소 1초 이상인 세그먼트만 포함
                segments.append((start_time, end_time))
            start_time = end_time
        
        return segments

    def process_directory(self, videos_dir):
        """Process all MP4 files in the directory"""
        video_list = []
        for file in os.listdir(videos_dir):
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(videos_dir, file)
                print(f"Processing video: {file}")
                
                # Generate segments for this video
                segments = self.generate_segments(video_path)
                
                # Add segments to video list
                video_list.extend([
                    (video_path, start, end)
                    for start, end in segments
                ])
        
        if not video_list:
            print("Error: No valid videos found to process")
            return None
            
        print(f"Total segments to process: {len(video_list)}")
        return self.process_videos(video_list)

    def save_results(self, results):
        """Save results to JSON files in mode-specific output directory"""
        # Save results with mode-specific names
        if self.mode == "video2text":
            captions_file = "v2t_captions.json"
            mapping_file = "v2t_mapping.json"
        else:  # text2video
            captions_file = "t2v_captions.json"
            mapping_file = "t2v_mapping.json"

        # Save captions
        output_path = os.path.join(self.output_dir, captions_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, indent=4, ensure_ascii=False, fp=f)

        # Save video mapping
        mapping_path = os.path.join(self.output_dir, mapping_file)
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.video_mapping, indent=4, ensure_ascii=False, fp=f)


def find_video_file(videos_dir, video_name):
    """Find video file in directory by name (ignoring extension)"""
    for file in os.listdir(videos_dir):
        if file.rsplit('.', 1)[0] == video_name:
            return os.path.join(videos_dir, file)
    return None


class TarsierVideoCaptioningPipeline:
    def __init__(self, model_path, keep_clips=False, segmentation_method="fixed", 
                 segmentation_params=None, mode='video2text'):
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

    def generate_segments(self, video_path):
        """Generate segments for a video using the selected segmentation method"""
        return self.segmenter.get_segments(video_path)

    def generate_caption(self, video_path):
        """Generate caption for a video clip using Tarsier"""
        instructions = [
            "<video>\nDescribe the video in detail.",
            "<video>\nWhat are the main actions and events happening in this scene?",
            "<video>\nDescribe the behavior and interactions within the video."
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
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        clip_id = f"clip_{self.clip_counter + 1}"
        
        # Create video mapping entry
        if video_id not in self.video_mapping:
            self.video_mapping[video_id] = {
                "video_path": video_path,
                "clips": []
            }
        
        # Extract clip
        clip_path = os.path.join(self.clips_dir, f"{clip_id}.mp4")
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
        
        # Create result entry
        result = {
            "video_path": video_path,
            "video_id": video_id,
            "clip_id": clip_id,
            "start_time": start_time,
            "end_time": end_time,
            "caption": caption
        }
        
        if caption_ko:
            result["caption_ko"] = caption_ko
        
        # Update video mapping
        self.video_mapping[video_id]["clips"].append({
            "clip_id": clip_id,
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
        for file in os.listdir(videos_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(videos_dir, file)
                # segmenter를 사용하여 세그먼트 생성
                segments = self.segmenter.get_segments(video_path)
                for start_time, end_time in segments:
                    video_list.append((video_path, start_time, end_time))
        
        return self.process_videos(video_list)

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