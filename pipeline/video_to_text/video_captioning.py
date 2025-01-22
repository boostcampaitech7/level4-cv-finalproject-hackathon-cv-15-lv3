import os
import json
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from decord import VideoReader, cpu
from moviepy import VideoFileClip
from utils.translator import DeepLTranslator


class VideoCaptioningPipeline:
    def __init__(self, model_path='mPLUG/mPLUG-Owl3-7B-240728', keep_clips=False, segment_duration=5, mode='video2text'):
        # Model initialization
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
        self.segment_duration = segment_duration

        # Create output and clips directories with mode-specific paths
        self.mode = mode
        self.output_dir = f"output/{mode}"
        self.clips_dir = f"clips/{mode}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)

        # Initialize video mapping structure
        self.video_mapping = {}

        # Initialize counters for IDs
        self.video_counter = 1
        self.clip_counter = 1
        self.video_name_to_id = {}

        # Initialize translator
        self.translator = DeepLTranslator()


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
            'max_new_tokens': 100,
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
