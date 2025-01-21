import json
import os
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from decord import VideoReader, cpu
import torch
from moviepy import VideoFileClip


class VideoCaptioningPipeline:
    def __init__(self, model_path='mPLUG/mPLUG-Owl3-7B-240728', keep_clips=False):
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
        
        # Create output and clips directories if they don't exist
        self.output_dir = "output"
        self.clips_dir = "clips"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clips_dir, exist_ok=True)
        
        # Initialize video mapping structure
        self.video_mapping = {}

        # Initialize counters for IDs
        self.video_counter = 1
        self.clip_counter = 1
        self.video_name_to_id = {}
        
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
                self.clip_counter += 1
                results.append(entry)
                
            finally:
                # Only remove clip if keep_clips is False
                if not self.keep_clips and os.path.exists(clip_path):
                    os.remove(clip_path)
        
        return results

    def save_results(self, results):
        """Save results to JSON files in output directory"""
        # Save results
        output_path = os.path.join(self.output_dir, "captions.json")
        with open(output_path, 'w') as f:
            json.dump(results, indent=4, fp=f)
            
        # Save video mapping
        mapping_path = os.path.join(self.output_dir, "captions_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(self.video_mapping, indent=4, fp=f)



def find_video_file(videos_dir, video_name):
    """Find video file in directory by name (ignoring extension)"""
    for file in os.listdir(videos_dir):
        # 파일 확장자를 제외한 이름 비교
        if file.rsplit('.', 1)[0] == video_name:
            return os.path.join(videos_dir, file)
    return None

def parse_args():
    parser = argparse.ArgumentParser(description='Video Captioning with Segments')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Directory containing video files')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to input JSON file containing video segments')
    parser.add_argument('--keep_clips', action='store_true',
                        help='Keep the extracted clips instead of deleting them')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize pipeline
    pipeline = VideoCaptioningPipeline(keep_clips=args.keep_clips)
    
    # Load segments from JSON
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)
    
    # Create video list from all videos and their segments
    video_list = []
    for video_data in input_data['videos']:
        video_path = find_video_file(args.videos_dir, video_data['video_name'])
        
        # Verify video exists
        if not video_path:
            print(f"Warning: Video not found: {video_data['video_name']}")
            continue
            
        # Add all segments for this video
        video_list.extend([
            (video_path, seg['start'], seg['end'])
            for seg in video_data['segments']
        ])
    
    if not video_list:
        print("Error: No valid videos found to process")
        exit(1)
    
    # Process videos
    results = pipeline.process_videos(video_list)
    
    # Save results (without specifying output path)
    pipeline.save_results(results)