import os
import json
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from transformers import AutoModel, AutoTokenizer

# Initialize model and tokenizer
torch.set_grad_enabled(False)
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', attn_implementation='eager', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

# Function to split video into 5-second clips
def split_video(video_path, output_dir, clip_duration=5):
    clips = []
    with VideoFileClip(video_path) as video:
        duration = video.duration
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        for start in range(0, int(duration), clip_duration):
            end = min(start + clip_duration, duration)
            clip_filename = f"{base_name}_{start:05d}_{int(end):05d}.mp4"
            clip_path = os.path.join(output_dir, clip_filename)
            video.subclipped(start, end).write_videofile(clip_path, codec='libx264', audio_codec='aac', threads=1, preset='ultrafast', logger=None)
            clips.append({
                "video_path": clip_path,
                "start_time": f"{start:.2f}",
                "end_time": f"{end:.2f}"
            })
    return clips

# Function to generate captions
def generate_caption(video_path):
    query = 'Here are some frames of a video. Describe this video in detail'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        response, _ = model.chat(tokenizer, query, [video_path], do_sample=False, max_new_tokens=1024, num_beams=1, use_meta=True)
    return response

# Function to append JSON data
def append_to_json(json_output_path, data, is_first):
    mode = 'a' if os.path.exists(json_output_path) else 'w'
    with open(json_output_path, mode) as json_file:
        if is_first:
            json_file.write('[\n')  # JSON 배열 시작
        else:
            json_file.write(',\n')  # 각 항목 구분을 위한 쉼표
        json.dump(data, json_file, indent=4)

# Main pipeline function
def process_videos(input_dir, output_dir, json_output_path):
    os.makedirs(output_dir, exist_ok=True)
    is_first = True  # JSON 배열의 첫 번째 항목 여부를 확인

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(f"Processing {video_path}...")

                # Step 1: Split video
                clips = split_video(video_path, output_dir)

                # Step 2: Generate captions
                for clip in clips:
                    caption = generate_caption(clip["video_path"])

                    # Step 3: Save JSON data immediately
                    data = {
                        "video_path": clip["video_path"],
                        "video_id": "",  # Empty field
                        "video_title": "",  # Empty field
                        "video_url": "",  # Empty field
                        "start_time": clip["start_time"],
                        "end_time": clip["end_time"],
                        "caption": caption
                    }
                    append_to_json(json_output_path, data, is_first)
                    is_first = False  # 첫 번째 항목 이후에는 쉼표 추가

                    # Step 4: Delete the split video
                    os.remove(clip["video_path"])

    # Step 5: Close JSON array
    with open(json_output_path, 'a') as json_file:
        json_file.write('\n]')

# Example usage
input_dir = '/data/ephemeral/home/data/split_exp/videos'  # Directory with original videos
output_dir = '/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/tmp_clip'  # Directory to store temporary clips
json_output_path = '/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/output.json'  # JSON output file

# Delete existing JSON file if exists
if os.path.exists(json_output_path):
    os.remove(json_output_path)

process_videos(input_dir, output_dir, json_output_path)
