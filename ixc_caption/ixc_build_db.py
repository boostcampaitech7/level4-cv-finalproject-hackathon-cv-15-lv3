import os
import json
import time
import torch
from transformers import AutoModel, AutoTokenizer

# Initialize model and tokenizer
torch.set_grad_enabled(False)
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', attn_implementation='eager', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

# Function to generate captions
def generate_caption(video_path):
    query = 'Here are some frames of a video. Describe this video in detail.'
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
def process_videos(input_dir, json_output_path):
    is_first = True  # JSON 배열의 첫 번째 항목 여부를 확인

    # Sort folders by name
    video_folders = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for folder in video_folders:
        video_id = os.path.basename(folder)
        video_files = sorted([f for f in os.listdir(folder) if f.endswith('.mp4')])

        for file in video_files:
            video_path = os.path.join(folder, file)
            print(f"Processing {video_path}...")

            # Measure time taken for caption generation
            start_time = time.time()
            caption = generate_caption(video_path)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Time taken for caption: {elapsed_time:.2f} seconds")

            # Save JSON data immediately
            data = {
                "video_path": video_path,
                "video_id": "",
                "title": "",  # Empty field
                "url": "",    # Empty field
                "start_time": "",  # Empty field (optional to fill later)
                "end_time": "",    # Empty field (optional to fill later)
                "caption": caption
            }
            append_to_json(json_output_path, data, is_first)
            is_first = False  # 첫 번째 항목 이후에는 쉼표 추가

    # Close JSON array
    with open(json_output_path, 'a') as json_file:
        json_file.write('\n]')

# Example usage
input_dir = '/data/ephemeral/home/data/gt_videos'  # Directory with video folders
json_output_path = '/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/json/caption_gt.json'  # JSON output file

# Delete existing JSON file if exists
if os.path.exists(json_output_path):
    os.remove(json_output_path)

process_videos(input_dir, json_output_path)
