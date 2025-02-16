# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils import load_model_and_processor, get_visual_type
import os
import torch

def generate_caption(model, processor, video_path, prompt, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
    """
    Generate a caption for a given video using the specified model and prompt.

    Args:
        model_name_or_path (str): Path to the model.
        video_path (str): Path to the video file.
        prompt (str): The prompt to guide the caption generation.
        max_n_frames (int): Maximum number of frames to sample from the video.
        max_new_tokens (int): Maximum number of tokens to generate.
        top_p (float): Top-p sampling parameter.
        temperature (float): Temperature for sampling.

    Returns:
        str: The generated caption for the video.
    """

    # Prepare inputs
    inputs = processor(prompt, video_path, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        # print(f"Prompt: {inputs.pop('prompt')}")
        inputs.pop('prompt')
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}

    # Generate caption
    outputs = model.generate(
        **inputs,
        do_sample=True if temperature > 0 else False,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        temperature=temperature,
        use_cache=True
    )
    
    output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text

import decord
import os
import numpy as np
from tqdm import tqdm

def check_corrupted_frames_decord(video_path: str):
    """decord를 사용하여 깨진 프레임을 감지하는 함수"""
    assert os.path.exists(video_path), f"File not found: {video_path}"

    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    fps=vr.get_avg_fps()
    print(f"FPS: {fps}")
    total_frames = len(vr)

    print(f"Total frames in video: {total_frames}")

    corrupted_frames = 0
    corrupted_indices = []

    prev_frame = None  # 이전 프레임을 저장하여 비교

    for i in tqdm(range(total_frames), desc="Checking frames"):
        try:
            frame = vr[i].asnumpy()

            # 현재 프레임이 이전 프레임과 동일하면 손상된 것으로 간주
            if prev_frame is not None and np.array_equal(frame, prev_frame):
                corrupted_frames += 1
                corrupted_indices.append(i)
                print(f"⚠️ Corrupted frame detected at index {i} (duplicate frame used for recovery)")

            prev_frame = frame  # 이전 프레임 업데이트

        except Exception as e:
            corrupted_frames += 1
            corrupted_indices.append(i)
            print(f"⚠️ Exception at frame {i}: {e}")

    if corrupted_frames == 0:
        print("✅ No corrupted frames found!")
    else:
        print(f"❌ Found {corrupted_frames} corrupted frames: {corrupted_indices}")

    return corrupted_indices

# Example usage
if __name__ == "__main__":
    model_path = "/data/ephemeral/home/Tarsier-7b"  # 모델 경로
    video_file = "/data/ephemeral/home/min/00012.mp4"  # 비디오 파일 경로
    
    #corrupted_frames = check_corrupted_frames_decord(video_file)
    model, processor = load_model_and_processor(model_path, max_n_frames=8)
    '''total_story = ""
    instruction = "<video>\nWho are the main characters in this scene?"  # 프롬프트
    caption = generate_caption(model, processor, video_file, instruction)
    total_story += f"Main characters: {caption}\n"
    print(f"Generated Caption: {caption}")

    instruction = "<video>\nDescribe the behavior and interactions within a video."  # 프롬프트
    caption = generate_caption(model, processor, video_file, instruction)
    total_story += f"Behavior and interactions: {caption}\n"
    print(f"Generated Caption: {caption}")
    
    instruction = "<video>\nDescribe the background of the scene"  # 프롬프트
    caption = generate_caption(model, processor, video_file, instruction)
    total_story += f"Background: {caption}\n"
    print(f"Generated Caption: {caption}")'''
    
    instruction = "<video>\nDescribe the video in detail."  # 프롬프트
    caption = generate_caption(model, processor, video_file, instruction)
    print(f"Generated Caption: {caption}")