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

def generate_caption(model_name_or_path, video_path, prompt, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
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
    model, processor = load_model_and_processor(model_name_or_path, max_n_frames=max_n_frames)

    # Prepare inputs
    inputs = processor(prompt, video_path, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        print(f"Prompt: {inputs.pop('prompt')}")
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

# Example usage
if __name__ == "__main__":
    model_path = "../../Tarsier-7b"  # 모델 경로
    video_file = "../dataset/videos/clip_026.mp4"  # 비디오 파일 경로
    instruction = "<video>\nDescribe the video in detail."  # 프롬프트

    caption = generate_caption(model_path, video_file, instruction)
    print(f"Generated Caption: {caption}")