import os
import json
import time
from utils import load_model_and_processor
import torch

# 비디오 클립에 대한 캡션을 생성하는 함수
def generate_caption(model, processor, video_path, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
    modified_prompt = "<video>\n Describe the video in detail."
    
    # 수정된 프롬프트를 처리
    inputs = processor(modified_prompt, video_path, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        inputs.pop('prompt')
    inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
    
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