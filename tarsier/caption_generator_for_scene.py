import os
import json
import time
from utils import load_model_and_processor, get_visual_type
import torch

# Function to generate captions for a video clip
def generate_caption(model, processor, video_path, prompts, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
    captions = {}
    for prompt in prompts:
        modified_prompt = f"<video>\n{prompt}"
        
        # Process the modified prompt
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
        captions[prompt] = output_text
    return captions

# Path setup
video_base_path = "/data/ephemeral/home/min/video_home"
json_file_path = "/data/ephemeral/home/min/test2.json"
model_path = "/data/ephemeral/home/min/Tarsier-7b"

# Prompts for caption generation
prompts = [
    "Who are the main characters in this scene?",
    "Describe the behavior and interactions within a video.",
    "Describe the background of the scene.",
    "Describe the video in detail."
]

# Load model and processor
print("Loading model and processor...")
model, processor = load_model_and_processor(model_path, max_n_frames=8)

# Load the existing JSON file
with open(json_file_path, 'r') as f:
    video_metadata = json.load(f)

# Start timing the entire process
start_time = time.time()

# Iterate through JSON metadata and process each clip
for video in video_metadata:
    video_path = os.path.join(video_base_path, video['video_path'])  # Adjusted to include video_path
    if os.path.exists(video_path):
        print(f"Processing clip: {video_path}")
        
        # Start timing for this video
        clip_start_time = time.time()
        
        captions = generate_caption(model, processor, video_path, prompts)
        
        # End timing for this video
        clip_end_time = time.time()
        print(f"Time taken for clip {video['video_path']}: {clip_end_time - clip_start_time:.2f} seconds")
        
        # Update the JSON structure with generated captions
        if 'caption' not in video:
            video['caption'] = {}
        video['caption'].update(captions)

# Save the updated JSON back to file
with open(json_file_path, 'w') as f:
    json.dump(video_metadata, f, indent=4)

# End timing the entire process
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("Captions generated and JSON updated successfully!")
