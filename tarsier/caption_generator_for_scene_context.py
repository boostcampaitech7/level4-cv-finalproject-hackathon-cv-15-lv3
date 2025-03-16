import os
import json
import time
from utils import load_model_and_processor, get_visual_type
import torch


# Function to generate captions for a video clip
def generate_caption(model, processor, video_path, prompts, prev_caption=None, max_n_frames=8, max_new_tokens=512, top_p=1, temperature=0):
    captions = {}
    for idx, prompt in enumerate(prompts):
        if idx == len(prompts) - 1 and prev_caption is not None:
            # Add context for the last prompt
            modified_prompt = f"<video>\nContext: {prev_caption}\n{prompt}"
        else:
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

        # Update caption with adjusted prompt description for the last prompt
        if idx == len(prompts) - 1:
            captions["Describe the video in detail, taking context into account."] = output_text
        else:
            captions[prompt] = output_text
    return captions


# Path setup
video_base_path = "/data/ephemeral/home/min/video_home"
json_file_path = "/data/ephemeral/home/min/test.json"
model_path = "/data/ephemeral/home/min/Tarsier-7b"

# Prompts for caption generation
prompts = [
    "Describe the video in detail.",
    "Describe the video in detail, taking context into account."
]

# Load model and processor
print("Loading model and processor...")
model, processor = load_model_and_processor(model_path, max_n_frames=8)

# Load the existing JSON file
with open(json_file_path, 'r') as f:
    video_metadata = json.load(f)

# Start timing the entire process
start_time = time.time()

# Variable to store the previous clip's detailed caption

# Iterate through folders and process each clip in sorted order
for folder_name in sorted(os.listdir(video_base_path)):  # Sort folders alphabetically
    folder_path = os.path.join(video_base_path, folder_name)
    previous_caption=None
    if os.path.isdir(folder_path):
        for clip_name in sorted(os.listdir(folder_path)):  # Sort clips alphabetically
            if clip_name.endswith(".mp4"):
                clip_path = os.path.join(folder_path, clip_name)

                # Match video_id and clip_id from JSON
                for video in video_metadata:
                    if video['video_id'] == folder_name and video['clip_id'] in clip_name:
                        print(f"Processing clip: {clip_path}")

                        # Start timing for this video
                        clip_start_time = time.time()
                        print(previous_caption)

                        # Generate captions with context if available
                        captions = generate_caption(
                            model, processor, clip_path, prompts, prev_caption=previous_caption
                        )

                        # Update the previous caption for context
                        previous_caption = captions.get("Describe the video in detail, taking context into account.", previous_caption)

                        # End timing for this video
                        clip_end_time = time.time()
                        print(f"Time taken for clip {clip_name}: {clip_end_time - clip_start_time:.2f} seconds")

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
