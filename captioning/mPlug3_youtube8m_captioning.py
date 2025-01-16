import json
import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from decord import VideoReader, cpu
import torch

# Load JSON data
with open('./Movieclips_annotations.json', 'r') as f:
    data = json.load(f)

# Process only the first 50 entries
data = data[:50]

# Model and configuration setup
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Initialize the model with specific attention implementation
model = AutoModel.from_pretrained(
    model_path,
    attn_implementation='sdpa',
    torch_dtype=torch.half,
    trust_remote_code=True
)
model.eval().cuda()

# Initialize tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

# Define messages for the model
messages_template = [
    {"role": "user", "content": "<|video|> Describe this video."},
    {"role": "assistant", "content": ""}
]

MAX_NUM_FRAMES = 16

def encode_video(video_path):
    """Encodes video frames for processing."""
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

# Process each video and generate captions
for entry in data:
    video_path = os.path.join('../../clip_videos', entry['video_path'])
    video_frames = encode_video(video_path)
    
    # Ensure that the number of video frames matches the expected number of media items
    if len(video_frames) == 0:
        print(f"No frames extracted for video: {video_path}")
        continue

    # Prepare messages for each video
    messages = [
        {"role": "user", "content": "<|video|> Describe this video in detail."},
        {"role": "assistant", "content": ""}
    ]

    # Process the video frames
    inputs = processor(messages, images=None, videos=[video_frames])

    # Prepare inputs for model generation
    inputs.to('cuda')
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens': 100,
        'decode_text': True,
    })

    # Generate output
    g = model.generate(**inputs)
    generated_caption = g[0]  # Assuming the model returns a list of captions

    # Update the JSON entry with the generated caption
    entry['caption'] = generated_caption

# Save the updated JSON data
with open('updated_Movieclips_annotations.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Captions generated and JSON updated for the first 50 videos.")