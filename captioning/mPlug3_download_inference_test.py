from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModel
from decord import VideoReader, cpu  # Ensure decord is installed: pip install decord
import torch

# Model and configuration setup
model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Initialize the model with specific attention implementation
model = AutoModel.from_pretrained(
    model_path,
    attn_implementation='sdpa',  # Ensure this is set correctly
    torch_dtype=torch.half,
    trust_remote_code=True
)
model.eval().cuda()

# Initialize tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

# Define messages for the model
messages = [
    {"role": "user", "content": "<|video|> Describe this video."},
    {"role": "assistant", "content": ""}
]

# Video paths
videos = ['/data/ephemeral/home/videos/Home Alone (1990) - Kevin Escapes Scene (5⧸5) ｜ Movieclips.mp4']
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

# Process video frames
video_frames = [encode_video(video) for video in videos]
inputs = processor(messages, images=None, videos=video_frames)

# Prepare inputs for model generation
inputs.to('cuda')
inputs.update({
    'tokenizer': tokenizer,
    'max_new_tokens': 100,
    'decode_text': True,
})

# Generate output
g = model.generate(**inputs)
print(g)