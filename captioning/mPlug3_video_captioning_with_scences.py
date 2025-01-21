import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
from decord import VideoReader, cpu
import torch
from scene_splitter import process_videos

def setup_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation='sdpa',
        torch_dtype=torch.half,
        trust_remote_code=True
    )
    model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)
    return model, tokenizer, processor

def encode_video(video_path, max_num_frames=60):
    vr = VideoReader(video_path, ctx=cpu(0))
    frame_idx = list(range(0, len(vr)))
    if len(frame_idx) > max_num_frames:
        frame_idx = frame_idx[::len(frame_idx) // max_num_frames]
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

def generate_caption(model, inputs, tokenizer):
    inputs.to('cuda')
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens': 200,
        'decode_text': True,
    })
    return model.generate(**inputs)

def main():
    # Split video into scenes
    videos_path = "../videos/Home Alone (1990) - Kevin Escapes Scene (5⧸5) ｜ Movieclips.mp4"
    clips_dir = "../videos/clips"
    # process_videos(videos_path, clips_dir)

    # Setup model
    model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
    model, tokenizer, processor = setup_model(model_path)

    # clips dir에 있는 파일 읽어서 캡션 생성
    base_name = os.path.splitext(os.path.basename(videos_path))[0]
    clips_dir = os.path.join(clips_dir, f"{base_name}")
    
    clips_dir = "../videos/clips/test_clips"
    
    clip_files = sorted(os.listdir(clips_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    for i, clip_file in enumerate(clip_files):
        print(clip_file)
        clip_path = os.path.join(clips_dir, clip_file)
        frames = encode_video(clip_path)
        messages = [
            {"role": "user", "content": "<|video|> Carefully analyze the entire video and generate a detailed yet concise caption that captures the overarching context, key events, and interactions observed throughout. Include descriptions of the setting, characters, actions, and any transitions or changes in the scene. Focus on providing a holistic summary that conveys the main idea or narrative of the video, ensuring that all information is grounded in what is explicitly shown, with no added assumptions or fabricated details."},
            {"role": "assistant", "content": ""}
        ]
        inputs = processor(messages, images=None, videos=[frames])
        caption = generate_caption(model, inputs, tokenizer)
        print(f"Scene {i+1} caption: {caption}")
    
if __name__ == "__main__":
    main()