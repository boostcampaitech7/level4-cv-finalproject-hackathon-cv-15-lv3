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
from tarsier import TarsierForConditionalGeneration, LlavaConfig
import torch
import base64

# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
from typing import List
import torch
from transformers import DataCollatorForSeq2Seq
from transformers.models.llava import LlavaProcessor
import re

# from utils import sample_image, sample_video, sample_gif, get_visual_type



class CustomImageProcessor:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, images: List[Image.Image], do_padding=False) -> torch.Tensor:
        if do_padding:
            images = [self.expand2square(
                img,
                tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
            ) for img in images]
        else:
            images = [self.resize2square(img) for img in images]
        images_pixel = self.processor(text="", images=images, return_tensors="pt")['pixel_values']
        return images_pixel  # [num_images, 3, 336, 336]

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def resize2square(self, pil_img: Image.Image):
        width, height = pil_img.size
        pil_img = pil_img.resize((max(width, height), max(width, height)))
        return pil_img

class Processor(object):
    def __init__(
            self,
            model_name_or_path,
            max_n_frames=8,
            max_seq_len=None,
            add_sep=False,
            do_image_padding=False,
        ):
        self.max_n_frames = max_n_frames
        self.max_seq_len = max_seq_len,
        self.add_sep = add_sep
        self.do_image_padding = do_image_padding
        if not self.do_image_padding:
            print(f"### do_image_padding is set as False, images will be resized directly!")

        self.setup(model_name_or_path)
        
    
    def setup(self, model_name_or_path):
        sub_processor = LlavaProcessor.from_pretrained(
            model_name_or_path,
            padding_side='left',
            trust_remote_code=True,
        )
        self.processor = CustomImageProcessor(sub_processor)
        self.tokenizer = sub_processor.tokenizer
        # self.pad_collator = DataCollatorForSeq2Seq(self.tokenizer, padding='longest')
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id

        if self.sep_id is None:
            self.add_sep = False
        if not self.max_seq_len:
            self.max_seq_len = self.tokenizer.model_max_length

    def process_prompt(self, prompt, images: List[Image.Image]=None):
        if not images:
            prompt = prompt.replace("<image>", "").replace("<video>", "")
        elif images is not None:
            prompt = prompt.replace("<video>", "<image>"*len(images))
            image_token_num = len(re.findall('<image>', prompt, re.S))
            if image_token_num == 0:
                prompt_parts = re.findall(r'USER:(.*)ASSISTANT:(.*)', prompt, re.S)
                if prompt_parts and len(prompt_parts) == 2:
                    p1, p2 = prompt_parts
                else:
                    p1 = prompt
                    p2 = ''
                prompt = f"USER: {'<image>'*len(images) + ' ' + p1.strip()} ASSISTANT: {p2.strip()}"
            assert image_token_num == len(images)
        
        if not re.findall(r'USER:(.*)ASSISTANT:(.*)', prompt, re.S):
            prompt = f'USER: {prompt} ASSISTANT: '
        return prompt

    def select_frames_sampler(self, visual_data_path):
        visual_type = get_visual_type(visual_data_path)
        if visual_type in ext2sampler:
            return ext2sampler[visual_type]
        else:
            raise ValueError(f"Unsupported data format: {visual_data_path}")
        
    def load_images(self, visual_data_path, n_frames=None, start_time=0, end_time=-1):
        sampler = self.select_frames_sampler(visual_data_path)
        return sampler(visual_data_path, n_frames=min(n_frames, self.max_n_frames) if n_frames else self.max_n_frames, start_time=start_time, end_time=end_time)

    def get_pixel_values(self, images):
        if images is not None and len(images) > 0:
            pixel_values = self.processor(images=images, do_padding=self.do_image_padding)
        else:
            pixel_values = None
        return pixel_values

    def get_text_inputs(self, text):
        prompt_ids = self.tokenizer.encode(text, add_special_tokens=True)  # will add <s>
        if self.add_sep:
            prompt_ids = prompt_ids + [self.sep_id]
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(dim=0)
        return prompt_ids

    def get_inputs(self, prompt, visual_data_file=None, images=None, n_frames=None, edit_prompt=False, return_prompt=False):
        if images is None:
            images = self.load_images(visual_data_file, n_frames) if visual_data_file else None
        if edit_prompt:
            prompt = self.process_prompt(prompt, images)
        text_inputs = self.get_text_inputs(prompt)
        pixel_values = self.get_pixel_values(images)
        inputs = {
            "input_ids": text_inputs,
            "pixel_values": pixel_values
        }
        if return_prompt:
            inputs['prompt'] = prompt
        return inputs

    def __call__(self, prompt, visual_data_file=None, images=None, n_frames=None, edit_prompt=False, return_prompt=False):
        return self.get_inputs(prompt, visual_data_file, images, n_frames, edit_prompt, return_prompt)


class Color:

    @staticmethod
    def red(x):
        return '\33[31m' +x + '\033[0m'
    
    @staticmethod
    def green(x):
        return '\33[32m' +x + '\033[0m'

    @staticmethod
    def yellow(x):
        return '\33[33m' +x + '\033[0m'

    @staticmethod
    def blue(x):
        return '\33[34m' +x + '\033[0m'

    @staticmethod
    def violet(x):
        return '\33[35m' +x + '\033[0m'


def load_model_and_processor(model_name_or_path, max_n_frames=8):
    print(Color.red(f"Load model and processor from: {model_name_or_path}; with max_n_frames={max_n_frames}"), flush=True)
    processor = Processor(
        model_name_or_path,
        max_n_frames=max_n_frames,
    )
    model_config = LlavaConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = TarsierForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=model_config,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    return model, processor

def file_to_base64(img_path):
    with open(img_path, 'rb') as video_file:
        video_b64_str = base64.b64encode(video_file.read()).decode()
    return video_b64_str


# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import os
from PIL import Image, ImageSequence
import decord

VALID_DATA_FORMAT_STRING = "Input data must be {'.jpg', '.jpeg', '.png', '.tif'} for image; or {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv', '.gif'}  for videos!"

# 均匀抽帧，必采样首尾帧。
def sample_frame_indices(start_frame, total_frames: int, n_frames: int):
    if n_frames == 1:
        return [0]  # sample first frame in default
    sample_ids = [round(i * (total_frames - 1) / (n_frames - 1)) for i in range(n_frames)]
    sample_ids = [i + start_frame for i in sample_ids]
    return sample_ids

def sample_video(
    video_path: str, 
    n_frames: int = None,
    start_time: int = 0,
    end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(video_path), f"File not found: {video_path}"
    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    vr.seek(0)
    fps=vr.get_avg_fps()
    total_frames = fps*3 #3초일때 가정, 5초면 바꾸기
    fps = vr.get_avg_fps()

    start_frame = 0
    end_frame = total_frames - 1
    if start_time > 0:
        start_frame = min((total_frames-1), int(fps*start_time))
    if end_time > 0:
        end_frame = max(start_frame, int(fps*end_time))
        end_frame = min(end_frame, (total_frames-1))
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )

    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(f).convert('RGB') for f in frames]
    return frames

def sample_gif(
        gif_path: str,
        n_frames:int = None,
        start_time: int = 0,
        end_time: int = -1
    ) -> List[Image.Image]:

    assert os.path.exists(gif_path), f"File not found: {gif_path}"
    
    gif_frames = Image.open(gif_path)

    start_frame = 0
    end_frame = gif_frames.n_frames - 1
    frame_indices = sample_frame_indices(
        start_frame=start_frame,
        total_frames=end_frame - start_frame + 1,
        n_frames=n_frames,
    )
        
    frames = []
    i = 0
    for frame in ImageSequence.Iterator(gif_frames):
        if i in frame_indices:
            frames.append(frame.convert('RGB'))
        i += 1
    return frames

def sample_image(
    image_path: str, 
    n_frames: int = None,
    start_time: int = 0,
    end_time: int = -1
    ):
    assert os.path.exists(image_path), f"File not found: {image_path}"
    image = Image.open(image_path).convert('RGB')
    return [image]

def get_visual_type(input_file):
    ext = os.path.splitext(input_file)[-1]
    if ext in {'.gif'}:
        return 'gif'
    elif ext in {'.mp4', '.avi', '.webm', '.mov', '.mkv', '.wmv'}:
        return 'video'
    elif ext in {'.jpg', '.jpeg', '.png', '.tif'}:
        return 'image'
    else:
        print(f"{VALID_DATA_FORMAT_STRING} But found {ext}!")
        return 'unk'

def get_benchmarks(benchmarks):
    final_benchmarks = []
    type2bm = {
        'dream': ['dream'],
        'caption': ['msvd-caption', 'msr-vtt-caption', 'vatex-caption'],
        'mc_qa': ['next-qa', 'egoschema', 'mvbench', 'video-mme'],
        'oe_qa': ['msvd-qa', 'msr-vtt-qa', 'tgif-qa', 'anet-qa'],
    }
    for bm in benchmarks:
        bm = bm.lower()
        if bm in final_benchmarks:
            continue
        if bm == 'all':
            for v in type2bm.values():
                final_benchmarks.extend(v)
            return final_benchmarks
        if bm in type2bm:
            final_benchmarks.extend(type2bm[bm])
        else:
            final_benchmarks.append(bm)
    return final_benchmarks

ext2sampler = {
    'image': sample_image,
    'gif': sample_gif,
    'video': sample_video
}