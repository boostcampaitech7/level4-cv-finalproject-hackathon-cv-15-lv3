import torch
from transformers import AutoModel, AutoTokenizer
import time

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', attn_implementation='eager',torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Here are some frames of a video. Describe this video in detail' #쿼리 입력
image = ['/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/ixc_caption/tmp_clip/video_32_00000_00005.mp4',] # 비디오 주소 입력력
start_time = time.time()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, max_new_tokens=1024,num_beams=1, use_meta=True,hd_num=24) # 모델 인자 변경 가능
    #meta_instruction를 통해 Role 지정가능, 지금은 Default 값 사용중
end_time = time.time()

elapsed_time = end_time - start_time
print(response)
print(f"Inference Time: {elapsed_time:.2f} seconds")



