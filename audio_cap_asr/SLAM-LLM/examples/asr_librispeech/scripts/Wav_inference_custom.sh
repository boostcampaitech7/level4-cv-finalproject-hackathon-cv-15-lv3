#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/SLAM_LLM/SLAM-LLM
cd $run_dir
code_dir=/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/SLAM_LLM/SLAM-LLM/examples/asr_librispeech

speech_encoder_path=/data/ephemeral/home/min/SLAM_model/asr/WavLM-Large.pt
llm_path=/data/ephemeral/home/min/SLAM_model/vicuna-7b-v1.5

#projector_ckpt_path=/data/ephemeral/home/SLAM_model/model.pt


output_dir=/data/ephemeral/home/min/SLAM_model/asr
ckpt_path=$output_dir/Linear
split=librispeech_test_clean
val_data_path=/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/SLAM_LLM/test_data.jsonl
decode_log=/data/ephemeral/home/min/level4-cv-finalproject-hackathon-cv-15-lv3/SLAM_LLM/output/output.json

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=wavlm \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        #++train_config.quantization=true\
        #++model_config.projector_ckpt_path=$projector_ckpt_path\
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
