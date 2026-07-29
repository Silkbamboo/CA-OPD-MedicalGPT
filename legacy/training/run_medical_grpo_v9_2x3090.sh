#!/usr/bin/env bash
set -euo pipefail

# GRPO v9：双卡 3090 24GB，5000条数据，600步
# 启动方式：bash run_medical_grpo_v9_2x3090.sh
#
# 关键改动（vs v8 单卡脚本）：
#   - torchrun --nproc_per_node 2  (DDP 双卡)
#   - per_device_train_batch_size 4  (每卡4 × 2卡 = 8条，保持与v8相同 effective batch)
#   - train_file_dir 换为 train_5000
#   - max_steps 600  (~1.07 epoch over 5000 samples)
#   - save_steps / eval_steps 150
#   - reward_components 精简为 triage,redflag,actionability

export OMP_NUM_THREADS=8
# 让 bitsandbytes QLoRA 在 DDP 下正常工作
export BITSANDBYTES_NOWELCOME=1

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node 2 \
  --master_port 29501 \
  /root/MedicalGPT/medical_grpo_training.py \
  --model_name_or_path /root/MedicalGPT/outputs-medical-grpo-qwen2.5-1.5b-merged-v7-for-v8 \
  --train_file_dir /root/MedicalGPT/grpo_data/train_5000 \
  --train_samples -1 \
  --eval_samples 24 \
  --eval_split_ratio 0.02 \
  --do_train \
  --max_steps 600 \
  --save_steps 150 \
  --save_strategy steps \
  --save_total_limit 4 \
  --output_dir /root/MedicalGPT/outputs-medical-grpo-qwen2.5-1.5b-v9-2x3090-600step \
  --torch_dtype float16 \
  --fp16 True \
  --bf16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 3.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --use_vllm False \
  --logging_steps 10 \
  --eval_steps 150 \
  --eval_strategy steps \
  --use_peft True \
  --qlora True \
  --load_in_4bit True \
  --lora_target_modules all \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_generations 8 \
  --gradient_accumulation_steps 1 \
  --max_prompt_length 1280 \
  --max_completion_length 384 \
  --preprocessing_num_workers 1 \
  --ddp_find_unused_parameters False
