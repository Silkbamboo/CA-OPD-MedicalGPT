#!/usr/bin/env bash
set -euo pipefail

DPO_BASE_MODEL_DIR="${DPO_BASE_MODEL_DIR:-/root/MedicalGPT/outputs-sft-medical-85-15-qwen2.5-1.5b-merged-for-dpo}"
TRAIN_DIR="${TRAIN_DIR:-./data/dpo_mix/zh_for_medical_general/train}"
VAL_DIR="${VAL_DIR:-./data/dpo_mix/zh_for_medical_general/val}"

if [ ! -d "$DPO_BASE_MODEL_DIR" ]; then
  echo "Missing merged SFT model directory: $DPO_BASE_MODEL_DIR"
  echo "Run /root/MedicalGPT/run_merge_sft_for_dpo_qwen2_5_1_5b_3090_24g.sh first, or override DPO_BASE_MODEL_DIR."
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python /root/MedicalGPT/dpo_training.py           --model_name_or_path "$DPO_BASE_MODEL_DIR"           --template_name qwen           --train_file_dir "$TRAIN_DIR"           --validation_file_dir "$VAL_DIR"           --per_device_train_batch_size 1           --gradient_accumulation_steps 16           --per_device_eval_batch_size 1           --do_train           --do_eval           --use_peft True           --qlora True           --load_in_4bit True           --max_train_samples -1           --max_eval_samples 100           --max_steps 600           --eval_steps 100           --save_steps 200           --logging_steps 10           --learning_rate 1e-5           --warmup_steps 50           --lr_scheduler_type cosine           --max_source_length 1536           --max_target_length 512           --output_dir outputs-dpo-zh-medical-qwen2.5-1.5b-3090-v1           --target_modules all           --lora_rank 16           --lora_alpha 32           --lora_dropout 0.05           --torch_dtype float16           --bf16 False           --fp16 True           --device_map auto           --report_to tensorboard           --remove_unused_columns False           --gradient_checkpointing True           --cache_dir ./cache
