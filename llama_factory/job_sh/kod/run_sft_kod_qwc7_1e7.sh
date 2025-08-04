#!/bin/bash


source /path/to/your/project/LLaMA-Factory/.venv/bin/activate
cd /path/to/your/project/LLaMA-Factory/
nvidia-smi

export HF_HOME="/path/to/your/project/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/path/to/your/project/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR=$TRITON_CACHE_DIR 

HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login YOUR_WANDB_TOKEN_HERE

echo "qwen2.5_c7_full_sft4o.yaml"
llamafactory-cli train my_yaml/8b/kod/qwen2.5_c7_full_sft4o.yaml \
    learning_rate=1e-7 \
    output_dir=saves/qwen25c7/best/sft_4o_sol_1e7

