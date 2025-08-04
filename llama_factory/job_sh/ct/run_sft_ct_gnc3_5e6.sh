#!/bin/bash


source /path/to/your/project/LLaMA-Factory/.venv/bin/activate

nvidia-smi

export HF_HOME="/path/to/your/project/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/path/to/your/project/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR=$TRITON_CACHE_DIR 

HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login YOUR_WANDB_TOKEN_HERE




echo "granite_full_sftct.yaml"
llamafactory-cli train \
    /path/to/your/project/LLaMA-Factory/my_yaml/2b/ct/granite_full_sftct.yaml \
    learning_rate=5e-6 \
    output_dir=saves/gnc3/eot/sft_ct_5e6




