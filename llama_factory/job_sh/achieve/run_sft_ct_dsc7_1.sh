#!/bin/bash


source /path/to/your/project/LLaMA-Factory/.venv/bin/activate

nvidia-smi

cd /path/to/your/project/LLaMA-Factory/

export HF_HOME="/path/to/your/project/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/path/to/your/project/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR=$TRITON_CACHE_DIR 

HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login YOUR_WANDB_TOKEN_HERE


echo "dscoder_7_full_sftct.yaml"
FORCE_TORCHRUN=1 llamafactory-cli train my_yaml/8b/ct/dscoder_7_full_sftct.yaml \
    per_device_train_batch_size=1 \
    learning_rate=1e-6 \
    output_dir=saves/dsc7/best/sft_ct_1e6_1