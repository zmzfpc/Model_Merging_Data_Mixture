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

# echo "qwen2.5_c1.5_full_sft4o.yaml"
# llamafactory-cli train \
#     /path/to/your/project/LLaMA-Factory/my_yaml/2b/qwen2.5_c15_full_sft4o.yaml \
#     learning_rate=5e-5 \
#     output_dir=saves/qwen2515c15/best/sft_4o_sol_5e5

# echo "qwen2.5_1.5_full_sft4o.yaml"
# llamafactory-cli train \
#     /path/to/your/project/LLaMA-Factory/my_yaml/2b/qwen2.5_full_sft4o.yaml \
#     learning_rate=5e-5 \
#     output_dir=saves/qwen2515/best/sft_4o_sol_5e5

echo "granite_full_sft4o.yaml"
llamafactory-cli train \
    /path/to/your/project/LLaMA-Factory/my_yaml/2b/granite_full_sft4o.yaml \
    learning_rate=5e-5 \
    output_dir=saves/gnc3/best/sft_4o_sol_5e5


echo "dscoder_13_full_sftoc.yaml"
llamafactory-cli train \
    /path/to/your/project/LLaMA-Factory/my_yaml/2b/dscoder_full_sft4o.yaml \
    learning_rate=5e-5 \
    output_dir=saves/dsc13/best/sft_4o_sol_5e5

