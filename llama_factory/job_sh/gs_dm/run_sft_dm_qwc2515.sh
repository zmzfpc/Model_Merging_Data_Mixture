#!/bin/bash


source /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/.venv/bin/activate

nvidia-smi

export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/dccstor/unified-trans/model_merging/granite33_2/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR=$TRITON_CACHE_DIR 

HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login 077132617119d77cc108c66d1da080152704ee04

echo "qwen2.5_c1.5_full_sft4o.yaml"
llamafactory-cli train \
    /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/my_yaml/2b/gs_dm/qwen2.5_c15_full_sft_dm.yaml \
        # learning_rate=5e-6 \
        # output_dir=saves/qwen2515c15/best/sft_4o_sol_5e6

