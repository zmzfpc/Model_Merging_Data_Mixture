#!/bin/bash


source /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/.venv/bin/activate

nvidia-smi

cd /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/

export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/dccstor/unified-trans/model_merging/granite33_2/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR=$TRITON_CACHE_DIR 

HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login 077132617119d77cc108c66d1da080152704ee04


echo "dscoder_7_full_sftct.yaml"
FORCE_TORCHRUN=1 llamafactory-cli train my_yaml/8b/ct/dscoder_7_full_sftct.yaml \
    per_device_train_batch_size=1 \
    learning_rate=1e-6 \
    output_dir=saves/dsc7/best/sft_ct_1e6_1