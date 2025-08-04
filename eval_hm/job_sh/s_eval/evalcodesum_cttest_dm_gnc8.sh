#!/usr/bin/env bash

# Load conda and activate merged environment
source /dccstor/unified-trans/model_merging/miniconda3/etc/profile.d/conda.sh
conda activate mergel

cd /dccstor/unified-trans/model_merging/granite33_2/eval_hm/
nvidia-smi

# Environment variables
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/dccstor/unified-trans/model_merging/granite33_2/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR="$TRITON_CACHE_DIR"

# Login to Hugging Face
HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

models=(
  "/dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/saves/gnc8/dm/5e6"
)
outputs=(
  "gnc8_pred_dm.jsonl"
)
data="/dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/data/instruct_code_docstring_train/test.jsonl"
side="sentence-transformers/all-mpnet-base-v2"
batch=64

# Loop through models and execute evaluation
for i in "${!models[@]}"; do
  model="${models[$i]}"
  save="/dccstor/unified-trans/model_merging/granite33_2/eval_hm/job_out/s_eval/gen_output/${outputs[$i]}"
  folder_vis="code_sum/${outputs[$i]}"

  echo "Running eval_codesum.py with model: $model"
  python eval_codesum.py \
    --model "$model" \
    --data "$data" \
    --side "$side" \
    --batch_size "$batch" \
    --bf16 --vllm \
    --save_path "$save"
  
  python json_to_markdowns.py $save $folder_vis --jsonl
done
