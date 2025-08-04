#!/usr/bin/env bash

# Load conda and activate merged environment
source /path/to/your/miniconda3/etc/profile.d/conda.sh
conda activate mergel

cd /path/to/your/project/LLaMA-Factory/
nvidia-smi

# Environment variables
export HF_HOME="/path/to/your/project/huggingface"
export TOKENIZERS_PARALLELISM="false"
export TRITON_CACHE_DIR="/path/to/your/project/autotune_cache"
export TRITON_AUTOTUNE_CACHE_DIR="$TRITON_CACHE_DIR"

# Login to Hugging Face
HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

models=(
  "/path/to/your/project/LLaMA-Factory/saves/qwen257/best/sft_ct"
)
outputs=(
  "qwen257_sft_preds_ct.jsonl"
)
data="/path/to/your/project/LLaMA-Factory/data/instruct_code_docstring_train/test.jsonl"
side="sentence-transformers/all-mpnet-base-v2"
batch=64

# Loop through models and execute evaluation
for i in "${!models[@]}"; do
  model="${models[$i]}"
  save="./data/instruct_code_docstring_train/${outputs[$i]}"
  folder_vis="code_sum/${outputs[$i]}"

  echo "Running eval_codesum.py with model: $model"
  python eval_codesum.py \
    --model "$model" \
    --data "$data" \
    --side "$side" \
    --batch_size "$batch" \
    --vllm \
    --save_path "$save"
  
  python json_to_markdowns.py $save $folder_vis --jsonl
done
