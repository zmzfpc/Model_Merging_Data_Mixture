#!/bin/bash
# evaltoken=YOUR_HF_TOKEN_HERE
source /path/to/your/project/evalplus/.venv/bin/activate

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/path/to/your/project/huggingface"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential


MODEL_NAME="/path/to/your/project/LLaMA-Factory/saves/qwen2515c15/best/sft_4o_sol_5e5" 

echo "Running evaluation with model: $MODEL_NAME"

evalplus.evaluate --model "$MODEL_NAME" --backend vllm --dataset humaneval --greedy

echo "Evaluation on HumanEval complete."


evalplus.evaluate --model "$MODEL_NAME" --backend vllm --dataset mbpp --greedy

echo "Evaluation complete."