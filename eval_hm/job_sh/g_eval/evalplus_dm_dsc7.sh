 #!/bin/bash
# evaltoken=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
source /dccstor/unified-trans/model_merging/granite33_2/evalplus/.venv/bin/activate

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential


MODEL_NAME="/dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/saves/dsc7/dm/5e6" 

echo "Running evaluation with model: $MODEL_NAME"

evalplus.evaluate --model "$MODEL_NAME" --backend vllm --dataset humaneval --greedy

echo "Evaluation on HumanEval complete."


evalplus.evaluate --model "$MODEL_NAME" --backend vllm --dataset mbpp --greedy

echo "Evaluation complete."