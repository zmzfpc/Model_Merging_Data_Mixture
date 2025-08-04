#!/bin/bash

cd /dccstor/unified-trans/model_merging/granite33_2/mergekit/

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Function to generate YAML configuration
generate_yaml() {
    local density1=$1
    local weight1=$2
    local density2=$3
    local weight2=$4
    local output_file=$5
    
    cat > "my_yaml/${output_file}.yml" << EOF
models:
  - model: /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/saves/qwen2515/best/sft_4o_sol
    parameters:
      density: ${density1}
      weight: ${weight1}
  - model: /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/saves/qwen2515/oneM/sft_ct
    parameters:
      density: ${density2}
      weight: ${weight2}
      
merge_method: dare_linear
chat_template: qwen
base_model: Qwen/Qwen2.5-1.5B-Instruct
dtype: bfloat16
EOF
}

# Define hyperparameter combinations to test
# Format: density1 weight1 density2 weight2 config_name
hyperparams=(
    "0.7 0.7 0.3 0.3 dare_qw2_original"
    "0.5 0.7 0.5 0.3 dare_qw2_balanced_density"
    "0.9 0.6 0.1 0.4 dare_qw2_high_density"
    "0.8 0.8 0.2 0.2 dare_qw2_high_weight"
    "0.6 0.5 0.4 0.5 dare_qw2_balanced_all"
    # Add more combinations as needed
)

echo "Generating and testing ${#hyperparams[@]} different hyperparameter configurations..."

for i in "${!hyperparams[@]}"; do
    # Parse hyperparameters
    read -r density1 weight1 density2 weight2 config_name <<< "${hyperparams[$i]}"
    
    echo "=========================================="
    echo "Configuration $((i+1))/${#hyperparams[@]}: ${config_name}"
    echo "Model 1: density=${density1}, weight=${weight1}"
    echo "Model 2: density=${density2}, weight=${weight2}"
    echo "=========================================="
    
    # Generate YAML file
    generate_yaml "$density1" "$weight1" "$density2" "$weight2" "$config_name"
    echo "Generated YAML: my_yaml/${config_name}.yml"
    
    # Create output directory
    mkdir -p "merged_model/${config_name}"
    
    # Run merge
    echo "Starting merge for ${config_name}..."
    source /dccstor/unified-trans/model_merging/granite33_2/mergekit/.venv/bin/activate
    mergekit-yaml "my_yaml/${config_name}.yml" \
        "merged_model/${config_name}" --cuda --trust-remote-code
    
    # Check if merge was successful
    if [[ $? -ne 0 ]]; then
        echo "Error: Merge failed for ${config_name}. Skipping evaluation..."
        continue
    fi
    
    echo "Merge completed successfully. Starting evaluation..."
    
    # Run evaluations
    source /dccstor/unified-trans/model_merging/granite33_2/evalplus/.venv/bin/activate
    
    # HumanEval
    echo "Running HumanEval evaluation..."
    evalplus.evaluate --model "merged_model/${config_name}" --backend vllm --dataset humaneval --greedy
    
    # MBPP
    echo "Running MBPP evaluation..."
    evalplus.evaluate --model "merged_model/${config_name}" --backend vllm --dataset mbpp --greedy
    
    echo "Evaluation complete for ${config_name}."
    echo ""
done

echo "All hyperparameter configurations tested!"

# Optional: Create a summary report
echo "Creating summary report..."
cat > "hyperparameter_summary.txt" << EOF
Hyperparameter Testing Summary
Generated on: $(date)

Configurations tested:
EOF

for hyperparam in "${hyperparams[@]}"; do
    read -r density1 weight1 density2 weight2 config_name <<< "$hyperparam"
    echo "- ${config_name}: Model1(d=${density1}, w=${weight1}), Model2(d=${density2}, w=${weight2})" >> "hyperparameter_summary.txt"
done

echo "Summary report saved to hyperparameter_summary.txt"
