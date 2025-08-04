#!/bin/bash

cd /dccstor/unified-trans/model_merging/granite33_2/mergekit/

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Define different hyperparameter configurations for DARE method
declare -A dare_configs=(
    ["dare_qw2_gs"]="Original configuration"
    ["dare_qw2_gs_density05"]="Density 0.5 configuration"
    ["dare_qw2_gs_density09"]="Density 0.9 configuration"
)

# You can also define configurations for other base models
base_model="qw2"

# Uncomment other configs as needed
yaml_files=(
    "dare_qw2_gs"
    "dare_qw2_gs_density05" 
    "dare_qw2_gs_density09"
    # Add more configurations here
)

echo "Starting model merging with ${#yaml_files[@]} different configurations..."

for i in "${!yaml_files[@]}"; do
    yaml_file="${yaml_files[$i]}"
    
    echo "=========================================="
    echo "Configuration $((i+1))/${#yaml_files[@]}: ${yaml_file}"
    echo "Description: ${dare_configs[$yaml_file]}"
    echo "=========================================="
    
    # Check if YAML file exists
    if [[ ! -f "my_yaml/${yaml_file}.yml" ]]; then
        echo "Warning: YAML file my_yaml/${yaml_file}.yml not found. Skipping..."
        continue
    fi
    
    echo "Merging model with YAML file: ${yaml_file}.yml"
    
    # Create output directory if it doesn't exist
    mkdir -p "merged_model/${yaml_file}"
    
    # Activate mergekit environment and run merge
    source /dccstor/unified-trans/model_merging/granite33_2/mergekit/.venv/bin/activate
    mergekit-yaml my_yaml/${yaml_file}.yml \
        merged_model/${yaml_file} --cuda --trust-remote-code
    
    # Check if merge was successful
    if [[ $? -ne 0 ]]; then
        echo "Error: Merge failed for ${yaml_file}. Skipping evaluation..."
        continue
    fi
    
    echo "Merge completed successfully. Starting evaluation..."
    
    # Activate evalplus environment and run evaluations
    source /dccstor/unified-trans/model_merging/granite33_2/evalplus/.venv/bin/activate
    
    # HumanEval evaluation
    echo "Running HumanEval evaluation for ${yaml_file}..."
    evalplus.evaluate --model merged_model/${yaml_file} --backend vllm --dataset humaneval --greedy
    
    if [[ $? -eq 0 ]]; then
        echo "HumanEval evaluation completed successfully."
    else
        echo "Warning: HumanEval evaluation failed for ${yaml_file}."
    fi
    
    # MBPP evaluation
    echo "Running MBPP evaluation for ${yaml_file}..."
    evalplus.evaluate --model merged_model/${yaml_file} --backend vllm --dataset mbpp --greedy
    
    if [[ $? -eq 0 ]]; then
        echo "MBPP evaluation completed successfully."
    else
        echo "Warning: MBPP evaluation failed for ${yaml_file}."
    fi
    
    echo "Evaluation complete for ${yaml_file}."
    echo ""
done

echo "All configurations processed!"
