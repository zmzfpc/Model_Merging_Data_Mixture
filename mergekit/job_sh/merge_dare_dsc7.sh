#!/bin/bash

cd /dccstor/unified-trans/model_merging/granite33_2/mergekit/

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Function to modify YAML parameters in-place using Python
modify_yaml_params() {
    local yaml_file=$1
    local model1_density=$2
    local model1_weight=$3
    local model2_density=$4
    local model2_weight=$5
    
    python3 << EOF
import yaml
import sys

yaml_file = "$yaml_file"
try:
    # Read the YAML file
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Modify parameters
    data['models'][0]['parameters']['density'] = float($model1_density)
    data['models'][0]['parameters']['weight'] = float($model1_weight)
    data['models'][1]['parameters']['density'] = float($model2_density)
    data['models'][1]['parameters']['weight'] = float($model2_weight)
    
    # Write back to file
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Updated {yaml_file} with new parameters")
    print(f"Model 1: density={$model1_density}, weight={$model1_weight}")
    print(f"Model 2: density={$model2_density}, weight={$model2_weight}")
    
except Exception as e:
    print(f"Error modifying YAML: {e}")
    sys.exit(1)
EOF
}

# Alternative function using sed (if Python/PyYAML is not available)
modify_yaml_params_sed() {
    local yaml_file=$1
    local model1_density=$2
    local model1_weight=$3
    local model2_density=$4
    local model2_weight=$5
    
    # Use sed to replace parameters
    # Replace first occurrence (model 1)
    sed -i "0,/density:/{s/density: .*/density: $model1_density/}" "$yaml_file"
    sed -i "0,/weight:/{s/weight: .*/weight: $model1_weight/}" "$yaml_file"
    
    # Replace second occurrence (model 2) 
    sed -i "0,/density: $model1_density/! {0,/density:/{s/density: .*/density: $model2_density/}}" "$yaml_file"
    sed -i "0,/weight: $model1_weight/! {0,/weight:/{s/weight: .*/weight: $model2_weight/}}" "$yaml_file"
    
    echo "Updated $yaml_file with sed"
    echo "Model 1: density=$model1_density, weight=$model1_weight"
    echo "Model 2: density=$model2_density, weight=$model2_weight"
}

# Base YAML file to modify
base_yaml="my_yaml/dare_dsc7_gs.yml"
original_yaml="${base_yaml}.original"

# Create backup of original YAML file
cp "$base_yaml" "$original_yaml"

# Define different hyperparameter combinations to test
# Format: "model1_density:model1_weight:model2_density:model2_weight:description"
hyperparams_combinations=(
    "0.5:0.1:0.5:0.9:1"
    "0.5:0.2:0.5:0.8:2"
    "0.5:0.3:0.5:0.7:3"
    "0.5:0.4:0.5:0.6:4"
    "0.5:0.5:0.5:0.5:5"
    "0.5:0.6:0.5:0.4:6"
    "0.5:0.7:0.5:0.3:7"
    "0.5:0.8:0.5:0.2:8"
    "0.5:0.9:0.5:0.1:9"
)

echo "Starting hyperparameter sweep with ${#hyperparams_combinations[@]} combinations..."

test_data="/dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/data/instruct_code_docstring_train/test.jsonl"
side="sentence-transformers/all-mpnet-base-v2"
batch=64


# Process each hyperparameter combination
for i in "${!hyperparams_combinations[@]}"; do
    
    IFS=':' read -r model1_density model1_weight model2_density model2_weight description <<< "${hyperparams_combinations[$i]}"
    
    # Create unique identifier for this combination
    experiment_id="exp${i}_d${model1_density//./_}_w${model1_weight//./_}_d${model2_density//./_}_w${model2_weight//./_}"
    output_dir="merged_model/dare_dsc7_gs_${experiment_id}"
    
    echo "=========================================="
    echo "Experiment $((i+1))/${#hyperparams_combinations[@]}: ${description}"
    echo "Model 1 - Density: ${model1_density}, Weight: ${model1_weight}"
    echo "Model 2 - Density: ${model2_density}, Weight: ${model2_weight}"
    echo "Output directory: ${output_dir}"
    echo "=========================================="
    
    # Restore original YAML and modify with current hyperparameters
    cp "$original_yaml" "$base_yaml"
    
    # Try Python method first, fall back to sed if needed
    if command -v python3 &> /dev/null && python3 -c "import yaml" 2>/dev/null; then
        modify_yaml_params "$base_yaml" "$model1_density" "$model1_weight" "$model2_density" "$model2_weight"
    else
        echo "PyYAML not available, using sed method..."
        modify_yaml_params_sed "$base_yaml" "$model1_density" "$model1_weight" "$model2_density" "$model2_weight"
    fi
    
    echo "Modified YAML file with new parameters"
    
    # Show current YAML content (first few lines)
    echo "Current YAML configuration:"
    head -15 "$base_yaml"
    echo ""
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Activate mergekit environment and run merge
    source /dccstor/unified-trans/model_merging/granite33_2/mergekit/.venv/bin/activate
    mergekit-yaml "$base_yaml" "$output_dir" --cuda --trust-remote-code
    
    if [ $? -eq 0 ]; then
        echo "✓ Merge successful for experiment $((i+1))"
        
        # Save the configuration used for this experiment
        cp "$base_yaml" "${output_dir}/mergekit_config.yml"
        
        # Activate evalplus environment and run evaluations
        source /dccstor/unified-trans/model_merging/granite33_2/evalplus/.venv/bin/activate
        
        echo "Starting HumanEval evaluation..."
        evalplus.evaluate --model "$output_dir" --backend vllm --dataset humaneval --greedy
        
        if [ $? -eq 0 ]; then
            echo "✓ HumanEval evaluation complete for experiment $((i+1))"
        else
            echo "✗ HumanEval evaluation failed for experiment $((i+1))"
        fi
        
        echo "Starting MBPP evaluation..."
        evalplus.evaluate --model "$output_dir" --backend vllm --dataset mbpp --greedy
        
        if [ $? -eq 0 ]; then
            echo "✓ MBPP evaluation complete for experiment $((i+1))"
        else
            echo "✗ MBPP evaluation failed for experiment $((i+1))"
        fi

        save="${output_dir}/eval_codesum_results.jsonl"

        python eval_codesum.py \
            --model "$output_dir" \
            --data "$test_data" \
            --side "$side" \
            --batch_size "$batch" \
            --bf16 --vllm \
            --save_path "$save"
        
    else
        echo "✗ Merge failed for experiment $((i+1)), skipping evaluation"
    fi

    rm -r "$output_dir"
    
    echo "Experiment $((i+1)) complete: ${description}"
    echo ""
done

# Restore original YAML file
cp "$original_yaml" "$base_yaml"
rm -f "$original_yaml"

echo "=========================================="
echo "All hyperparameter experiments completed!"
echo "Results saved in merged_model/dare_qw2_gs_exp* directories"
echo "Original YAML file restored"
echo "=========================================="
