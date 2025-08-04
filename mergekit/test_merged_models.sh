#!/bin/bash

cd /path/to/your/project/mergekit/

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/path/to/your/project/huggingface"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# Test data configurations
test_data="/path/to/your/project/LLaMA-Factory/data/instruct_code_docstring_train/test.jsonl"
side="sentence-transformers/all-mpnet-base-v2"
batch=64

# Function to test a single merged model
test_merged_model() {
    local model_path=$1
    local model_name=$2
    local log_file="test_results/${model_name}_test.log"
    
    echo "=========================================="
    echo "Testing merged model: ${model_name}"
    echo "Model path: ${model_path}"
    echo "Log file: ${log_file}"
    echo "=========================================="
    
    # Create results directory
    mkdir -p test_results
    
    # Check if model exists
    if [ ! -d "$model_path" ]; then
        echo "Error: Model directory not found: $model_path"
        echo "$(date): ERROR - Model directory not found: $model_path" >> "$log_file"
        return 1
    fi
    
    echo "$(date): Starting evaluation for $model_name" >> "$log_file"
    
    # Test 1: HumanEval evaluation
    echo "Starting HumanEval evaluation..."
    echo "$(date): Starting HumanEval evaluation" >> "$log_file"
    
    source /path/to/your/project/evalplus/.venv/bin/activate
    
    evalplus.evaluate --model "$model_path" --backend vllm --dataset humaneval --greedy 2>&1 | tee -a "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ HumanEval evaluation completed successfully"
        echo "$(date): HumanEval evaluation completed successfully" >> "$log_file"
    else
        echo "✗ HumanEval evaluation failed"
        echo "$(date): HumanEval evaluation failed" >> "$log_file"
    fi
    
    # Test 2: MBPP evaluation
    echo "Starting MBPP evaluation..."
    echo "$(date): Starting MBPP evaluation" >> "$log_file"
    
    evalplus.evaluate --model "$model_path" --backend vllm --dataset mbpp --greedy 2>&1 | tee -a "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ MBPP evaluation completed successfully"
        echo "$(date): MBPP evaluation completed successfully" >> "$log_file"
    else
        echo "✗ MBPP evaluation failed"
        echo "$(date): MBPP evaluation failed" >> "$log_file"
    fi
    
    # Test 3: Code Summarization evaluation
    echo "Starting Code Summarization evaluation..."
    echo "$(date): Starting Code Summarization evaluation" >> "$log_file"
    
    save_path="test_results/${model_name}_codesum_results.jsonl"
    
    python eval_codesum.py \
        --model "$model_path" \
        --data "$test_data" \
        --side "$side" \
        --batch_size "$batch" \
        --bf16 --vllm \
        --save_path "$save_path" 2>&1 | tee -a "$log_file"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Code Summarization evaluation completed successfully"
        echo "$(date): Code Summarization evaluation completed successfully" >> "$log_file"
        echo "Results saved to: $save_path"
    else
        echo "✗ Code Summarization evaluation failed"
        echo "$(date): Code Summarization evaluation failed" >> "$log_file"
    fi
    
    echo "$(date): Evaluation completed for $model_name" >> "$log_file"
    echo "Model evaluation completed: ${model_name}"
    echo ""
}

# Function to extract and summarize results
extract_results() {
    local model_name=$1
    local log_file="test_results/${model_name}_test.log"
    local summary_file="test_results/${model_name}_summary.txt"
    
    echo "Extracting results for: ${model_name}"
    
    # Extract HumanEval results
    humaneval_base=$(grep -A2 "humaneval (base tests)" "$log_file" | grep "pass@1:" | awk '{print $2}' | head -1)
    humaneval_plus=$(grep -A2 "humaneval+ (base + extra tests)" "$log_file" | grep "pass@1:" | awk '{print $2}' | head -1)
    
    # Extract MBPP results
    mbpp_base=$(grep -A2 "mbpp (base tests)" "$log_file" | grep "pass@1:" | awk '{print $2}' | head -1)
    mbpp_plus=$(grep -A2 "mbpp+ (base + extra tests)" "$log_file" | grep "pass@1:" | awk '{print $2}' | head -1)
    
    # Create summary
    cat > "$summary_file" << EOF
=== Model Evaluation Summary ===
Model: ${model_name}
Date: $(date)

=== HumanEval Results ===
HumanEval (base): ${humaneval_base:-"N/A"}
HumanEval+: ${humaneval_plus:-"N/A"}

=== MBPP Results ===
MBPP (base): ${mbpp_base:-"N/A"}
MBPP+: ${mbpp_plus:-"N/A"}

=== Code Summarization ===
Results file: test_results/${model_name}_codesum_results.jsonl

=== Detailed Logs ===
Full log: ${log_file}
EOF
    
    echo "Summary saved to: $summary_file"
}

# Main testing function
main() {
    echo "=========================================="
    echo "Merged Model Evaluation Script"
    echo "Started: $(date)"
    echo "=========================================="
    
    # Define base models and merge methods
    base_models=("dsc7" "dsc13" "qwc7" "qwc15")
    merge_methods=("linear" "ties" "dare" "della")
    
    # Test all merged models
    total_models=0
    successful_tests=0
    
    for base_model in "${base_models[@]}"; do
        for method in "${merge_methods[@]}"; do
            model_name="${method}_${base_model}_gs"
            model_path="merged_model/${model_name}"
            
            echo "Checking model: $model_name"
            
            if [ -d "$model_path" ]; then
                total_models=$((total_models + 1))
                
                test_merged_model "$model_path" "$model_name"
                
                if [ $? -eq 0 ]; then
                    successful_tests=$((successful_tests + 1))
                    extract_results "$model_name"
                else
                    echo "Failed to test model: $model_name"
                fi
            else
                echo "Model not found: $model_path"
            fi
        done
    done
    
    # Generate overall summary
    echo "=========================================="
    echo "Testing completed!"
    echo "Total models found: $total_models"
    echo "Successful tests: $successful_tests"
    echo "Failed tests: $((total_models - successful_tests))"
    echo "Results directory: test_results/"
    echo "=========================================="
    
    # Create consolidated summary
    consolidated_summary="test_results/consolidated_summary.csv"
    echo "model_name,humaneval_base,humaneval_plus,mbpp_base,mbpp_plus" > "$consolidated_summary"
    
    for base_model in "${base_models[@]}"; do
        for method in "${merge_methods[@]}"; do
            model_name="${method}_${base_model}_gs"
            summary_file="test_results/${model_name}_summary.txt"
            
            if [ -f "$summary_file" ]; then
                humaneval_base=$(grep "HumanEval (base):" "$summary_file" | awk '{print $3}')
                humaneval_plus=$(grep "HumanEval+:" "$summary_file" | awk '{print $2}')
                mbpp_base=$(grep "MBPP (base):" "$summary_file" | awk '{print $3}')
                mbpp_plus=$(grep "MBPP+:" "$summary_file" | awk '{print $2}')
                
                echo "${model_name},${humaneval_base:-N/A},${humaneval_plus:-N/A},${mbpp_base:-N/A},${mbpp_plus:-N/A}" >> "$consolidated_summary"
            fi
        done
    done
    
    echo "Consolidated summary saved to: $consolidated_summary"
}

# Check if specific model is provided as argument
if [ $# -eq 1 ]; then
    model_name="$1"
    model_path="merged_model/${model_name}"
    
    echo "Testing specific model: $model_name"
    test_merged_model "$model_path" "$model_name"
    extract_results "$model_name"
else
    # Test all models
    main
fi

echo "Evaluation script completed at: $(date)"
