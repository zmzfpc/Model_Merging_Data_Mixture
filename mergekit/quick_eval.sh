#!/bin/bash

# Quick evaluation script for testing specific merged models
# Usage: ./quick_eval.sh [model_name] [task_type]

set -e

# Default configurations
MODELS_DIR="merged_model"
RESULTS_DIR="quick_eval_results"
PYTHON_EVAL_SCRIPT="evaluate_models.py"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to show usage
show_usage() {
    echo "Usage: $0 [model_name] [task_type]"
    echo ""
    echo "model_name: Name of the model directory in $MODELS_DIR (optional)"
    echo "task_type: code_generation, general_nlp, reasoning, or all (default: code_generation)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Evaluate all models on code generation"
    echo "  $0 dare_dsc7_d0.3_w0.5               # Evaluate specific model on code generation"
    echo "  $0 dare_dsc7_d0.3_w0.5 all           # Evaluate specific model on all tasks"
    echo ""
}

# Parse arguments
MODEL_NAME=""
TASK_TYPE="code_generation"

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

if [ ! -z "$1" ]; then
    MODEL_NAME="$1"
fi

if [ ! -z "$2" ]; then
    TASK_TYPE="$2"
fi

echo "Starting quick evaluation..."
echo "Models directory: $MODELS_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Task type: $TASK_TYPE"

# Check if evaluation script exists
if [ ! -f "$PYTHON_EVAL_SCRIPT" ]; then
    echo "Error: $PYTHON_EVAL_SCRIPT not found"
    echo "Please make sure the evaluation script is in the current directory"
    exit 1
fi

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory $MODELS_DIR not found"
    echo "Please make sure merged models are in the $MODELS_DIR directory"
    exit 1
fi

# Install required packages if needed
echo "Checking Python dependencies..."
python3 -c "import pandas, json, logging" 2>/dev/null || {
    echo "Installing pandas..."
    pip install pandas
}

# Run evaluation
if [ ! -z "$MODEL_NAME" ]; then
    # Evaluate specific model
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Error: Model $MODEL_PATH not found"
        echo "Available models:"
        ls "$MODELS_DIR" | head -10
        exit 1
    fi
    
    echo "Evaluating model: $MODEL_NAME"
    
    if [ "$TASK_TYPE" = "all" ]; then
        python3 "$PYTHON_EVAL_SCRIPT" --model "$MODEL_PATH" --results_dir "$RESULTS_DIR"
    else
        python3 "$PYTHON_EVAL_SCRIPT" --model "$MODEL_PATH" --results_dir "$RESULTS_DIR" --tasks "$TASK_TYPE"
    fi
else
    # Evaluate all models
    echo "Evaluating all models in $MODELS_DIR"
    
    if [ "$TASK_TYPE" = "all" ]; then
        python3 "$PYTHON_EVAL_SCRIPT" --models_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR"
    else
        python3 "$PYTHON_EVAL_SCRIPT" --models_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR" --tasks "$TASK_TYPE"
    fi
fi

# Show results summary
echo ""
echo "Evaluation completed!"
echo ""

if [ -f "$RESULTS_DIR/evaluation_summary.csv" ]; then
    echo "Summary results:"
    echo "=================="
    head -5 "$RESULTS_DIR/evaluation_summary.csv" | column -t -s ','
    echo ""
    echo "Full results available in: $RESULTS_DIR/evaluation_summary.csv"
fi

if [ -f "$RESULTS_DIR/detailed_results.json" ]; then
    echo "Detailed results available in: $RESULTS_DIR/detailed_results.json"
fi

echo ""
echo "Log file: $RESULTS_DIR/../evaluation.log"
echo ""

# Quick performance check
if [ -f "$RESULTS_DIR/evaluation_summary.csv" ]; then
    echo "Quick performance summary:"
    echo "=========================="
    
    # Extract HumanEval scores if available
    if grep -q "humaneval" "$RESULTS_DIR/evaluation_summary.csv"; then
        echo "HumanEval scores:"
        grep -v "model_name" "$RESULTS_DIR/evaluation_summary.csv" | \
        awk -F, '{print $1 ": " $3}' | head -10
    fi
    
    echo ""
    echo "To see full results, run:"
    echo "  cat $RESULTS_DIR/evaluation_summary.csv | column -t -s ','"
fi
