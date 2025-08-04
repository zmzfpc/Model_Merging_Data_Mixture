#!/bin/bash

# Multi-Model Weight Analysis Runner
# This script helps run the multi-model weight analyzer with common configurations

set -e

# Default paths - update these for your setup
BASE_MODEL_DEFAULT="/path/to/your/huggingface/hub/models--ibm-granite--granite-8b-code-base"
OUTPUT_DIR_DEFAULT="multi_model_weight_analysis_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Multi-Model Weight Analysis Runner${NC}"
echo "=================================="

# Function to check if a path exists
check_path() {
    if [ ! -d "$1" ]; then
        echo -e "${RED}❌ Path does not exist: $1${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ Found: $1${NC}"
    return 0
}

# Function to find models in saves directory
find_sft_models() {
    local saves_dir="saves"
    local models=()
    
    if [ -d "$saves_dir" ]; then
        echo -e "${BLUE}🔍 Searching for SFT models in $saves_dir...${NC}"
        
        # Look for common SFT model patterns
        for model_dir in "$saves_dir"/*; do
            if [ -d "$model_dir" ] && [ -f "$model_dir/config.json" ]; then
                model_name=$(basename "$model_dir")
                echo -e "${GREEN}  Found: $model_name${NC}"
                models+=("$model_dir")
            fi
        done
    fi
    
    # Return the models array
    printf '%s\n' "${models[@]}"
}

# Function to run analysis
run_analysis() {
    local base_model="$1"
    local output_dir="$2"
    shift 2
    local sft_models=("$@")
    
    echo -e "${BLUE}🚀 Starting analysis...${NC}"
    echo "Base model: $base_model"
    echo "SFT models: ${sft_models[*]}"
    echo "Output directory: $output_dir"
    
    # Build command
    local cmd="python multi_model_weight_analyzer.py --base_model \"$base_model\" --sft_models"
    for model in "${sft_models[@]}"; do
        cmd="$cmd \"$model\""
    done
    cmd="$cmd --output \"$output_dir\""
    
    echo -e "${YELLOW}Command: $cmd${NC}"
    
    # Run the analysis
    if eval "$cmd"; then
        echo -e "${GREEN}✅ Analysis completed successfully!${NC}"
        echo -e "${GREEN}📁 Results saved to: $output_dir${NC}"
        
        # Show what was generated
        echo -e "${BLUE}📊 Generated files:${NC}"
        find "$output_dir" -name "*.png" -o -name "*.csv" -o -name "*.txt" | head -10
        if [ $(find "$output_dir" -name "*.png" -o -name "*.csv" -o -name "*.txt" | wc -l) -gt 10 ]; then
            echo "  ... and more"
        fi
    else
        echo -e "${RED}❌ Analysis failed!${NC}"
        exit 1
    fi
}

# Main script logic
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage modes:${NC}"
    echo "1. Auto-discover models: $0 auto [base_model_path]"
    echo "2. Manual specification: $0 manual base_model sft_model1 sft_model2 ..."
    echo "3. Interactive mode: $0 interactive"
    echo ""
    echo "Examples:"
    echo "  $0 auto"
    echo "  $0 manual /path/to/base saves/model1 saves/model2"
    echo "  $0 interactive"
    exit 1
fi

MODE="$1"
shift

case "$MODE" in
    "auto")
        echo -e "${BLUE}🔍 Auto-discovery mode${NC}"
        
        # Use provided base model or default
        if [ $# -gt 0 ]; then
            BASE_MODEL="$1"
        else
            BASE_MODEL="$BASE_MODEL_DEFAULT"
        fi
        
        echo "Base model: $BASE_MODEL"
        
        if ! check_path "$BASE_MODEL"; then
            echo -e "${RED}Please provide a valid base model path${NC}"
            exit 1
        fi
        
        # Find SFT models
        mapfile -t SFT_MODELS < <(find_sft_models)
        
        if [ ${#SFT_MODELS[@]} -lt 1 ]; then
            echo -e "${RED}❌ No SFT models found in saves/ directory${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}Found ${#SFT_MODELS[@]} SFT models${NC}"
        
        # Limit to first 5 models to avoid memory issues
        if [ ${#SFT_MODELS[@]} -gt 5 ]; then
            echo -e "${YELLOW}⚠️  Limiting to first 5 models to avoid memory issues${NC}"
            SFT_MODELS=("${SFT_MODELS[@]:0:5}")
        fi
        
        run_analysis "$BASE_MODEL" "$OUTPUT_DIR_DEFAULT" "${SFT_MODELS[@]}"
        ;;
        
    "manual")
        echo -e "${BLUE}📝 Manual specification mode${NC}"
        
        if [ $# -lt 2 ]; then
            echo -e "${RED}❌ Need at least base_model and one sft_model${NC}"
            echo "Usage: $0 manual base_model sft_model1 [sft_model2 ...]"
            exit 1
        fi
        
        BASE_MODEL="$1"
        shift
        SFT_MODELS=("$@")
        
        # Check all paths
        if ! check_path "$BASE_MODEL"; then
            exit 1
        fi
        
        for model in "${SFT_MODELS[@]}"; do
            if ! check_path "$model"; then
                exit 1
            fi
        done
        
        run_analysis "$BASE_MODEL" "$OUTPUT_DIR_DEFAULT" "${SFT_MODELS[@]}"
        ;;
        
    "interactive")
        echo -e "${BLUE}🎯 Interactive mode${NC}"
        
        # Get base model
        echo -n "Enter base model path (or press Enter for default): "
        read -r user_base
        if [ -z "$user_base" ]; then
            BASE_MODEL="$BASE_MODEL_DEFAULT"
        else
            BASE_MODEL="$user_base"
        fi
        
        if ! check_path "$BASE_MODEL"; then
            exit 1
        fi
        
        # Get SFT models
        SFT_MODELS=()
        echo "Enter SFT model paths (one per line, empty line to finish):"
        while true; do
            echo -n "SFT model ${#SFT_MODELS[@]}: "
            read -r sft_path
            if [ -z "$sft_path" ]; then
                break
            fi
            if check_path "$sft_path"; then
                SFT_MODELS+=("$sft_path")
            else
                echo -e "${YELLOW}⚠️  Skipping invalid path${NC}"
            fi
        done
        
        if [ ${#SFT_MODELS[@]} -eq 0 ]; then
            echo -e "${RED}❌ No valid SFT models provided${NC}"
            exit 1
        fi
        
        # Get output directory
        echo -n "Output directory (press Enter for default): "
        read -r user_output
        if [ -z "$user_output" ]; then
            OUTPUT_DIR="$OUTPUT_DIR_DEFAULT"
        else
            OUTPUT_DIR="$user_output"
        fi
        
        run_analysis "$BASE_MODEL" "$OUTPUT_DIR" "${SFT_MODELS[@]}"
        ;;
        
    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        echo "Valid modes: auto, manual, interactive"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Analysis complete!${NC}"
