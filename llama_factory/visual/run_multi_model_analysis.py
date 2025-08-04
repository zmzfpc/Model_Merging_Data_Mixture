#!/usr/bin/env python3
"""
Example usage script for the Multi-Model Weight Analyzer

This script demonstrates how to use the multi_model_weight_analyzer.py
with typical model paths and configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_analysis_example():
    """Run an example analysis with placeholder paths"""
    
    # Example paths - replace these with your actual model paths
    base_model = "/path/to/base/model"  # e.g., "ibm-granite/granite-8b-code-base"
    sft_models = [
        "/path/to/sft/model1",  # e.g., "saves/granite-8b-kod-sft"
        "/path/to/sft/model2",  # e.g., "saves/granite-8b-ct-sft"
        "/path/to/sft/model3",  # e.g., "saves/granite-8b-oc-sft"
    ]
    output_dir = "weight_analysis_results"
    
    # Build command
    cmd = [
        "python", "multi_model_weight_analyzer.py",
        "--base_model", base_model,
        "--sft_models"] + sft_models + [
        "--output", output_dir
    ]
    
    print("Running multi-model weight analysis...")
    print(f"Command: {' '.join(cmd)}")
    
    # Check if paths exist
    if not Path(base_model).exists():
        print(f"❌ Base model path does not exist: {base_model}")
        print("Please update the base_model path in this script.")
        return
    
    for i, sft_model in enumerate(sft_models):
        if not Path(sft_model).exists():
            print(f"❌ SFT model {i+1} path does not exist: {sft_model}")
            print("Please update the sft_models paths in this script.")
            return
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

def run_granite_example():
    """Example for Granite models in the current workspace"""
    
    # Look for Granite models in common locations
    possible_base_paths = [
        "updated_model/granite-8b-code-base",
        "saves/granite-8b-base",
        "/path/to/your/huggingface/hub/models--ibm-granite--granite-8b-code-base/snapshots",
    ]
    
    possible_sft_paths = [
        "saves/granite-8b-kod-sft",
        "saves/granite-8b-ct-sft", 
        "saves/granite-8b-oc-sft",
        "saves/qwen-kod-sft",
        "saves/deepseek-kod-sft",
    ]
    
    # Find existing models
    existing_base = None
    for base_path in possible_base_paths:
        if Path(base_path).exists():
            existing_base = base_path
            break
    
    existing_sft = []
    for sft_path in possible_sft_paths:
        if Path(sft_path).exists():
            existing_sft.append(sft_path)
    
    if existing_base and len(existing_sft) >= 2:
        print(f"Found base model: {existing_base}")
        print(f"Found SFT models: {existing_sft}")
        
        cmd = [
            "python", "multi_model_weight_analyzer.py",
            "--base_model", existing_base,
            "--sft_models"] + existing_sft[:3] + [  # Limit to first 3 SFT models
            "--output", "granite_weight_analysis"
        ]
        
        print("Running Granite model analysis...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True)
            print("✅ Granite analysis completed successfully!")
            print("Results saved to: granite_weight_analysis")
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed: {e}")
    else:
        print("❌ Could not find sufficient Granite models in expected locations.")
        print(f"Base models searched: {possible_base_paths}")
        print(f"SFT models searched: {possible_sft_paths}")
        print("Please update paths or ensure models are available.")

def main():
    print("Multi-Model Weight Analyzer - Example Usage")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "granite":
        run_granite_example()
    else:
        print("Usage options:")
        print("1. python example_usage.py granite     # Try to find Granite models automatically")
        print("2. Edit this script to set your own model paths and run directly")
        print()
        
        choice = input("Run example with placeholder paths? (y/n): ").lower().strip()
        if choice == 'y':
            print("\n⚠️  Note: You'll need to edit the paths in this script before running.")
            run_analysis_example()
        else:
            print("Please edit the model paths in this script and run again.")

if __name__ == "__main__":
    main()
