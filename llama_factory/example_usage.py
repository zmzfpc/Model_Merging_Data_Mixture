#!/usr/bin/env python3
"""
Example usage of the Universal Model Weight Difference Visualizer

This script demonstrates how to use the visualize_model_diff.py tool
with different model pairs and configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_analysis(model1_path, model2_path, output_dir, description=""):
    """Run the model difference analysis"""
    
    print(f"\n{'='*60}")
    print(f"🔍 ANALYSIS: {description}")
    print(f"{'='*60}")
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Output:  {output_dir}")
    print()
    
    cmd = [
        sys.executable, "visualize_model_diff.py",
        "--model1", model1_path,
        "--model2", model2_path,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Analysis completed successfully!")
        
        # Check output files
        output_path = Path(output_dir)
        if output_path.exists():
            files = list(output_path.glob("*"))
            print(f"📁 Generated {len(files)} files:")
            for file in sorted(files):
                print(f"   - {file.name}")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except FileNotFoundError:
        print("❌ visualize_model_diff.py not found. Make sure it's in the current directory.")

def main():
    """Run example analyses"""
    
    print("🚀 Universal Model Diff Visualizer - Usage Examples")
    print("="*60)
    
    # Example 1: Compare the two SFT checkpoints (already exists)
    print("\nExample 1: Comparing existing SFT checkpoints")
    run_analysis(
        model1_path="saves/qwen25c7/best/sft_4o_sol_5e6",
        model2_path="saves/qwen25c7/best/sft_ct_1e6", 
        output_dir="comparison_sft_models",
        description="SFT Models: GPT-4 Solutions vs Custom Task"
    )
    
    # Example 2: Instructions for other model comparisons
    print(f"\n{'='*60}")
    print("📚 MORE USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "description": "Compare two different LoRA adaptations",
            "cmd": "python visualize_model_diff.py --model1 path/to/lora1 --model2 path/to/lora2 --output lora_comparison"
        },
        {
            "description": "Compare different training epochs",
            "cmd": "python visualize_model_diff.py --model1 model/epoch_1 --model2 model/epoch_5 --output epoch_comparison"
        },
        {
            "description": "Compare different hyperparameter settings",
            "cmd": "python visualize_model_diff.py --model1 model/lr_1e-5 --model2 model/lr_5e-6 --output lr_comparison"
        },
        {
            "description": "Compare different fine-tuning datasets",
            "cmd": "python visualize_model_diff.py --model1 model/dataset_A --model2 model/dataset_B --output dataset_comparison"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i+1}: {example['description']}")
        print(f"Command: {example['cmd']}")
    
    print(f"\n{'='*60}")
    print("📊 OUTPUT FILES EXPLANATION")
    print("="*60)
    
    outputs = [
        "layer_summary.csv - Layer-by-layer statistics and metrics",
        "detailed_analysis.csv - Complete weight-level analysis data", 
        "config_comparison.json - Model configuration differences",
        "comprehensive_analysis.png - 14-panel visualization dashboard"
    ]
    
    for output in outputs:
        print(f"📄 {output}")
    
    print(f"\n{'='*60}")
    print("🎯 KEY FEATURES")
    print("="*60)
    
    features = [
        "🔍 Automatic architecture detection (Qwen2, LLaMA, etc.)",
        "📊 Layer-wise difference analysis with heatmaps",
        "🧠 Component breakdown (MLP, Attention, LayerNorm)",
        "📈 Trend analysis with smoothing and error bars", 
        "🎨 Comprehensive 14-panel visualization dashboard",
        "💾 CSV exports for further analysis",
        "⚡ Memory efficient safetensors loading",
        "🎯 Statistical insights and pattern detection"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n✨ Ready to analyze your model pairs!")
    print("Simply run: python visualize_model_diff.py --model1 <path1> --model2 <path2>")

if __name__ == "__main__":
    main() 