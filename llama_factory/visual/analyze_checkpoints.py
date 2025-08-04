#!/usr/bin/env python3
"""
Script to analyze and compare two SFT checkpoints.
Computes weight differences, overlaps, layer-wise distributions, and generates visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors import safe_open
from pathlib import Path
import torch
from scipy.stats import pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except:
    plt.style.use('default')
    print("Using default matplotlib style")

class CheckpointAnalyzer:
    def __init__(self, checkpoint1_path, checkpoint2_path, output_dir="analysis_results"):
        self.checkpoint1_path = Path(checkpoint1_path)
        self.checkpoint2_path = Path(checkpoint2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model configurations
        self.config1 = self.load_config(self.checkpoint1_path)
        self.config2 = self.load_config(self.checkpoint2_path)
        
        # Load weight mappings
        self.weight_map1 = self.load_weight_map(self.checkpoint1_path)
        self.weight_map2 = self.load_weight_map(self.checkpoint2_path)
        
        # Results storage
        self.layer_analysis = {}
        self.overall_stats = {}
        
    def load_config(self, checkpoint_path):
        """Load model configuration"""
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_weight_map(self, checkpoint_path):
        """Load weight mapping from safetensors index"""
        index_path = checkpoint_path / "model.safetensors.index.json"
        with open(index_path, 'r') as f:
            return json.load(f)["weight_map"]
    
    def load_weights(self, checkpoint_path, weight_map):
        """Load all weights from safetensors files"""
        weights = {}
        
        # Get unique safetensors files
        safetensor_files = set(weight_map.values())
        
        for file in safetensor_files:
            file_path = checkpoint_path / file
            print(f"Loading {file}...")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in weight_map:
                        weights[key] = f.get_tensor(key)
        
        return weights
    
    def compute_weight_differences(self, weights1, weights2):
        """Compute various weight difference metrics"""
        common_keys = set(weights1.keys()) & set(weights2.keys())
        print(f"Found {len(common_keys)} common weight tensors")
        
        results = {}
        
        for key in common_keys:
            w1, w2 = weights1[key], weights2[key]
            
            if w1.shape != w2.shape:
                print(f"Warning: Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
                continue
            
            # Compute differences
            diff = w1 - w2
            
            # Various metrics
            results[key] = {
                'l2_norm_diff': torch.norm(diff).item(),
                'l2_norm_w1': torch.norm(w1).item(),
                'l2_norm_w2': torch.norm(w2).item(),
                'relative_diff': (torch.norm(diff) / torch.norm(w1)).item(),
                'cosine_sim': torch.nn.functional.cosine_similarity(
                    w1.flatten().unsqueeze(0), 
                    w2.flatten().unsqueeze(0)
                ).item(),
                'mean_abs_diff': torch.mean(torch.abs(diff)).item(),
                'std_diff': torch.std(diff).item(),
                'max_abs_diff': torch.max(torch.abs(diff)).item(),
                'shape': list(w1.shape),
                'num_params': w1.numel()
            }
        
        return results
    
    def analyze_by_layer(self, weight_diffs):
        """Analyze differences by transformer layer"""
        layer_stats = defaultdict(lambda: {
            'l2_norm_diffs': [],
            'relative_diffs': [],
            'cosine_sims': [],
            'param_counts': [],
            'components': []
        })
        
        # Categorize weights by layer and component
        for key, stats in weight_diffs.items():
            if 'model.layers.' in key:
                # Extract layer number
                layer_num = int(key.split('.')[2])
                component = key.split('.')[3:]  # e.g., ['self_attn', 'q_proj', 'weight']
                component_name = '.'.join(component)
                
                layer_stats[layer_num]['l2_norm_diffs'].append(stats['l2_norm_diff'])
                layer_stats[layer_num]['relative_diffs'].append(stats['relative_diff'])
                layer_stats[layer_num]['cosine_sims'].append(stats['cosine_sim'])
                layer_stats[layer_num]['param_counts'].append(stats['num_params'])
                layer_stats[layer_num]['components'].append(component_name)
            elif key in ['model.embed_tokens.weight', 'lm_head.weight']:
                # Special layers
                layer_stats[-1 if 'embed' in key else 999]['l2_norm_diffs'].append(stats['l2_norm_diff'])
                layer_stats[-1 if 'embed' in key else 999]['relative_diffs'].append(stats['relative_diff'])
                layer_stats[-1 if 'embed' in key else 999]['cosine_sims'].append(stats['cosine_sim'])
                layer_stats[-1 if 'embed' in key else 999]['param_counts'].append(stats['num_params'])
                layer_stats[-1 if 'embed' in key else 999]['components'].append(key)
        
        # Compute aggregated statistics per layer
        layer_summary = {}
        for layer_num, stats in layer_stats.items():
            layer_summary[layer_num] = {
                'mean_l2_diff': np.mean(stats['l2_norm_diffs']),
                'std_l2_diff': np.std(stats['l2_norm_diffs']),
                'mean_relative_diff': np.mean(stats['relative_diffs']),
                'mean_cosine_sim': np.mean(stats['cosine_sims']),
                'total_params': sum(stats['param_counts']),
                'num_components': len(stats['components']),
                'components': stats['components']
            }
        
        return layer_summary
    
    def create_summary_table(self, weight_diffs, layer_summary):
        """Create comprehensive summary tables"""
        
        # Overall statistics table
        overall_df = pd.DataFrame([
            {
                'Metric': 'Total Parameters',
                'Value': f"{sum([s['num_params'] for s in weight_diffs.values()]):,}",
                'Description': 'Total number of parameters compared'
            },
            {
                'Metric': 'Common Weights',
                'Value': len(weight_diffs),
                'Description': 'Number of weight tensors found in both checkpoints'
            },
            {
                'Metric': 'Mean L2 Difference',
                'Value': f"{np.mean([s['l2_norm_diff'] for s in weight_diffs.values()]):.6f}",
                'Description': 'Average L2 norm of weight differences'
            },
            {
                'Metric': 'Mean Relative Difference',
                'Value': f"{np.mean([s['relative_diff'] for s in weight_diffs.values()]):.6f}",
                'Description': 'Average relative difference (||diff|| / ||w1||)'
            },
            {
                'Metric': 'Mean Cosine Similarity',
                'Value': f"{np.mean([s['cosine_sim'] for s in weight_diffs.values()]):.6f}",
                'Description': 'Average cosine similarity between weights'
            }
        ])
        
        # Layer-wise summary table
        layer_data = []
        for layer_num, stats in sorted(layer_summary.items()):
            layer_name = 'Embedding' if layer_num == -1 else 'LM Head' if layer_num == 999 else f'Layer {layer_num}'
            layer_data.append({
                'Layer': layer_name,
                'Mean L2 Diff': f"{stats['mean_l2_diff']:.6f}",
                'Mean Relative Diff': f"{stats['mean_relative_diff']:.6f}",
                'Mean Cosine Sim': f"{stats['mean_cosine_sim']:.6f}",
                'Total Params': f"{stats['total_params']:,}",
                'Components': stats['num_components']
            })
        
        layer_df = pd.DataFrame(layer_data)
        
        # Component-wise analysis
        component_stats = defaultdict(list)
        for key, stats in weight_diffs.items():
            if 'model.layers.' in key:
                component = '.'.join(key.split('.')[3:])
                component_stats[component].append(stats)
        
        component_data = []
        for component, stat_list in component_stats.items():
            component_data.append({
                'Component': component,
                'Count': len(stat_list),
                'Mean L2 Diff': f"{np.mean([s['l2_norm_diff'] for s in stat_list]):.6f}",
                'Mean Relative Diff': f"{np.mean([s['relative_diff'] for s in stat_list]):.6f}",
                'Mean Cosine Sim': f"{np.mean([s['cosine_sim'] for s in stat_list]):.6f}",
                'Total Params': f"{sum([s['num_params'] for s in stat_list]):,}"
            })
        
        component_df = pd.DataFrame(component_data)
        
        return overall_df, layer_df, component_df
    
    def create_visualizations(self, weight_diffs, layer_summary):
        """Create comprehensive visualizations"""
        
        # Set style
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Layer-wise L2 differences
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Extract regular layers (0-27)
        regular_layers = {k: v for k, v in layer_summary.items() if 0 <= k <= 27}
        layer_nums = sorted(regular_layers.keys())
        
        # Plot 1: L2 differences by layer
        l2_diffs = [regular_layers[i]['mean_l2_diff'] for i in layer_nums]
        axes[0, 0].plot(layer_nums, l2_diffs, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean L2 Difference by Layer', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Layer Number')
        axes[0, 0].set_ylabel('Mean L2 Difference')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Relative differences by layer
        rel_diffs = [regular_layers[i]['mean_relative_diff'] for i in layer_nums]
        axes[0, 1].plot(layer_nums, rel_diffs, 'o-', color='orange', linewidth=2, markersize=6)
        axes[0, 1].set_title('Mean Relative Difference by Layer', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Layer Number')
        axes[0, 1].set_ylabel('Mean Relative Difference')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cosine similarities by layer
        cos_sims = [regular_layers[i]['mean_cosine_sim'] for i in layer_nums]
        axes[0, 2].plot(layer_nums, cos_sims, 'o-', color='green', linewidth=2, markersize=6)
        axes[0, 2].set_title('Mean Cosine Similarity by Layer', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Layer Number')
        axes[0, 2].set_ylabel('Mean Cosine Similarity')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Distribution of L2 differences
        all_l2_diffs = [s['l2_norm_diff'] for s in weight_diffs.values()]
        axes[1, 0].hist(all_l2_diffs, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution of L2 Differences', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('L2 Difference')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(np.mean(all_l2_diffs), color='red', linestyle='--', label=f'Mean: {np.mean(all_l2_diffs):.4f}')
        axes[1, 0].legend()
        
        # Plot 5: Distribution of cosine similarities
        all_cos_sims = [s['cosine_sim'] for s in weight_diffs.values()]
        axes[1, 1].hist(all_cos_sims, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Distribution of Cosine Similarities', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Cosine Similarity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(all_cos_sims), color='red', linestyle='--', label=f'Mean: {np.mean(all_cos_sims):.4f}')
        axes[1, 1].legend()
        
        # Plot 6: Parameter count by layer
        param_counts = [regular_layers[i]['total_params'] for i in layer_nums]
        axes[1, 2].bar(layer_nums, param_counts, alpha=0.7, color='purple')
        axes[1, 2].set_title('Parameter Count by Layer', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Layer Number')
        axes[1, 2].set_ylabel('Parameter Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'checkpoint_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Component-wise analysis
        component_stats = defaultdict(list)
        for key, stats in weight_diffs.items():
            if 'model.layers.' in key:
                component = key.split('.')[3]  # self_attn, mlp, etc.
                component_stats[component].append(stats['l2_norm_diff'])
        
        if component_stats:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            components = list(component_stats.keys())
            means = [np.mean(component_stats[comp]) for comp in components]
            stds = [np.std(component_stats[comp]) for comp in components]
            
            x_pos = np.arange(len(components))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Component')
            ax.set_ylabel('Mean L2 Difference')
            ax.set_title('L2 Differences by Component Type', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(components, rotation=45)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'component_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_detailed_results(self, weight_diffs, layer_summary, overall_df, layer_df, component_df):
        """Save detailed results to files"""
        
        # Save summary tables
        overall_df.to_csv(self.output_dir / 'overall_summary.csv', index=False)
        layer_df.to_csv(self.output_dir / 'layer_summary.csv', index=False)
        component_df.to_csv(self.output_dir / 'component_summary.csv', index=False)
        
        # Save detailed weight differences
        detailed_results = []
        for key, stats in weight_diffs.items():
            detailed_results.append({
                'weight_name': key,
                **stats
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(self.output_dir / 'detailed_weight_differences.csv', index=False)
        
        # Save layer summary as JSON
        with open(self.output_dir / 'layer_analysis.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_layer_summary = {}
            for k, v in layer_summary.items():
                json_layer_summary[str(k)] = {
                    key: float(val) if isinstance(val, (np.float32, np.float64)) else val 
                    for key, val in v.items() if key != 'components'
                }
            json.dump(json_layer_summary, f, indent=2)
        
        print(f"\nResults saved to {self.output_dir}/")
        print("Files generated:")
        print("- overall_summary.csv")
        print("- layer_summary.csv") 
        print("- component_summary.csv")
        print("- detailed_weight_differences.csv")
        print("- layer_analysis.json")
        print("- checkpoint_analysis.png")
        print("- component_analysis.png")
    
    def run_analysis(self):
        """Run the complete analysis"""
        print("=== Checkpoint Analysis ===")
        print(f"Checkpoint 1: {self.checkpoint1_path.name}")
        print(f"Checkpoint 2: {self.checkpoint2_path.name}")
        print()
        
        # Load weights
        print("Loading weights from checkpoint 1...")
        weights1 = self.load_weights(self.checkpoint1_path, self.weight_map1)
        print("Loading weights from checkpoint 2...")
        weights2 = self.load_weights(self.checkpoint2_path, self.weight_map2)
        
        # Compute differences
        print("\nComputing weight differences...")
        weight_diffs = self.compute_weight_differences(weights1, weights2)
        
        # Analyze by layer
        print("Analyzing by layer...")
        layer_summary = self.analyze_by_layer(weight_diffs)
        
        # Create summary tables
        print("Creating summary tables...")
        overall_df, layer_df, component_df = self.create_summary_table(weight_diffs, layer_summary)
        
        # Display tables
        print("\n=== OVERALL SUMMARY ===")
        print(overall_df.to_string(index=False))
        
        print("\n=== LAYER-WISE SUMMARY ===")
        print(layer_df.to_string(index=False))
        
        print("\n=== COMPONENT-WISE SUMMARY ===")
        print(component_df.to_string(index=False))
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_visualizations(weight_diffs, layer_summary)
        
        # Save results
        print("Saving detailed results...")
        self.save_detailed_results(weight_diffs, layer_summary, overall_df, layer_df, component_df)
        
        return weight_diffs, layer_summary

def main():
    # Define checkpoint paths
    checkpoint1_path = "saves/qwen25c7/best/sft_4o_sol_5e6"
    checkpoint2_path = "saves/qwen25c7/best/sft_ct_1e6"
    
    # Initialize analyzer
    analyzer = CheckpointAnalyzer(checkpoint1_path, checkpoint2_path)
    
    # Run analysis
    weight_diffs, layer_summary = analyzer.run_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Check the 'analysis_results' directory for detailed outputs!")

if __name__ == "__main__":
    main() 