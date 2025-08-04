#!/usr/bin/env python3
"""
Universal Model Weight Difference Visualizer

A comprehensive tool to analyze and visualize weight differences between any two
fine-tuned models that share the same base architecture.

Usage:
    python visualize_model_diff.py --model1 path/to/model1 --model2 path/to/model2
    python visualize_model_diff.py --model1 path/to/model1 --model2 path/to/model2 --output results_dir
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from safetensors import safe_open
from collections import defaultdict
import warnings
from typing import Dict, Tuple, List, Optional
import re
from scipy.stats import skew, kurtosis, ks_2samp
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ModelDiffVisualizer:
    def __init__(self, model1_path: str, model2_path: str, output_dir: str = "model_diff_analysis"):
        self.model1_path = Path(model1_path)
        self.model2_path = Path(model2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "layer_by_layer").mkdir(exist_ok=True)
        (self.output_dir / "weight_distributions").mkdir(exist_ok=True)
        (self.output_dir / "neuron_analysis").mkdir(exist_ok=True)
        
        print(f"🔍 Analyzing models:")
        print(f"   Model 1: {self.model1_path.name}")
        print(f"   Model 2: {self.model2_path.name}")
        print(f"   Output: {self.output_dir}")
        
        # Load configurations and detect architecture
        self.config1 = self._load_config(self.model1_path)
        self.config2 = self._load_config(self.model2_path)
        self.architecture = self._detect_architecture()
        
        # Load weight mappings
        self.weight_map1 = self._load_weight_map(self.model1_path)
        self.weight_map2 = self._load_weight_map(self.model2_path)
        
        print(f"📐 Detected architecture: {self.architecture}")
        print(f"🔢 Model has {self.config1.get('num_hidden_layers', 'unknown')} layers")

    def _load_config(self, model_path: Path) -> dict:
        """Load model configuration"""
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_weight_map(self, model_path: Path) -> dict:
        """Load weight mapping from safetensors index"""
        index_path = model_path / "model.safetensors.index.json"
        with open(index_path, 'r') as f:
            return json.load(f)["weight_map"]

    def _detect_architecture(self) -> str:
        """Detect model architecture from config"""
        if "architectures" in self.config1:
            arch = self.config1["architectures"][0]
            return arch
        elif "model_type" in self.config1:
            return self.config1["model_type"]
        else:
            return "unknown"

    def load_weights(self, model_path: Path, weight_map: dict) -> dict:
        """Load all weights from safetensors files"""
        weights = {}
        safetensor_files = set(weight_map.values())
        
        for i, file in enumerate(safetensor_files, 1):
            file_path = model_path / file
            print(f"   Loading {file} ({i}/{len(safetensor_files)})...")
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in weight_map:
                        weights[key] = f.get_tensor(key)
        
        return weights

    def compute_weight_differences(self, weights1: dict, weights2: dict) -> dict:
        """Compute comprehensive weight difference metrics"""
        common_keys = set(weights1.keys()) & set(weights2.keys())
        print(f"🔍 Found {len(common_keys)} common weight tensors")
        
        results = {}
        
        for key in common_keys:
            w1, w2 = weights1[key], weights2[key]
            
            if w1.shape != w2.shape:
                print(f"⚠️  Shape mismatch for {key}: {w1.shape} vs {w2.shape}")
                continue
            
            # Compute differences
            diff = w1 - w2
            
            # Comprehensive metrics
            results[key] = {
                'l2_norm_diff': torch.norm(diff).item(),
                'l2_norm_w1': torch.norm(w1).item(),
                'l2_norm_w2': torch.norm(w2).item(),
                'relative_diff': (torch.norm(diff) / (torch.norm(w1) + 1e-8)).item(),
                'cosine_sim': torch.nn.functional.cosine_similarity(
                    w1.flatten().unsqueeze(0), 
                    w2.flatten().unsqueeze(0)
                ).item(),
                'mean_abs_diff': torch.mean(torch.abs(diff)).item(),
                'std_diff': torch.std(diff).item(),
                'max_abs_diff': torch.max(torch.abs(diff)).item(),
                'frobenius_norm_diff': torch.norm(diff, p='fro').item(),
                'shape': list(w1.shape),
                'num_params': w1.numel(),
                'sparsity_diff': (torch.sum(torch.abs(diff) > 1e-6).item() / w1.numel()),
                # Store raw tensors for detailed analysis
                'weights1': w1,
                'weights2': w2,
                'diff': diff
            }
        
        return results

    def analyze_weight_distributions(self, weight_diffs: dict) -> dict:
        """Analyze weight distributions for each layer and component"""
        print("🔬 Analyzing weight distributions...")
        
        distribution_analysis = {}
        
        for weight_name, stats in weight_diffs.items():
            w1, w2, diff = stats['weights1'], stats['weights2'], stats['diff']
            
            # Compute distribution statistics
            dist_stats = {
                'weight_name': weight_name,
                'shape': stats['shape'],
                # Model 1 distribution
                'w1_mean': torch.mean(w1).item(),
                'w1_std': torch.std(w1).item(),
                'w1_min': torch.min(w1).item(),
                'w1_max': torch.max(w1).item(),
                'w1_median': torch.median(w1).item(),
                'w1_skewness': skew(w1.flatten().float().numpy()),
                'w1_kurtosis': kurtosis(w1.flatten().float().numpy()),
                # Model 2 distribution
                'w2_mean': torch.mean(w2).item(),
                'w2_std': torch.std(w2).item(),
                'w2_min': torch.min(w2).item(),
                'w2_max': torch.max(w2).item(),
                'w2_median': torch.median(w2).item(),
                'w2_skewness': skew(w2.flatten().float().numpy()),
                'w2_kurtosis': kurtosis(w2.flatten().float().numpy()),
                # Difference distribution
                'diff_mean': torch.mean(diff).item(),
                'diff_std': torch.std(diff).item(),
                'diff_min': torch.min(diff).item(),
                'diff_max': torch.max(diff).item(),
                'diff_median': torch.median(diff).item(),
                'diff_skewness': skew(diff.flatten().float().numpy()),
                'diff_kurtosis': kurtosis(diff.flatten().float().numpy()),
                # Statistical tests
                'ks_statistic': ks_2samp(w1.flatten().float().numpy(), w2.flatten().float().numpy()).statistic,
                'ks_pvalue': ks_2samp(w1.flatten().float().numpy(), w2.flatten().float().numpy()).pvalue,
            }
            
            # Add neuron-level analysis for 2D weights (matrices)
            if len(w1.shape) == 2:
                dist_stats.update(self._analyze_neuron_level(w1, w2, diff, weight_name))
            
            distribution_analysis[weight_name] = dist_stats
        
        return distribution_analysis

    def _analyze_neuron_level(self, w1: torch.Tensor, w2: torch.Tensor, diff: torch.Tensor, weight_name: str) -> dict:
        """Analyze individual neurons (rows/columns) in weight matrices"""
        
        neuron_stats = {}
        
        # Analyze by output neurons (rows)
        if w1.shape[0] > 1:
            out_neuron_diffs = torch.norm(diff, dim=1)  # L2 norm per output neuron
            neuron_stats.update({
                'out_neuron_mean_diff': torch.mean(out_neuron_diffs).item(),
                'out_neuron_std_diff': torch.std(out_neuron_diffs).item(),
                'out_neuron_max_diff': torch.max(out_neuron_diffs).item(),
                'out_neuron_min_diff': torch.min(out_neuron_diffs).item(),
                'out_neuron_max_idx': torch.argmax(out_neuron_diffs).item(),
                'out_neuron_min_idx': torch.argmin(out_neuron_diffs).item(),
            })
        
        # Analyze by input neurons (columns)
        if w1.shape[1] > 1:
            in_neuron_diffs = torch.norm(diff, dim=0)  # L2 norm per input neuron
            neuron_stats.update({
                'in_neuron_mean_diff': torch.mean(in_neuron_diffs).item(),
                'in_neuron_std_diff': torch.std(in_neuron_diffs).item(),
                'in_neuron_max_diff': torch.max(in_neuron_diffs).item(),
                'in_neuron_min_diff': torch.min(in_neuron_diffs).item(),
                'in_neuron_max_idx': torch.argmax(in_neuron_diffs).item(),
                'in_neuron_min_idx': torch.argmin(in_neuron_diffs).item(),
            })
        
        return neuron_stats

    def create_weight_distribution_plots(self, weight_diffs: dict, layer_data: dict):
        """Create detailed weight distribution plots for each layer"""
        print("🎨 Creating weight distribution plots...")
        
        for layer_num in sorted(layer_data.keys()):
            if layer_num < 0 or layer_num >= 900:  # Skip special layers for now
                continue
                
            layer_info = layer_data[layer_num]
            self._create_layer_distribution_plot(layer_num, layer_info, weight_diffs)
    
    def _create_layer_distribution_plot(self, layer_num: int, layer_info: dict, weight_diffs: dict):
        """Create comprehensive distribution plot for a single layer"""
        
        # Collect all weights for this layer
        layer_weights = []
        for comp_type, comp_weights in layer_info.items():
            for weight_info in comp_weights:
                weight_name = weight_info['name']
                if weight_name in weight_diffs:
                    layer_weights.append((weight_name, weight_diffs[weight_name], comp_type))
        
        if not layer_weights:
            return
        
        # Create figure with subplots
        n_weights = len(layer_weights)
        n_cols = min(3, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        if n_weights == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_weights == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (weight_name, weight_data, comp_type) in enumerate(layer_weights):
            if i >= len(axes):
                break
                
            ax = axes[i]
            w1, w2, diff = weight_data['weights1'], weight_data['weights2'], weight_data['diff']
            
            # Create distribution comparison
            w1_flat = w1.flatten().float().numpy()
            w2_flat = w2.flatten().float().numpy()
            diff_flat = diff.flatten().float().numpy()
            
            # Plot distributions
            ax.hist(w1_flat, bins=50, alpha=0.6, label='Model 1', color='blue', density=True)
            ax.hist(w2_flat, bins=50, alpha=0.6, label='Model 2', color='red', density=True)
            ax.hist(diff_flat, bins=50, alpha=0.6, label='Difference', color='green', density=True)
            
            ax.set_title(f'{comp_type}: {weight_name.split(".")[-2:]}'.replace("['", "").replace("']", ""))
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'L2 diff: {weight_data["l2_norm_diff"]:.4f}\n'
            stats_text += f'Rel diff: {weight_data["relative_diff"]:.4f}\n'
            stats_text += f'Cos sim: {weight_data["cosine_sim"]:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(layer_weights), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_num} - Weight Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "weight_distributions" / f'layer_{layer_num:02d}_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer {layer_num} distribution plot")

    def create_neuron_analysis_plots(self, weight_diffs: dict, layer_data: dict):
        """Create neuron-by-neuron analysis plots"""
        print("🧠 Creating neuron-level analysis plots...")
        
        for layer_num in sorted(layer_data.keys()):
            if layer_num < 0 or layer_num >= 900:  # Skip special layers
                continue
                
            layer_info = layer_data[layer_num]
            self._create_neuron_comparison_plot(layer_num, layer_info, weight_diffs)
    
    def _create_neuron_comparison_plot(self, layer_num: int, layer_info: dict, weight_diffs: dict):
        """Create neuron-by-neuron comparison plots for a layer"""
        
        # Focus on major weight matrices (not biases)
        major_weights = []
        for comp_type, comp_weights in layer_info.items():
            for weight_info in comp_weights:
                weight_name = weight_info['name']
                if (weight_name in weight_diffs and 
                    weight_name.endswith('.weight') and 
                    len(weight_diffs[weight_name]['shape']) == 2):
                    major_weights.append((weight_name, weight_diffs[weight_name], comp_type))
        
        if not major_weights:
            return
        
        # Create figure
        n_weights = len(major_weights)
        fig = plt.figure(figsize=(20, 5 * n_weights))
        gs = GridSpec(n_weights, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        for i, (weight_name, weight_data, comp_type) in enumerate(major_weights):
            w1, w2, diff = weight_data['weights1'], weight_data['weights2'], weight_data['diff']
            
            # 1. Output neuron differences heatmap
            ax1 = fig.add_subplot(gs[i, 0])
            if w1.shape[0] > 1:
                out_neuron_diffs = torch.norm(diff, dim=1).float().numpy()
                im1 = ax1.imshow(out_neuron_diffs.reshape(-1, 1), aspect='auto', cmap='viridis')
                ax1.set_title(f'{comp_type}\nOutput Neuron Diffs')
                ax1.set_ylabel('Neuron Index')
                plt.colorbar(im1, ax=ax1)
            
            # 2. Input neuron differences heatmap  
            ax2 = fig.add_subplot(gs[i, 1])
            if w1.shape[1] > 1:
                in_neuron_diffs = torch.norm(diff, dim=0).float().numpy()
                im2 = ax2.imshow(in_neuron_diffs.reshape(1, -1), aspect='auto', cmap='viridis')
                ax2.set_title(f'{comp_type}\nInput Neuron Diffs')
                ax2.set_xlabel('Neuron Index')
                plt.colorbar(im2, ax=ax2)
            
            # 3. Weight matrix difference heatmap (sampled if too large)
            ax3 = fig.add_subplot(gs[i, 2])
            diff_matrix = diff.float().numpy()
            if diff_matrix.size > 10000:  # Sample large matrices
                step_r = max(1, diff_matrix.shape[0] // 100)
                step_c = max(1, diff_matrix.shape[1] // 100)
                diff_matrix = diff_matrix[::step_r, ::step_c]
            
            im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', 
                           vmin=-np.abs(diff_matrix).max(), vmax=np.abs(diff_matrix).max())
            ax3.set_title(f'{comp_type}\nWeight Differences')
            ax3.set_xlabel('Input Dimension')
            ax3.set_ylabel('Output Dimension')
            plt.colorbar(im3, ax=ax3)
            
            # 4. Distribution of neuron-level differences
            ax4 = fig.add_subplot(gs[i, 3])
            if w1.shape[0] > 1 and w1.shape[1] > 1:
                out_diffs = torch.norm(diff, dim=1).float().numpy()
                in_diffs = torch.norm(diff, dim=0).float().numpy()
                
                ax4.hist(out_diffs, bins=30, alpha=0.6, label='Output Neurons', color='blue')
                ax4.hist(in_diffs, bins=30, alpha=0.6, label='Input Neurons', color='red')
                ax4.set_title(f'{comp_type}\nNeuron Diff Distribution')
                ax4.set_xlabel('L2 Difference')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f'Out mean: {np.mean(out_diffs):.4f}\n'
                stats_text += f'In mean: {np.mean(in_diffs):.4f}\n'
                stats_text += f'Out std: {np.std(out_diffs):.4f}\n'
                stats_text += f'In std: {np.std(in_diffs):.4f}'
                ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        
        plt.suptitle(f'Layer {layer_num} - Neuron-Level Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = self.output_dir / "neuron_analysis" / f'layer_{layer_num:02d}_neurons.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🧠 Saved layer {layer_num} neuron analysis")

    def create_layer_by_layer_summary(self, layer_data: dict, distribution_analysis: dict):
        """Create individual summary plots for each layer"""
        print("📊 Creating layer-by-layer summary plots...")
        
        for layer_num in sorted(layer_data.keys()):
            if layer_num < 0 or layer_num >= 900:  # Skip special layers
                continue
                
            self._create_single_layer_summary(layer_num, layer_data[layer_num], distribution_analysis)
    
    def _create_single_layer_summary(self, layer_num: int, layer_info: dict, distribution_analysis: dict):
        """Create a comprehensive summary plot for a single layer"""
        
        # Collect layer statistics
        layer_stats = []
        for comp_type, comp_weights in layer_info.items():
            for weight_info in comp_weights:
                weight_name = weight_info['name']
                if weight_name in distribution_analysis:
                    stats = distribution_analysis[weight_name]
                    stats['component'] = comp_type
                    stats['param_type'] = weight_info['param_type']
                    layer_stats.append(stats)
        
        if not layer_stats:
            return
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Component comparison - L2 differences
        ax1 = fig.add_subplot(gs[0, 0])
        comp_diffs = defaultdict(list)
        for stat in layer_stats:
            if stat['param_type'] == 'weight':  # Focus on weights, not biases
                comp_diffs[stat['component']].append(stat['diff_std'])
        
        if comp_diffs:
            components = list(comp_diffs.keys())
            means = [np.mean(comp_diffs[comp]) for comp in components]
            stds = [np.std(comp_diffs[comp]) for comp in components]
            
            bars = ax1.bar(components, means, yerr=stds, capsize=5, alpha=0.7)
            ax1.set_title('Component Difference Std')
            ax1.set_ylabel('Std of Differences')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{mean:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Weight distribution comparison
        ax2 = fig.add_subplot(gs[0, 1])
        w1_means = [stat['w1_mean'] for stat in layer_stats if stat['param_type'] == 'weight']
        w2_means = [stat['w2_mean'] for stat in layer_stats if stat['param_type'] == 'weight']
        
        if w1_means and w2_means:
            ax2.scatter(w1_means, w2_means, alpha=0.7, s=50)
            ax2.plot([min(w1_means + w2_means), max(w1_means + w2_means)], 
                    [min(w1_means + w2_means), max(w1_means + w2_means)], 'r--', alpha=0.7)
            ax2.set_xlabel('Model 1 Weight Means')
            ax2.set_ylabel('Model 2 Weight Means')
            ax2.set_title('Weight Mean Comparison')
            ax2.grid(True, alpha=0.3)
        
        # 3. Statistical significance
        ax3 = fig.add_subplot(gs[0, 2])
        ks_stats = [stat['ks_statistic'] for stat in layer_stats if stat['param_type'] == 'weight']
        p_values = [stat['ks_pvalue'] for stat in layer_stats if stat['param_type'] == 'weight']
        
        if ks_stats and p_values:
            scatter = ax3.scatter(ks_stats, -np.log10(np.array(p_values) + 1e-10), 
                                alpha=0.7, s=50, c=range(len(ks_stats)), cmap='viridis')
            ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            ax3.set_xlabel('KS Statistic')
            ax3.set_ylabel('-log10(p-value)')
            ax3.set_title('Statistical Significance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Distribution shapes comparison
        ax4 = fig.add_subplot(gs[0, 3])
        skew_diff = [stat['w2_skewness'] - stat['w1_skewness'] for stat in layer_stats if stat['param_type'] == 'weight']
        kurt_diff = [stat['w2_kurtosis'] - stat['w1_kurtosis'] for stat in layer_stats if stat['param_type'] == 'weight']
        
        if skew_diff and kurt_diff:
            ax4.scatter(skew_diff, kurt_diff, alpha=0.7, s=50)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Skewness Difference (M2-M1)')
            ax4.set_ylabel('Kurtosis Difference (M2-M1)')
            ax4.set_title('Distribution Shape Changes')
            ax4.grid(True, alpha=0.3)
        
        # 5-8. Individual component histograms
        subplot_idx = 4
        for comp_type in ['attention', 'mlp', 'layernorm']:
            comp_stats = [stat for stat in layer_stats 
                         if stat['component'] == comp_type and stat['param_type'] == 'weight']
            
            if comp_stats and subplot_idx < 8:
                ax = fig.add_subplot(gs[1, subplot_idx-4])
                
                # Plot distribution of differences for this component
                diff_stds = [stat['diff_std'] for stat in comp_stats]
                if diff_stds:
                    ax.hist(diff_stds, bins=min(20, len(diff_stds)), alpha=0.7, 
                           color=plt.cm.tab10(subplot_idx-4))
                    ax.set_title(f'{comp_type.title()} Diff Std')
                    ax.set_xlabel('Std of Differences')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_std = np.mean(diff_stds)
                    ax.axvline(mean_std, color='red', linestyle='--', 
                              label=f'Mean: {mean_std:.4f}')
                    ax.legend()
                
                subplot_idx += 1
        
        # 9. Summary statistics table
        ax_table = fig.add_subplot(gs[2, :2])
        ax_table.axis('off')
        
        # Compute summary statistics
        weight_stats = [stat for stat in layer_stats if stat['param_type'] == 'weight']
        if weight_stats:
            summary_data = [
                ['Component Types', len(set(stat['component'] for stat in weight_stats))],
                ['Weight Matrices', len(weight_stats)],
                ['Avg Diff Std', f"{np.mean([stat['diff_std'] for stat in weight_stats]):.6f}"],
                ['Max Diff Std', f"{np.max([stat['diff_std'] for stat in weight_stats]):.6f}"],
                ['Avg KS Statistic', f"{np.mean([stat['ks_statistic'] for stat in weight_stats]):.6f}"],
                ['Significant Changes', f"{sum(1 for stat in weight_stats if stat['ks_pvalue'] < 0.05)}"],
            ]
            
            table = ax_table.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                                  cellLoc='left', loc='center', colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax_table.set_title(f'Layer {layer_num} Summary Statistics', 
                              fontsize=14, fontweight='bold', pad=20)
        
        # 10. Insights text
        ax_insights = fig.add_subplot(gs[2, 2:])
        ax_insights.axis('off')
        
        if weight_stats:
            # Generate insights
            max_diff_stat = max(weight_stats, key=lambda x: x['diff_std'])
            min_diff_stat = min(weight_stats, key=lambda x: x['diff_std'])
            
            insights_text = f"""
LAYER {layer_num} INSIGHTS:

🔍 Most Changed Component:
   {max_diff_stat['component']}: {max_diff_stat['weight_name'].split('.')[-2]}
   Diff Std: {max_diff_stat['diff_std']:.6f}

🔍 Least Changed Component:
   {min_diff_stat['component']}: {min_diff_stat['weight_name'].split('.')[-2]}
   Diff Std: {min_diff_stat['diff_std']:.6f}

📊 Distribution Changes:
   Avg Skewness Change: {np.mean([stat['w2_skewness'] - stat['w1_skewness'] for stat in weight_stats]):.4f}
   Avg Kurtosis Change: {np.mean([stat['w2_kurtosis'] - stat['w1_kurtosis'] for stat in weight_stats]):.4f}

🧪 Statistical Tests:
   Significant changes: {sum(1 for stat in weight_stats if stat['ks_pvalue'] < 0.05)}/{len(weight_stats)}
   Avg p-value: {np.mean([stat['ks_pvalue'] for stat in weight_stats]):.6f}
            """
            
            ax_insights.text(0.05, 0.95, insights_text, transform=ax_insights.transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.suptitle(f'Layer {layer_num} - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = self.output_dir / "layer_by_layer" / f'layer_{layer_num:02d}_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer {layer_num} comprehensive summary")

    def parse_layer_info(self, weight_name: str) -> Tuple[Optional[int], str, str]:
        """Parse layer number, component type, and parameter type from weight name"""
        
        # Handle embedding and LM head
        if 'embed_tokens' in weight_name or 'embed_positions' in weight_name:
            return -1, 'embedding', weight_name.split('.')[-1]
        elif 'lm_head' in weight_name or 'output' in weight_name:
            return 999, 'lm_head', weight_name.split('.')[-1]
        elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
            if 'model.norm' in weight_name:  # Final layer norm
                return 998, 'final_norm', weight_name.split('.')[-1]
        
        # Extract layer number
        layer_match = re.search(r'layers?\.(\d+)\.', weight_name)
        if not layer_match:
            return None, 'other', weight_name.split('.')[-1]
        
        layer_num = int(layer_match.group(1))
        
        # Determine component type
        if 'self_attn' in weight_name or 'attention' in weight_name:
            component = 'attention'
        elif 'mlp' in weight_name or 'feed_forward' in weight_name:
            component = 'mlp'
        elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
            component = 'layernorm'
        else:
            component = 'other'
        
        param_type = weight_name.split('.')[-1]  # weight, bias, etc.
        
        return layer_num, component, param_type

    def organize_by_layers(self, weight_diffs: dict) -> dict:
        """Organize weight differences by layer and component"""
        layer_data = defaultdict(lambda: defaultdict(list))
        
        for weight_name, stats in weight_diffs.items():
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            
            if layer_num is not None:
                layer_data[layer_num][component].append({
                    'name': weight_name,
                    'param_type': param_type,
                    **stats
                })
        
        return dict(layer_data)

    def create_layer_summary(self, layer_data: dict) -> pd.DataFrame:
        """Create layer-wise summary statistics"""
        summary_data = []
        
        for layer_num in sorted(layer_data.keys()):
            layer_info = layer_data[layer_num]
            
            # Get layer name
            if layer_num == -1:
                layer_name = 'Embedding'
            elif layer_num == 998:
                layer_name = 'Final Norm'
            elif layer_num == 999:
                layer_name = 'LM Head'
            else:
                layer_name = f'Layer {layer_num}'
            
            # Aggregate statistics across all components in this layer
            all_weights = []
            for component_weights in layer_info.values():
                all_weights.extend(component_weights)
            
            if all_weights:
                summary_data.append({
                    'layer_num': layer_num,
                    'layer_name': layer_name,
                    'total_params': sum(w['num_params'] for w in all_weights),
                    'mean_l2_diff': np.mean([w['l2_norm_diff'] for w in all_weights]),
                    'mean_rel_diff': np.mean([w['relative_diff'] for w in all_weights]),
                    'mean_cosine_sim': np.mean([w['cosine_sim'] for w in all_weights]),
                    'mean_frob_diff': np.mean([w['frobenius_norm_diff'] for w in all_weights]),
                    'num_components': len(layer_info),
                    'components': list(layer_info.keys())
                })
        
        return pd.DataFrame(summary_data)

    def create_comprehensive_visualization(self, layer_data: dict, layer_summary: pd.DataFrame):
        """Create comprehensive visualizations"""
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Filter to regular layers only for most plots
        regular_layers = layer_summary[
            (layer_summary['layer_num'] >= 0) & 
            (layer_summary['layer_num'] < 900)
        ].sort_values('layer_num')
        
        # 1. Layer-wise L2 differences line plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(regular_layers['layer_num'], regular_layers['mean_l2_diff'], 
                'o-', linewidth=3, markersize=8, color='#2E86AB')
        ax1.fill_between(regular_layers['layer_num'], regular_layers['mean_l2_diff'], 
                        alpha=0.3, color='#2E86AB')
        ax1.set_title('L2 Differences Across Layers', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('Mean L2 Difference')
        ax1.grid(True, alpha=0.3)
        
        # 2. Relative differences
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(regular_layers['layer_num'], regular_layers['mean_rel_diff'], 
                'o-', linewidth=3, markersize=8, color='#A23B72')
        ax2.fill_between(regular_layers['layer_num'], regular_layers['mean_rel_diff'], 
                        alpha=0.3, color='#A23B72')
        ax2.set_title('Relative Differences Across Layers', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Mean Relative Difference')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cosine similarities
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(regular_layers['layer_num'], regular_layers['mean_cosine_sim'], 
                'o-', linewidth=3, markersize=8, color='#F18F01')
        ax3.fill_between(regular_layers['layer_num'], regular_layers['mean_cosine_sim'], 
                        alpha=0.3, color='#F18F01')
        ax3.set_title('Cosine Similarities Across Layers', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Mean Cosine Similarity')
        ax3.grid(True, alpha=0.3)
        
        # 4. Parameter count by layer
        ax4 = fig.add_subplot(gs[0, 3])
        bars = ax4.bar(regular_layers['layer_num'], regular_layers['total_params'], 
                      alpha=0.7, color='#C73E1D')
        ax4.set_title('Parameters per Layer', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Layer Number')
        ax4.set_ylabel('Parameter Count')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Component-wise heatmap across layers
        ax5 = fig.add_subplot(gs[1, :2])
        
        # Create component matrix
        component_types = ['attention', 'mlp', 'layernorm']
        layer_comp_matrix = []
        layer_labels = []
        
        for _, row in regular_layers.iterrows():
            layer_num = row['layer_num']
            layer_info = layer_data.get(layer_num, {})
            layer_labels.append(f"L{layer_num}")
            
            comp_values = []
            for comp_type in component_types:
                if comp_type in layer_info:
                    weights = layer_info[comp_type]
                    avg_diff = np.mean([w['l2_norm_diff'] for w in weights])
                    comp_values.append(avg_diff)
                else:
                    comp_values.append(0)
            layer_comp_matrix.append(comp_values)
        
        if layer_comp_matrix:
            im = ax5.imshow(np.array(layer_comp_matrix).T, aspect='auto', cmap='viridis')
            ax5.set_yticks(range(len(component_types)))
            ax5.set_yticklabels(component_types)
            ax5.set_xticks(range(0, len(layer_labels), max(1, len(layer_labels)//10)))
            ax5.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), max(1, len(layer_labels)//10))])
            ax5.set_title('Component Differences Heatmap', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax5, label='Mean L2 Difference')
        
        # 6. Distribution of differences
        ax6 = fig.add_subplot(gs[1, 2])
        all_l2_diffs = []
        for layer_info in layer_data.values():
            for comp_weights in layer_info.values():
                all_l2_diffs.extend([w['l2_norm_diff'] for w in comp_weights])
        
        ax6.hist(all_l2_diffs, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax6.axvline(np.mean(all_l2_diffs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(all_l2_diffs):.4f}')
        ax6.set_title('Distribution of L2 Differences', fontsize=14, fontweight='bold')
        ax6.set_xlabel('L2 Difference')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Component comparison
        ax7 = fig.add_subplot(gs[1, 3])
        comp_stats = defaultdict(list)
        
        for layer_info in layer_data.values():
            for comp_type, comp_weights in layer_info.items():
                if comp_type in component_types:
                    comp_stats[comp_type].extend([w['l2_norm_diff'] for w in comp_weights])
        
        comp_means = [np.mean(comp_stats[comp]) for comp in component_types if comp in comp_stats]
        comp_stds = [np.std(comp_stats[comp]) for comp in component_types if comp in comp_stats]
        valid_comps = [comp for comp in component_types if comp in comp_stats]
        
        bars = ax7.bar(valid_comps, comp_means, yerr=comp_stds, capsize=5, 
                      alpha=0.7, color=['#A23B72', '#F18F01', '#C73E1D'])
        ax7.set_title('Component Type Comparison', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Mean L2 Difference')
        ax7.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean in zip(bars, comp_means):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 8. Layer depth analysis (smoothed trend)
        ax8 = fig.add_subplot(gs[2, 0])
        from scipy.ndimage import gaussian_filter1d
        smoothed_l2 = gaussian_filter1d(regular_layers['mean_l2_diff'].values, sigma=1)
        
        ax8.plot(regular_layers['layer_num'], regular_layers['mean_l2_diff'], 
                'o', alpha=0.6, color='#2E86AB', label='Actual')
        ax8.plot(regular_layers['layer_num'], smoothed_l2, 
                '-', linewidth=3, color='#A23B72', label='Smoothed Trend')
        ax8.set_title('Layer Depth Trend Analysis', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Layer Number')
        ax8.set_ylabel('Mean L2 Difference')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Special layers comparison
        ax9 = fig.add_subplot(gs[2, 1])
        special_layers = layer_summary[
            (layer_summary['layer_num'] < 0) | (layer_summary['layer_num'] >= 900)
        ]
        
        if not special_layers.empty:
            bars = ax9.bar(special_layers['layer_name'], special_layers['mean_l2_diff'],
                          alpha=0.7, color=['#F18F01', '#C73E1D', '#2E86AB'])
            ax9.set_title('Special Layers Comparison', fontsize=14, fontweight='bold')
            ax9.set_ylabel('Mean L2 Difference')
            ax9.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, special_layers['mean_l2_diff']):
                ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
        
        # 10. Correlation analysis
        ax10 = fig.add_subplot(gs[2, 2])
        all_rel_diffs = []
        all_cos_sims = []
        
        for layer_info in layer_data.values():
            for comp_weights in layer_info.values():
                all_rel_diffs.extend([w['relative_diff'] for w in comp_weights])
                all_cos_sims.extend([w['cosine_sim'] for w in comp_weights])
        
        ax10.scatter(all_rel_diffs, all_cos_sims, alpha=0.6, s=30, color='#A23B72')
        ax10.set_xlabel('Relative Difference')
        ax10.set_ylabel('Cosine Similarity')
        ax10.set_title('Relative Diff vs Cosine Similarity', fontsize=14, fontweight='bold')
        ax10.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(all_rel_diffs) > 1:
            corr = np.corrcoef(all_rel_diffs, all_cos_sims)[0,1]
            ax10.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax10.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
        
        # 11. Summary statistics table
        ax11 = fig.add_subplot(gs[2, 3])
        ax11.axis('off')
        
        summary_stats = [
            ['Total Layers', len(regular_layers)],
            ['Avg L2 Diff', f'{regular_layers["mean_l2_diff"].mean():.4f}'],
            ['Max L2 Diff', f'{regular_layers["mean_l2_diff"].max():.4f}'],
            ['Min L2 Diff', f'{regular_layers["mean_l2_diff"].min():.4f}'],
            ['Avg Cosine Sim', f'{regular_layers["mean_cosine_sim"].mean():.4f}'],
            ['Total Parameters', f'{regular_layers["total_params"].sum():,}']
        ]
        
        table = ax11.table(cellText=summary_stats, colLabels=['Metric', 'Value'],
                          cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax11.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # 12. Layer progression with error bars
        ax12 = fig.add_subplot(gs[3, :2])
        
        # Calculate std for each layer
        layer_stds = []
        for _, row in regular_layers.iterrows():
            layer_num = row['layer_num']
            layer_info = layer_data.get(layer_num, {})
            all_diffs = []
            for comp_weights in layer_info.values():
                all_diffs.extend([w['l2_norm_diff'] for w in comp_weights])
            layer_stds.append(np.std(all_diffs) if all_diffs else 0)
        
        ax12.errorbar(regular_layers['layer_num'], regular_layers['mean_l2_diff'], 
                     yerr=layer_stds, fmt='o-', linewidth=2, markersize=6, 
                     capsize=3, capthick=1, color='#2E86AB')
        ax12.fill_between(regular_layers['layer_num'], 
                         regular_layers['mean_l2_diff'] - np.array(layer_stds),
                         regular_layers['mean_l2_diff'] + np.array(layer_stds),
                         alpha=0.2, color='#2E86AB')
        ax12.set_title('Layer-wise Differences with Variability', fontsize=14, fontweight='bold')
        ax12.set_xlabel('Layer Number')
        ax12.set_ylabel('L2 Difference (± Std)')
        ax12.grid(True, alpha=0.3)
        
        # 13. Component breakdown pie chart
        ax13 = fig.add_subplot(gs[3, 2])
        comp_totals = {}
        for comp_type in component_types:
            total = sum(sum(w['num_params'] for w in layer_info.get(comp_type, []))
                       for layer_info in layer_data.values())
            if total > 0:
                comp_totals[comp_type] = total
        
        if comp_totals:
            ax13.pie(comp_totals.values(), labels=comp_totals.keys(), autopct='%1.1f%%',
                    colors=['#A23B72', '#F18F01', '#C73E1D'])
            ax13.set_title('Parameter Distribution by Component', fontsize=14, fontweight='bold')
        
        # 14. Final insights text
        ax14 = fig.add_subplot(gs[3, 3])
        ax14.axis('off')
        
        # Generate insights
        max_diff_layer = regular_layers.loc[regular_layers['mean_l2_diff'].idxmax()]
        min_diff_layer = regular_layers.loc[regular_layers['mean_l2_diff'].idxmin()]
        
        insights_text = f"""
KEY INSIGHTS:

📊 Most Changed Layer:
   {max_diff_layer['layer_name']} 
   (L2 diff: {max_diff_layer['mean_l2_diff']:.4f})

📊 Least Changed Layer:
   {min_diff_layer['layer_name']}
   (L2 diff: {min_diff_layer['mean_l2_diff']:.4f})

🧠 Component Impact:
   MLP: {comp_means[1]:.4f}
   Attention: {comp_means[0]:.4f}
   LayerNorm: {comp_means[2]:.4f}

📈 Overall Pattern:
   Mean similarity: {regular_layers['mean_cosine_sim'].mean():.4f}
   Total parameters: {regular_layers['total_params'].sum():,}
        """
        
        ax14.text(0.05, 0.95, insights_text, transform=ax14.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.suptitle(f'Weight Difference Analysis: {self.model1_path.name} vs {self.model2_path.name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save the plot
        output_path = self.output_dir / 'comprehensive_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comprehensive visualization saved to: {output_path}")

    def save_results(self, layer_summary: pd.DataFrame, layer_data: dict, distribution_analysis: dict):
        """Save detailed results to files"""
        
        # Save layer summary
        layer_summary.to_csv(self.output_dir / 'layer_summary.csv', index=False)
        
        # Save detailed layer data
        detailed_data = []
        for layer_num, layer_info in layer_data.items():
            for comp_type, comp_weights in layer_info.items():
                for weight_info in comp_weights:
                    # Remove tensor data for CSV export
                    export_info = {k: v for k, v in weight_info.items() 
                                 if k not in ['weights1', 'weights2', 'diff']}
                    detailed_data.append({
                        'layer_num': layer_num,
                        'component': comp_type,
                        'weight_name': weight_info['name'],
                        'param_type': weight_info['param_type'],
                        **export_info
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(self.output_dir / 'detailed_analysis.csv', index=False)
        
        # Save distribution analysis
        dist_df = pd.DataFrame.from_dict(distribution_analysis, orient='index')
        dist_df.to_csv(self.output_dir / 'weight_distribution_analysis.csv')
        
        # Save configuration comparison
        config_comparison = {
            'model1_path': str(self.model1_path),
            'model2_path': str(self.model2_path),
            'architecture': self.architecture,
            'config_diff': {
                key: {
                    'model1': self.config1.get(key),
                    'model2': self.config2.get(key),
                    'same': self.config1.get(key) == self.config2.get(key)
                }
                for key in set(self.config1.keys()) | set(self.config2.keys())
            }
        }
        
        with open(self.output_dir / 'config_comparison.json', 'w') as f:
            json.dump(config_comparison, f, indent=2, default=str)
        
        print(f"📁 Results saved to {self.output_dir}/:")
        print("   - layer_summary.csv")
        print("   - detailed_analysis.csv")
        print("   - weight_distribution_analysis.csv")
        print("   - config_comparison.json")
        print("   - comprehensive_analysis.png")
        print("   - layer_by_layer/*.png")
        print("   - weight_distributions/*.png")  
        print("   - neuron_analysis/*.png")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n🚀 Starting comprehensive analysis...")
        
        # Load weights
        print("📥 Loading model weights...")
        weights1 = self.load_weights(self.model1_path, self.weight_map1)
        weights2 = self.load_weights(self.model2_path, self.weight_map2)
        
        # Compute differences
        print("🧮 Computing weight differences...")
        weight_diffs = self.compute_weight_differences(weights1, weights2)
        
        # Analyze weight distributions
        # distribution_analysis = self.analyze_weight_distributions(weight_diffs)
        distribution_analysis = {}
        
        # Organize by layers
        print(" Organizing by layers...")
        layer_data = self.organize_by_layers(weight_diffs)
        
        # Create summary
        print(" Creating layer summary...")
        layer_summary = self.create_layer_summary(layer_data)
        
        # Create main comprehensive visualization
        print(" Creating main visualization...")
        self.create_comprehensive_visualization(layer_data, layer_summary)
        
        # Create detailed analyses
        # print(" Creating detailed weight distribution plots...")
        # self.create_weight_distribution_plots(weight_diffs, layer_data)
        
        # print(" Creating neuron-level analysis...")
        # self.create_neuron_analysis_plots(weight_diffs, layer_data)
        
        # print(" Creating layer-by-layer summaries...")
        # self.create_layer_by_layer_summary(layer_data, distribution_analysis)
        
        # Save results
        print(" Saving results...")
        self.save_results(layer_summary, layer_data, distribution_analysis)
        
        print("\n Comprehensive analysis complete!")
        
        # Clean up memory by removing tensor references
        for stats in weight_diffs.values():
            if 'weights1' in stats:
                del stats['weights1']
            if 'weights2' in stats:
                del stats['weights2'] 
            if 'diff' in stats:
                del stats['diff']
        
        return layer_data, layer_summary, distribution_analysis

def main():
    parser = argparse.ArgumentParser(description='Visualize weight differences between two SFT models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--output', type=str, default='model_diff_analysis', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ModelDiffVisualizer(args.model1, args.model2, args.output)
    
    # Run analysis
    layer_data, layer_summary, distribution_analysis = visualizer.run_analysis()
    
    print(f"\n🎯 Key Findings:")
    print(f"   - Analyzed {len(layer_summary)} layers")
    print(f"   - Max L2 difference: {layer_summary['mean_l2_diff'].max():.6f}")
    print(f"   - Min L2 difference: {layer_summary['mean_l2_diff'].min():.6f}")
    print(f"   - Average cosine similarity: {layer_summary['mean_cosine_sim'].mean():.6f}")
    output_path = Path(args.output)
    layer_plots = len(list((output_path / 'layer_by_layer').glob('*.png'))) if (output_path / 'layer_by_layer').exists() else 0
    dist_plots = len(list((output_path / 'weight_distributions').glob('*.png'))) if (output_path / 'weight_distributions').exists() else 0
    neuron_plots = len(list((output_path / 'neuron_analysis').glob('*.png'))) if (output_path / 'neuron_analysis').exists() else 0
    
    print(f"   - Generated {layer_plots} layer-by-layer plots")
    print(f"   - Generated {dist_plots} distribution plots")
    print(f"   - Generated {neuron_plots} neuron analysis plots")

if __name__ == "__main__":
    main() 