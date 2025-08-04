#!/usr/bin/env python3
"""
Weight Position Distribution Analyzer

Analyzes how weight values are distributed across their flattened positions,
comparing two SFT models to understand spatial patterns in weight changes.

Usage:
    python weight_position_analysis.py --model1 path/to/model1 --model2 path/to/model2
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
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class WeightPositionAnalyzer:
    def __init__(self, model1_path: str, model2_path: str, output_dir: str = "position_analysis"):
        self.model1_path = Path(model1_path)
        self.model2_path = Path(model2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "layer_positions").mkdir(exist_ok=True)
        (self.output_dir / "component_positions").mkdir(exist_ok=True)
        (self.output_dir / "difference_maps").mkdir(exist_ok=True)
        
        print(f"🔍 Analyzing weight positions:")
        print(f"   Model 1: {self.model1_path.name}")
        print(f"   Model 2: {self.model2_path.name}")
        print(f"   Output: {self.output_dir}")
        
        # Load configurations
        self.config1 = self._load_config(self.model1_path)
        self.config2 = self._load_config(self.model2_path)
        self.architecture = self._detect_architecture()
        
        # Load weight mappings
        self.weight_map1 = self._load_weight_map(self.model1_path)
        self.weight_map2 = self._load_weight_map(self.model2_path)
        
        print(f"📐 Architecture: {self.architecture}")
        print(f"🔢 Layers: {self.config1.get('num_hidden_layers', 'unknown')}")

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
        """Detect model architecture"""
        if "architectures" in self.config1:
            return self.config1["architectures"][0]
        elif "model_type" in self.config1:
            return self.config1["model_type"]
        else:
            return "unknown"

    def load_weights(self, model_path: Path, weight_map: dict) -> dict:
        """Load weights from safetensors files"""
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

    def parse_layer_info(self, weight_name: str) -> Tuple[Optional[int], str, str]:
        """Parse layer info from weight name"""
        
        # Handle special layers
        if 'embed_tokens' in weight_name or 'embed_positions' in weight_name:
            return -1, 'embedding', weight_name.split('.')[-1]
        elif 'lm_head' in weight_name or 'output' in weight_name:
            return 999, 'lm_head', weight_name.split('.')[-1]
        elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
            if 'model.norm' in weight_name:
                return 998, 'final_norm', weight_name.split('.')[-1]
        
        # Extract layer number
        layer_match = re.search(r'layers?\.(\d+)\.', weight_name)
        if not layer_match:
            return None, 'other', weight_name.split('.')[-1]
        
        layer_num = int(layer_match.group(1))
        
        # Component type
        if 'self_attn' in weight_name or 'attention' in weight_name:
            component = 'attention'
        elif 'mlp' in weight_name or 'feed_forward' in weight_name:
            component = 'mlp'
        elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
            component = 'layernorm'
        else:
            component = 'other'
        
        param_type = weight_name.split('.')[-1]
        return layer_num, component, param_type

    def analyze_position_distributions(self, weights1: dict, weights2: dict) -> dict:
        """Analyze weight values by their flattened positions"""
        print("🔬 Analyzing position distributions...")
        
        position_analysis = {}
        common_keys = set(weights1.keys()) & set(weights2.keys())
        
        for weight_name in common_keys:
            w1, w2 = weights1[weight_name], weights2[weight_name]
            
            if w1.shape != w2.shape:
                continue
                
            # Flatten weights
            w1_flat = w1.flatten().float()
            w2_flat = w2.flatten().float()
            diff_flat = (w1_flat - w2_flat)
            
            # Create position indices
            positions = torch.arange(len(w1_flat))
            
            position_analysis[weight_name] = {
                'positions': positions.numpy(),
                'w1_values': w1_flat.numpy(),
                'w2_values': w2_flat.numpy(),
                'differences': diff_flat.numpy(),
                'shape': list(w1.shape),
                'total_params': len(w1_flat),
                # Position-wise statistics
                'w1_position_mean': torch.mean(w1_flat).item(),
                'w2_position_mean': torch.mean(w2_flat).item(),
                'diff_position_mean': torch.mean(diff_flat).item(),
                'position_correlation': pearsonr(w1_flat.numpy(), w2_flat.numpy())[0],
                'max_diff_position': torch.argmax(torch.abs(diff_flat)).item(),
                'max_diff_value': torch.max(torch.abs(diff_flat)).item(),
            }
        
        return position_analysis

    def create_layer_position_plots(self, position_analysis: dict):
        """Create position distribution plots for each layer"""
        print("🎨 Creating layer position plots...")
        
        # Organize by layers
        layer_weights = defaultdict(list)
        for weight_name, data in position_analysis.items():
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if layer_num is not None and 0 <= layer_num <= 50:  # Regular layers only
                layer_weights[layer_num].append((weight_name, data, component, param_type))
        
        # Create plots for each layer
        for layer_num in sorted(layer_weights.keys()):
            self._create_single_layer_value_plot(layer_num, layer_weights[layer_num])
            self._create_single_layer_diff_plot(layer_num, layer_weights[layer_num])

    def _create_single_layer_value_plot(self, layer_num: int, layer_data: List):
        """Create weight value plot for a single layer"""
        
        # Filter to weight matrices only (skip biases for clarity)
        weight_matrices = [(name, data, comp, ptype) for name, data, comp, ptype in layer_data 
                          if ptype == 'weight' and len(data['shape']) >= 2]
        
        if not weight_matrices:
            return
        
        n_weights = len(weight_matrices)
        n_cols = min(2, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 12, n_rows * 8))
        if n_weights == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_weights > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (weight_name, data, component, param_type) in enumerate(weight_matrices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            positions = data['positions']
            w1_values = data['w1_values']
            w2_values = data['w2_values']
            
            # Subsample for large tensors to avoid overcrowding
            if len(positions) > 10000:
                step = len(positions) // 10000
                positions = positions[::step]
                w1_values = w1_values[::step]
                w2_values = w2_values[::step]
            
            # Plot weight values by position
            ax.scatter(positions, w1_values, alpha=0.6, s=1, color='blue', label='Model 1')
            ax.scatter(positions, w2_values, alpha=0.6, s=1, color='red', label='Model 2')
            
            ax.set_title(f'{component.upper()}: {weight_name.split(".")[-2]}')
            ax.set_xlabel('Position (Flattened Index)')
            ax.set_ylabel('Weight Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value statistics text
            stats_text = f'Shape: {data["shape"]}\n'
            stats_text += f'Total params: {data["total_params"]:,}\n'
            stats_text += f'Correlation: {data["position_correlation"]:.4f}\n'
            stats_text += f'Model 1 mean: {data["w1_position_mean"]:.6f}\n'
            stats_text += f'Model 2 mean: {data["w2_position_mean"]:.6f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(weight_matrices), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_num} - Weight Values by Position', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "layer_positions" / f'layer_{layer_num:02d}_values.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer {layer_num} values plot")

    def _create_single_layer_diff_plot(self, layer_num: int, layer_data: List):
        """Create weight difference plot for a single layer"""
        
        # Filter to weight matrices only (skip biases for clarity)
        weight_matrices = [(name, data, comp, ptype) for name, data, comp, ptype in layer_data 
                          if ptype == 'weight' and len(data['shape']) >= 2]
        
        if not weight_matrices:
            return
        
        n_weights = len(weight_matrices)
        n_cols = min(2, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 12, n_rows * 8))
        if n_weights == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_weights > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (weight_name, data, component, param_type) in enumerate(weight_matrices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            positions = data['positions']
            differences = data['differences']
            
            # Subsample for large tensors to avoid overcrowding
            if len(positions) > 10000:
                step = len(positions) // 10000
                positions = positions[::step]
                differences = differences[::step]
            
            # Plot differences by position
            ax.scatter(positions, differences, alpha=0.6, s=1, color='green', label='Differences')
            
            # Highlight positions with large differences
            large_diff_mask = np.abs(differences) > np.percentile(np.abs(differences), 95)
            if np.any(large_diff_mask):
                ax.scatter(positions[large_diff_mask], differences[large_diff_mask], 
                          s=3, color='red', alpha=0.8, label='Large Diff (top 5%)')
            
            # Add zero line for reference
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_title(f'{component.upper()}: {weight_name.split(".")[-2]}')
            ax.set_xlabel('Position (Flattened Index)')
            ax.set_ylabel('Weight Difference (Model2 - Model1)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add difference statistics text
            diff_mean = np.mean(differences)
            diff_std = np.std(differences)
            diff_max_abs = np.max(np.abs(differences))
            
            stats_text = f'Shape: {data["shape"]}\n'
            stats_text += f'Total params: {data["total_params"]:,}\n'
            stats_text += f'Diff mean: {diff_mean:.6f}\n'
            stats_text += f'Diff std: {diff_std:.6f}\n'
            stats_text += f'Max |diff|: {diff_max_abs:.6f}\n'
            stats_text += f'Max diff pos: {data["max_diff_position"]}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(weight_matrices), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_num} - Weight Differences by Position', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "layer_positions" / f'layer_{layer_num:02d}_differences.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer {layer_num} differences plot")

    def create_difference_maps(self, position_analysis: dict):
        """Create difference heatmaps for weight matrices"""
        print("🗺️ Creating difference maps...")
        
        # Organize by layers
        layer_weights = defaultdict(list)
        for weight_name, data in position_analysis.items():
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 50 and 
                param_type == 'weight' and len(data['shape']) == 2):
                layer_weights[layer_num].append((weight_name, data, component))
        
        # Create difference maps for selected layers
        sample_layers = sorted(layer_weights.keys())[::4]  # Every 4th layer
        
        for layer_num in sample_layers:
            self._create_layer_difference_map(layer_num, layer_weights[layer_num])

    def _create_layer_difference_map(self, layer_num: int, layer_data: List):
        """Create difference heatmap for a layer"""
        
        if not layer_data:
            return
        
        n_weights = len(layer_data)
        n_cols = min(3, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        if n_weights == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_weights > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (weight_name, data, component) in enumerate(layer_data):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Reshape differences back to original shape
            shape = data['shape']
            diff_matrix = data['differences'].reshape(shape)
            
            # Sample large matrices
            if diff_matrix.size > 10000:
                step_r = max(1, shape[0] // 100)
                step_c = max(1, shape[1] // 100)
                diff_matrix = diff_matrix[::step_r, ::step_c]
                title_suffix = " (sampled)"
            else:
                title_suffix = ""
            
            # Create heatmap
            im = ax.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', 
                          vmin=-np.abs(diff_matrix).max(), vmax=np.abs(diff_matrix).max())
            
            ax.set_title(f'{component.upper()}: {weight_name.split(".")[-2]}{title_suffix}')
            ax.set_xlabel('Input Dimension')
            ax.set_ylabel('Output Dimension')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(len(layer_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_num} - Weight Difference Heatmaps', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "difference_maps" / f'layer_{layer_num:02d}_diffmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🗺️ Saved layer {layer_num} difference map")

    def create_component_analysis(self, position_analysis: dict):
        """Analyze position patterns by component type"""
        print("🧠 Creating component position analysis...")
        
        # Group by component type
        component_data = defaultdict(list)
        
        for weight_name, data in position_analysis.items():
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 50 and 
                param_type == 'weight' and component in ['attention', 'mlp', 'layernorm']):
                component_data[component].append(data)
        
        # Create component comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        components = ['attention', 'mlp', 'layernorm']
        colors = ['blue', 'red', 'green']
        
        for i, component in enumerate(components):
            if component not in component_data:
                continue
                
            comp_data = component_data[component]
            
            # Plot 1: Average position-wise correlation
            ax1 = axes[0, i]
            correlations = [d['position_correlation'] for d in comp_data]
            ax1.hist(correlations, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
            ax1.set_title(f'{component.upper()}\nPosition Correlations')
            ax1.set_xlabel('Correlation (Model1 vs Model2)')
            ax1.set_ylabel('Frequency')
            ax1.axvline(np.mean(correlations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(correlations):.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Max difference positions
            ax2 = axes[1, i]
            max_diff_positions = [d['max_diff_position'] / d['total_params'] for d in comp_data]
            ax2.hist(max_diff_positions, bins=20, alpha=0.7, color=colors[i], edgecolor='black')
            ax2.set_title(f'{component.upper()}\nMax Diff Position (Normalized)')
            ax2.set_xlabel('Relative Position of Max Difference')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(max_diff_positions), color='red', linestyle='--',
                       label=f'Mean: {np.mean(max_diff_positions):.4f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Component-wise Position Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "component_positions" / 'component_position_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🧠 Saved component analysis")

    def create_overall_position_summary(self, position_analysis: dict):
        """Create overall summary of position patterns"""
        print("📊 Creating overall position summary...")
        
        # Collect statistics
        all_correlations = []
        all_max_diff_positions = []
        component_stats = defaultdict(list)
        layer_stats = defaultdict(list)
        
        for weight_name, data in position_analysis.items():
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            
            if layer_num is not None and 0 <= layer_num <= 50 and param_type == 'weight':
                all_correlations.append(data['position_correlation'])
                all_max_diff_positions.append(data['max_diff_position'] / data['total_params'])
                
                component_stats[component].append(data['position_correlation'])
                layer_stats[layer_num].append(data['position_correlation'])
        
        # Create summary plot
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Overall correlation distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(all_correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Overall Position Correlations')
        ax1.set_xlabel('Correlation')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(all_correlations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_correlations):.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Max difference position distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(all_max_diff_positions, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Max Difference Positions')
        ax2.set_xlabel('Relative Position')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(all_max_diff_positions), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_max_diff_positions):.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Component comparison
        ax3 = fig.add_subplot(gs[0, 2])
        comp_names = list(component_stats.keys())
        comp_means = [np.mean(component_stats[comp]) for comp in comp_names]
        comp_stds = [np.std(component_stats[comp]) for comp in comp_names]
        
        bars = ax3.bar(comp_names, comp_means, yerr=comp_stds, capsize=5, alpha=0.7)
        ax3.set_title('Component Correlation Comparison')
        ax3.set_ylabel('Mean Position Correlation')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, mean in zip(bars, comp_means):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 4. Layer-wise correlation trend
        ax4 = fig.add_subplot(gs[0, 3])
        layer_nums = sorted(layer_stats.keys())
        layer_means = [np.mean(layer_stats[layer]) for layer in layer_nums]
        
        ax4.plot(layer_nums, layer_means, 'o-', linewidth=2, markersize=6)
        ax4.set_title('Layer-wise Position Correlations')
        ax4.set_xlabel('Layer Number')
        ax4.set_ylabel('Mean Position Correlation')
        ax4.grid(True, alpha=0.3)
        
        # 5-8. Position pattern examples (sample 4 weights)
        sample_weights = list(position_analysis.items())[:4]
        
        for i, (weight_name, data) in enumerate(sample_weights):
            if i >= 4:
                break
                
            ax = fig.add_subplot(gs[1 + i//2, i%2])
            
            positions = data['positions']
            differences = data['differences']
            
            # Subsample for plotting
            if len(positions) > 1000:
                step = len(positions) // 1000
                positions = positions[::step]
                differences = differences[::step]
            
            ax.scatter(positions, differences, alpha=0.6, s=1)
            ax.set_title(f'Difference Pattern\n{weight_name.split(".")[-2]}')
            ax.set_xlabel('Position')
            ax.set_ylabel('Weight Difference')
            ax.grid(True, alpha=0.3)
        
        # Summary statistics table
        ax_table = fig.add_subplot(gs[2, 2:])
        ax_table.axis('off')
        
        summary_data = [
            ['Total Weights Analyzed', len(position_analysis)],
            ['Mean Position Correlation', f'{np.mean(all_correlations):.4f}'],
            ['Std Position Correlation', f'{np.std(all_correlations):.4f}'],
            ['Mean Max Diff Position', f'{np.mean(all_max_diff_positions):.4f}'],
            ['Components Analyzed', len(component_stats)],
            ['Layers Analyzed', len(layer_stats)]
        ]
        
        table = ax_table.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                              cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax_table.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Weight Position Analysis Summary', fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = self.output_dir / 'position_analysis_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved overall summary")

    def save_position_data(self, position_analysis: dict):
        """Save position analysis data to CSV"""
        print("💾 Saving position data...")
        
        # Create summary DataFrame
        summary_data = []
        for weight_name, data in position_analysis.items():
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            
            summary_data.append({
                'weight_name': weight_name,
                'layer_num': layer_num,
                'component': component,
                'param_type': param_type,
                'shape': str(data['shape']),
                'total_params': data['total_params'],
                'position_correlation': data['position_correlation'],
                'max_diff_position': data['max_diff_position'],
                'max_diff_value': data['max_diff_value'],
                'w1_position_mean': data['w1_position_mean'],
                'w2_position_mean': data['w2_position_mean'],
                'diff_position_mean': data['diff_position_mean']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'position_analysis_summary.csv', index=False)
        
        print(f"   📄 Saved position analysis CSV")

    def run_analysis(self):
        """Run the complete position analysis"""
        print("\n🚀 Starting position analysis...")
        
        # Load weights
        print("📥 Loading model weights...")
        weights1 = self.load_weights(self.model1_path, self.weight_map1)
        weights2 = self.load_weights(self.model2_path, self.weight_map2)
        
        # Analyze positions
        print("🔬 Analyzing position distributions...")
        position_analysis = self.analyze_position_distributions(weights1, weights2)
        
        # Create visualizations
        print("🎨 Creating visualizations...")
        self.create_layer_position_plots(position_analysis)
        self.create_difference_maps(position_analysis)
        self.create_component_analysis(position_analysis)
        self.create_overall_position_summary(position_analysis)
        
        # Save data
        self.save_position_data(position_analysis)
        
        print("\n✅ Position analysis complete!")
        
        # Count generated files
        layer_plots = len(list((self.output_dir / 'layer_positions').glob('*.png')))
        diff_maps = len(list((self.output_dir / 'difference_maps').glob('*.png')))
        comp_plots = len(list((self.output_dir / 'component_positions').glob('*.png')))
        
        print(f"📊 Generated:")
        print(f"   - {layer_plots//2} layer value plots")
        print(f"   - {layer_plots//2} layer difference plots") 
        print(f"   - {diff_maps} difference heatmaps")
        print(f"   - {comp_plots} component analysis plots")
        print(f"   - 1 overall summary plot")
        print(f"   - 1 CSV data file")
        
        return position_analysis

def main():
    parser = argparse.ArgumentParser(description='Analyze weight position distributions')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--output', type=str, default='position_analysis', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = WeightPositionAnalyzer(args.model1, args.model2, args.output)
    
    # Run analysis
    position_analysis = analyzer.run_analysis()
    
    print(f"\n🎯 Key Findings:")
    print(f"   - Analyzed {len(position_analysis)} weight tensors")
    
    # Calculate some summary statistics
    correlations = [data['position_correlation'] for data in position_analysis.values()]
    max_diffs = [data['max_diff_value'] for data in position_analysis.values()]
    
    print(f"   - Mean position correlation: {np.mean(correlations):.4f}")
    print(f"   - Mean max difference: {np.mean(max_diffs):.6f}")
    print(f"   - Results saved to: {args.output}/")

if __name__ == "__main__":
    main() 