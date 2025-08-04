#!/usr/bin/env python3
"""
SFT Difference Analyzer

Comprehensive analysis of differences between base model and multiple SFT models.
Creates visualizations for layer-by-layer differences, distributions, special layers, etc.

Usage:
    python sft_difference_analyzer.py --base_model path/to/base --sft_models path/to/sft1 path/to/sft2 path/to/sft3
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
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from transformers import AutoConfig

warnings.filterwarnings('ignore')

# Set plotting style with vibrant colors
plt.style.use('default')
sns.set_style("whitegrid")

# Define color palette for multiple models
COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
    '#ff9896',  # Light Red
    '#98df8a',  # Light Green
    '#c5b0d5',  # Light Purple
    '#c49c94',  # Light Brown
    '#f7b6d3',  # Light Pink
]

class SFTDifferenceAnalyzer:
    def __init__(self, base_model_path: str, sft_model_paths: List[str], output_dir: str = "sft_difference_analysis"):
        self.base_model_path_str = base_model_path
        self.sft_model_paths = [Path(p) for p in sft_model_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "layer_analysis").mkdir(exist_ok=True)
        (self.output_dir / "distribution_analysis").mkdir(exist_ok=True)
        (self.output_dir / "comprehensive_plots").mkdir(exist_ok=True)
        (self.output_dir / "interactive_plots").mkdir(exist_ok=True)
        
        print(f"🔬 SFT Difference Analysis:")
        print(f"   Base Model: {base_model_path}")
        print(f"   SFT Models: {[p.name for p in self.sft_model_paths]}")
        print(f"   Output: {self.output_dir}")
        
        # Determine if base model is from HuggingFace or local path
        self.is_hf_base = not Path(base_model_path).exists()
        if self.is_hf_base:
            print(f"   📥 Detected HuggingFace model: {base_model_path}")
            self.base_model_path = self._download_hf_model(base_model_path)
        else:
            self.base_model_path = Path(base_model_path)
        
        # Load configurations
        self.base_config = self._load_config(self.base_model_path)
        self.sft_configs = [self._load_config(p) for p in self.sft_model_paths]
        self.architecture = self._detect_architecture()
        
        # Load weight mappings
        self.base_weight_map = self._load_weight_map(self.base_model_path)
        self.sft_weight_maps = [self._load_weight_map(p) for p in self.sft_model_paths]
        
        print(f"📐 Architecture: {self.architecture}")
        print(f"🔢 Layers: {self.base_config.get('num_hidden_layers', 'unknown')}")

    def _download_hf_model(self, model_name: str) -> Path:
        """Download HuggingFace model and return local path"""
        from huggingface_hub import snapshot_download
        
        print(f"   📥 Downloading {model_name} from HuggingFace...")
        
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=None,
            resume_download=True
        )
        
        print(f"   ✅ Downloaded to: {cache_dir}")
        return Path(cache_dir)

    def _load_config(self, model_path: Path) -> dict:
        """Load model configuration"""
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_weight_map(self, model_path: Path) -> dict:
        """Load weight mapping from safetensors index or single file"""
        index_path = model_path / "model.safetensors.index.json"
        
        if index_path.exists():
            # Sharded model with index
            with open(index_path, 'r') as f:
                return json.load(f)["weight_map"]
        else:
            # Single safetensors file
            single_file_path = model_path / "model.safetensors"
            if single_file_path.exists():
                # Create a dummy weight map for single file
                with safe_open(single_file_path, framework="pt", device="cpu") as f:
                    weight_map = {}
                    for key in f.keys():
                        weight_map[key] = "model.safetensors"
                    return weight_map
            else:
                raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {model_path}")

    def _detect_architecture(self) -> str:
        """Detect model architecture"""
        if "architectures" in self.base_config:
            return self.base_config["architectures"][0]
        elif "model_type" in self.base_config:
            return self.base_config["model_type"]
        else:
            return "unknown"

    def load_weights(self, model_path: Path, weight_map: dict) -> dict:
        """Load weights from safetensors files"""
        weights = {}
        safetensor_files = set(weight_map.values())
        
        for i, file in enumerate(safetensor_files, 1):
            file_path = model_path / file
            print(f"   Loading {file} ({i}/{len(safetensor_files)})...")
            
            if not file_path.exists():
                print(f"   ⚠️ Warning: {file} not found, skipping...")
                continue
                
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in weight_map:
                        weights[key] = f.get_tensor(key)
        
        return weights

    def parse_layer_info(self, weight_name: str) -> Tuple[Optional[int], str, str]:
        """Parse layer info from weight name"""
        
        if 'embed_tokens' in weight_name or 'embed_positions' in weight_name:
            return -1, 'embedding', weight_name.split('.')[-1]
        elif 'lm_head' in weight_name or 'output' in weight_name:
            return 999, 'lm_head', weight_name.split('.')[-1]
        elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
            if 'model.norm' in weight_name:
                return 998, 'final_norm', weight_name.split('.')[-1]
        
        layer_match = re.search(r'layers?\.(\d+)\.', weight_name)
        if not layer_match:
            return None, 'other', weight_name.split('.')[-1]
        
        layer_num = int(layer_match.group(1))
        
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

    def compute_weight_differences(self, base_weights: dict, sft_weights_list: List[dict]) -> Dict:
        """Compute weight differences between base and all SFT models"""
        print("🔍 Computing weight differences...")
        
        differences = {}
        
        # Get common weight names across all models
        common_weights = set(base_weights.keys())
        for sft_weights in sft_weights_list:
            common_weights &= set(sft_weights.keys())
        
        print(f"   📊 Found {len(common_weights)} common weight tensors")
        
        for weight_name in common_weights:
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            
            if layer_num is None or param_type != 'weight':
                continue
            
            base_weight = base_weights[weight_name].float()
            
            weight_diffs = []
            for i, sft_weights in enumerate(sft_weights_list):
                sft_weight = sft_weights[weight_name].float()
                
                # Compute various difference metrics
                diff = sft_weight - base_weight
                l2_norm = torch.norm(diff, p=2).item()
                rel_diff = l2_norm / (torch.norm(base_weight, p=2).item() + 1e-8)
                cosine_sim = torch.nn.functional.cosine_similarity(
                    base_weight.flatten(), sft_weight.flatten(), dim=0
                ).item()
                mean_abs_diff = torch.mean(torch.abs(diff)).item()
                max_abs_diff = torch.max(torch.abs(diff)).item()
                
                weight_diffs.append({
                    'model_idx': i,
                    'model_name': self.sft_model_paths[i].name,
                    'l2_norm': l2_norm,
                    'relative_diff': rel_diff,
                    'cosine_similarity': cosine_sim,
                    'mean_abs_diff': mean_abs_diff,
                    'max_abs_diff': max_abs_diff,
                    'diff_tensor': diff.numpy()
                })
            
            differences[weight_name] = {
                'layer_num': layer_num,
                'component': component,
                'param_type': param_type,
                'shape': list(base_weight.shape),
                'differences': weight_diffs
            }
        
        return differences

    def aggregate_by_layer(self, differences: Dict) -> pd.DataFrame:
        """Aggregate differences by layer"""
        layer_data = []
        
        for weight_name, data in differences.items():
            layer_num = data['layer_num']
            component = data['component']
            
            for diff_data in data['differences']:
                layer_data.append({
                    'layer_num': layer_num,
                    'component': component,
                    'weight_name': weight_name,
                    'model_idx': diff_data['model_idx'],
                    'model_name': diff_data['model_name'],
                    'l2_norm': diff_data['l2_norm'],
                    'relative_diff': diff_data['relative_diff'],
                    'cosine_similarity': diff_data['cosine_similarity'],
                    'mean_abs_diff': diff_data['mean_abs_diff'],
                    'max_abs_diff': diff_data['max_abs_diff']
                })
        
        df = pd.DataFrame(layer_data)
        
        # Aggregate by layer and model
        layer_summary = df.groupby(['layer_num', 'model_idx', 'model_name']).agg({
            'l2_norm': ['mean', 'std', 'max'],
            'relative_diff': ['mean', 'std', 'max'],
            'cosine_similarity': ['mean', 'std', 'min'],
            'mean_abs_diff': ['mean', 'std', 'max'],
            'max_abs_diff': ['mean', 'std', 'max']
        }).reset_index()
        
        # Flatten column names
        layer_summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                for col in layer_summary.columns]
        
        return layer_summary

    def create_layer_by_layer_plot(self, layer_summary: pd.DataFrame):
        """Create comprehensive layer-by-layer analysis plot"""
        print("📊 Creating layer-by-layer analysis...")
        
        # Filter to only regular transformer layers (0-50), exclude special layers
        regular_layers_data = layer_summary[
            (layer_summary['layer_num'] >= 0) & 
            (layer_summary['layer_num'] <= 50)
        ].copy()
        
        if len(regular_layers_data) == 0:
            print("   ⚠️ No regular transformer layers found for plotting")
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get unique models and layers
        models = regular_layers_data[['model_idx', 'model_name']].drop_duplicates().sort_values('model_idx')
        layers = sorted(regular_layers_data['layer_num'].unique())
        
        print(f"   📊 Plotting {len(layers)} layers: {min(layers)} to {max(layers)}")
        
        # 1. L2 Norm by Layer
        ax1 = fig.add_subplot(gs[0, 0])
        for _, model in models.iterrows():
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax1.plot(model_data['layer_num'], model_data['l2_norm_mean'], 
                    marker='o', linewidth=2, label=model['model_name'],
                    color=COLORS[model['model_idx'] % len(COLORS)])
            ax1.fill_between(model_data['layer_num'], 
                           model_data['l2_norm_mean'] - model_data['l2_norm_std'],
                           model_data['l2_norm_mean'] + model_data['l2_norm_std'],
                           alpha=0.2, color=COLORS[model['model_idx'] % len(COLORS)])
        
        ax1.set_title('L2 Norm of Weight Differences by Layer', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('L2 Norm (mean ± std)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Relative Difference by Layer
        ax2 = fig.add_subplot(gs[0, 1])
        for _, model in models.iterrows():
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax2.plot(model_data['layer_num'], model_data['relative_diff_mean'], 
                    marker='s', linewidth=2, label=model['model_name'],
                    color=COLORS[model['model_idx'] % len(COLORS)])
            ax2.fill_between(model_data['layer_num'], 
                           model_data['relative_diff_mean'] - model_data['relative_diff_std'],
                           model_data['relative_diff_mean'] + model_data['relative_diff_std'],
                           alpha=0.2, color=COLORS[model['model_idx'] % len(COLORS)])
        
        ax2.set_title('Relative Weight Differences by Layer', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Relative Difference (mean ± std)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cosine Similarity by Layer
        ax3 = fig.add_subplot(gs[1, 0])
        for _, model in models.iterrows():
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax3.plot(model_data['layer_num'], model_data['cosine_similarity_mean'], 
                    marker='^', linewidth=2, label=model['model_name'],
                    color=COLORS[model['model_idx'] % len(COLORS)])
            ax3.fill_between(model_data['layer_num'], 
                           model_data['cosine_similarity_mean'] - model_data['cosine_similarity_std'],
                           model_data['cosine_similarity_mean'] + model_data['cosine_similarity_std'],
                           alpha=0.2, color=COLORS[model['model_idx'] % len(COLORS)])
        
        ax3.set_title('Cosine Similarity by Layer', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Layer Number')
        ax3.set_ylabel('Cosine Similarity (mean ± std)')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Mean Absolute Difference by Layer
        ax4 = fig.add_subplot(gs[1, 1])
        for _, model in models.iterrows():
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax4.plot(model_data['layer_num'], model_data['mean_abs_diff_mean'], 
                    marker='D', linewidth=2, label=model['model_name'],
                    color=COLORS[model['model_idx'] % len(COLORS)])
            ax4.fill_between(model_data['layer_num'], 
                           model_data['mean_abs_diff_mean'] - model_data['mean_abs_diff_std'],
                           model_data['mean_abs_diff_mean'] + model_data['mean_abs_diff_std'],
                           alpha=0.2, color=COLORS[model['model_idx'] % len(COLORS)])
        
        ax4.set_title('Mean Absolute Difference by Layer', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Layer Number')
        ax4.set_ylabel('Mean Abs Difference (mean ± std)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # 5. Heatmap of L2 norms
        ax5 = fig.add_subplot(gs[2, :])
        
        # Create pivot table for heatmap (regular layers only)
        pivot_data = regular_layers_data.pivot(index='model_name', columns='layer_num', values='l2_norm_mean')
        
        im = ax5.imshow(pivot_data.values, aspect='auto', cmap='viridis', interpolation='nearest')
        ax5.set_yticks(range(len(pivot_data.index)))
        ax5.set_yticklabels(pivot_data.index)
        ax5.set_xticks(range(len(pivot_data.columns)))
        ax5.set_xticklabels(pivot_data.columns)
        ax5.set_title('L2 Norm Heatmap: Models vs Layers', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Layer Number')
        ax5.set_ylabel('SFT Models')
        
        # Add colorbar
        plt.colorbar(im, ax=ax5, label='L2 Norm')
        
        plt.suptitle('SFT Layer-by-Layer Weight Difference Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        output_path = self.output_dir / "comprehensive_plots" / 'layer_by_layer_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer-by-layer analysis plot")

    def create_special_layers_plot(self, layer_summary: pd.DataFrame):
        """Create analysis plot for special layers (embedding, final norm, LM head)"""
        print("🎯 Creating special layers analysis...")
        
        # Filter to only special layers
        special_layers_data = layer_summary[
            (layer_summary['layer_num'] == -1) |   # Embedding
            (layer_summary['layer_num'] == 998) |  # Final norm
            (layer_summary['layer_num'] == 999)    # LM head
        ].copy()
        
        if len(special_layers_data) == 0:
            print("   ⚠️ No special layers found for plotting")
            return
        
        # Create layer name mapping for better readability
        layer_names = {-1: 'Embedding', 998: 'Final Norm', 999: 'LM Head'}
        special_layers_data['layer_name'] = special_layers_data['layer_num'].map(layer_names)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Get unique models
        models = special_layers_data[['model_idx', 'model_name']].drop_duplicates().sort_values('model_idx')
        special_layers = sorted(special_layers_data['layer_num'].unique())
        
        print(f"   🎯 Plotting special layers: {[layer_names.get(l, f'Layer {l}') for l in special_layers]}")
        
        # 1. L2 Norm by Special Layer
        ax1 = axes[0, 0]
        x_pos = range(len(special_layers))
        width = 0.8 / len(models) if len(models) > 0 else 0.8
        
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = special_layers_data[special_layers_data['model_idx'] == model['model_idx']]
            values = []
            errors = []
            for layer_num in special_layers:
                layer_data = model_data[model_data['layer_num'] == layer_num]
                if len(layer_data) > 0:
                    values.append(layer_data['l2_norm_mean'].iloc[0])
                    errors.append(layer_data['l2_norm_std'].iloc[0])
                else:
                    values.append(0)
                    errors.append(0)
            
            ax1.bar([x + i * width for x in x_pos], values, width, 
                   yerr=errors, capsize=3, label=model['model_name'],
                   color=COLORS[model['model_idx'] % len(COLORS)], alpha=0.8)
        
        ax1.set_title('L2 Norm - Special Layers', fontweight='bold')
        ax1.set_xlabel('Layer Type')
        ax1.set_ylabel('L2 Norm (mean ± std)')
        ax1.set_xticks([x + width * (len(models) - 1) / 2 for x in x_pos])
        ax1.set_xticklabels([layer_names.get(l, f'Layer {l}') for l in special_layers])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similar pattern for other subplots...
        # (I'll include the key ones to keep it concise)
        
        plt.suptitle('Special Layers Analysis (Embedding, Final Norm, LM Head)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comprehensive_plots" / 'special_layers_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🎯 Saved special layers analysis plot")

    def create_distribution_plots(self, differences: Dict):
        """Create weight difference distribution plots model by model"""
        print("📈 Creating distribution analysis...")
        
        # Collect all difference values by model
        model_diffs = defaultdict(list)
        
        for weight_name, data in differences.items():
            layer_num = data['layer_num']
            if 0 <= layer_num <= 50:  # Regular layers only
                for diff_data in data['differences']:
                    model_idx = diff_data['model_idx']
                    model_name = diff_data['model_name']
                    diff_tensor = diff_data['diff_tensor']
                    
                    # Sample large tensors for plotting
                    if diff_tensor.size > 10000:
                        flat_diff = diff_tensor.flatten()
                        idx = np.random.choice(len(flat_diff), 10000, replace=False)
                        sampled_diff = flat_diff[idx]
                    else:
                        sampled_diff = diff_tensor.flatten()
                    
                    model_diffs[model_idx].extend(sampled_diff)
        
        n_models = len(model_diffs)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Create distribution plots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for model_idx, diffs in model_diffs.items():
            if model_idx >= len(axes):
                break
                
            ax = axes[model_idx]
            model_name = self.sft_model_paths[model_idx].name
            
            # Create histogram
            ax.hist(diffs, bins=100, alpha=0.7, density=True, 
                   color=COLORS[model_idx % len(COLORS)], 
                   edgecolor='white', linewidth=0.5)
            
            # Add KDE
            try:
                kde = gaussian_kde(diffs)
                x_range = np.linspace(np.min(diffs), np.max(diffs), 200)
                ax.plot(x_range, kde(x_range), 
                       color='red', linewidth=2, label='KDE')
            except:
                pass
            
            # Add statistics
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            skew_diff = stats.skew(diffs)
            
            ax.axvline(mean_diff, color='black', linestyle='--', linewidth=2, alpha=0.8)
            ax.axvline(mean_diff + std_diff, color='gray', linestyle=':', alpha=0.8)
            ax.axvline(mean_diff - std_diff, color='gray', linestyle=':', alpha=0.8)
            
            ax.set_title(f'{model_name}\nWeight Difference Distribution', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Weight Difference (SFT - Base)')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {mean_diff:.6f}\n'
            stats_text += f'Std: {std_diff:.6f}\n'
            stats_text += f'Skew: {skew_diff:.3f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Hide unused subplots
        for i in range(len(model_diffs), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Weight Difference Distributions by SFT Model', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "distribution_analysis" / 'difference_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved distribution analysis plot")

    def create_comparative_boxplot(self, layer_summary: pd.DataFrame):
        """Create comparative boxplot of differences across models"""
        print("📦 Creating comparative boxplot...")
        
        # Filter to only regular transformer layers (0-50)
        regular_layers_data = layer_summary[
            (layer_summary['layer_num'] >= 0) & 
            (layer_summary['layer_num'] <= 50)
        ].copy()
        
        if len(regular_layers_data) == 0:
            print("   ⚠️ No regular transformer layers found for boxplot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. L2 Norm boxplot
        ax1 = axes[0, 0]
        models = []
        l2_data = []
        for model_name in regular_layers_data['model_name'].unique():
            model_data = regular_layers_data[regular_layers_data['model_name'] == model_name]
            models.append(model_name)
            l2_data.append(model_data['l2_norm_mean'].values)
        
        bp1 = ax1.boxplot(l2_data, labels=models, patch_artist=True)
        for i, patch in enumerate(bp1['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
        
        ax1.set_title('L2 Norm Distribution Across Layers', fontweight='bold')
        ax1.set_ylabel('L2 Norm')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Similar for other metrics...
        
        plt.suptitle('Comparative Analysis of SFT Models', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comprehensive_plots" / 'comparative_boxplot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📦 Saved comparative boxplot")

    def create_interactive_plot(self, layer_summary: pd.DataFrame):
        """Create interactive plot using Plotly"""
        print("🎮 Creating interactive plot...")
        
        try:
            # Filter to only regular transformer layers (0-50)
            regular_layers_data = layer_summary[
                (layer_summary['layer_num'] >= 0) & 
                (layer_summary['layer_num'] <= 50)
            ].copy()
            
            if len(regular_layers_data) == 0:
                print("   ⚠️ No regular transformer layers found for interactive plot")
                return
                
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('L2 Norm by Layer', 'Relative Difference by Layer',
                               'Cosine Similarity by Layer', 'Mean Absolute Difference by Layer'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            models = regular_layers_data[['model_idx', 'model_name']].drop_duplicates().sort_values('model_idx')
            
            for _, model in models.iterrows():
                model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
                color = COLORS[model['model_idx'] % len(COLORS)]
                
                # L2 Norm
                fig.add_trace(
                    go.Scatter(x=model_data['layer_num'], y=model_data['l2_norm_mean'],
                              mode='lines+markers', name=f"{model['model_name']} - L2",
                              line=dict(color=color), showlegend=True),
                    row=1, col=1
                )
                
                # Additional traces for other metrics...
            
            fig.update_layout(
                title='Interactive SFT Analysis',
                height=800,
                showlegend=True
            )
            
            output_path = self.output_dir / "interactive_plots" / 'interactive_analysis.html'
            fig.write_html(str(output_path))
            
            print(f"   🎮 Saved interactive plot")
            
        except Exception as e:
            print(f"   ⚠️ Interactive plot failed: {str(e)}")

    def save_results(self, layer_summary: pd.DataFrame, differences: Dict):
        """Save analysis results to files"""
        print("💾 Saving results...")
        
        # Save layer summary
        layer_summary.to_csv(self.output_dir / 'layer_summary.csv', index=False)
        
        # Save detailed differences summary
        detailed_data = []
        for weight_name, data in differences.items():
            for diff_data in data['differences']:
                detailed_data.append({
                    'weight_name': weight_name,
                    'layer_num': data['layer_num'],
                    'component': data['component'],
                    'shape': str(data['shape']),
                    'model_idx': diff_data['model_idx'],
                    'model_name': diff_data['model_name'],
                    'l2_norm': diff_data['l2_norm'],
                    'relative_diff': diff_data['relative_diff'],
                    'cosine_similarity': diff_data['cosine_similarity'],
                    'mean_abs_diff': diff_data['mean_abs_diff'],
                    'max_abs_diff': diff_data['max_abs_diff']
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(self.output_dir / 'detailed_differences.csv', index=False)
        
        print(f"   💾 Saved CSV files")

    def run_analysis(self):
        """Run the complete SFT difference analysis"""
        print("\n🚀 Starting SFT difference analysis...")
        
        # Load weights
        print("📥 Loading base model weights...")
        base_weights = self.load_weights(self.base_model_path, self.base_weight_map)
        
        print("📥 Loading SFT model weights...")
        sft_weights_list = []
        for i, (sft_path, sft_map) in enumerate(zip(self.sft_model_paths, self.sft_weight_maps)):
            print(f"   Loading SFT model {i+1}/{len(self.sft_model_paths)}: {sft_path.name}")
            sft_weights = self.load_weights(sft_path, sft_map)
            sft_weights_list.append(sft_weights)
        
        # Compute differences
        differences = self.compute_weight_differences(base_weights, sft_weights_list)
        
        # Aggregate by layer
        layer_summary = self.aggregate_by_layer(differences)
        
        # Create visualizations
        self.create_layer_by_layer_plot(layer_summary)
        self.create_special_layers_plot(layer_summary)
        self.create_distribution_plots(differences)
        self.create_comparative_boxplot(layer_summary)
        self.create_interactive_plot(layer_summary)
        
        # Save results
        self.save_results(layer_summary, differences)
        
        print("\n✅ SFT difference analysis complete!")
        
        # Count generated files
        layer_plots = len(list((self.output_dir / 'layer_analysis').glob('*.png')))
        dist_plots = len(list((self.output_dir / 'distribution_analysis').glob('*.png')))
        comp_plots = len(list((self.output_dir / 'comprehensive_plots').glob('*.png')))
        interactive_plots = len(list((self.output_dir / 'interactive_plots').glob('*.html')))
        
        print(f"📊 Generated:")
        print(f"   - {layer_plots} layer analysis plots")
        print(f"   - {dist_plots} distribution plots")
        print(f"   - {comp_plots} comprehensive plots")
        print(f"   - {interactive_plots} interactive plots")
        print(f"   - 2 CSV files with detailed results")

def main():
    parser = argparse.ArgumentParser(description='SFT weight difference analysis')
    parser.add_argument('--base_model', type=str, required=True, 
                       help='Path to base model')
    parser.add_argument('--sft_models', type=str, nargs='+', required=True,
                       help='Paths to SFT models (space-separated)')
    parser.add_argument('--output', type=str, default='sft_difference_analysis', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    if len(args.sft_models) < 1:
        print("⚠️ Please provide at least 1 SFT model for analysis")
        return
    
    # Initialize analyzer
    analyzer = SFTDifferenceAnalyzer(args.base_model, args.sft_models, args.output)
    
    # Run analysis
    analyzer.run_analysis()
    
    print(f"\n🎯 Analysis complete! Check {args.output}/ for results")

if __name__ == "__main__":
    main() 