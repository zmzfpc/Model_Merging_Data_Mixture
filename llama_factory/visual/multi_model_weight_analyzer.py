#!/usr/bin/env python3
"""
Multi-Model Weight Difference Analyzer

This script analyzes weight differences between a base model and multiple SFT models,
creating comprehensive visualizations including:
1. Layer-by-layer weight differences in a single plot
2. Weight difference distributions model by model
3. Comparative analysis across all models

Usage:
    python multi_model_weight_analyzer.py --base_model path/to/base --sft_models path/to/sft1 path/to/sft2 path/to/sft3
    python multi_model_weight_analyzer.py --base_model path/to/base --sft_models path/to/sft1 path/to/sft2 --output results_dir
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
from collections import defaultdict, OrderedDict
import warnings
from typing import Dict, Tuple, List, Optional, Union
import re
from scipy.stats import skew, kurtosis, wasserstein_distance
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import math
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Define color palettes
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
LAYER_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

class MultiModelWeightAnalyzer:
    def __init__(self, base_model_path: str, sft_model_paths: List[str], output_dir: str = "multi_model_analysis"):
        self.base_model_path = Path(base_model_path)
        self.sft_model_paths = [Path(p) for p in sft_model_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "layer_analysis").mkdir(exist_ok=True)
        (self.output_dir / "distribution_analysis").mkdir(exist_ok=True)
        (self.output_dir / "comparative_analysis").mkdir(exist_ok=True)
        (self.output_dir / "summary_tables").mkdir(exist_ok=True)
        
        print(f"🔍 Multi-Model Weight Analysis:")
        print(f"   Base Model: {self.base_model_path.name}")
        print(f"   SFT Models: {[p.name for p in self.sft_model_paths]}")
        print(f"   Output: {self.output_dir}")
        
        # Load configurations
        self.base_config = self._load_config(self.base_model_path)
        self.sft_configs = [self._load_config(path) for path in self.sft_model_paths]
        self.architecture = self._detect_architecture()
        
        # Verify architecture compatibility
        self._verify_compatibility()
        
        print(f"📐 Architecture: {self.architecture}")
        print(f"🔢 Layers: {self.base_config.get('num_hidden_layers', 'unknown')}")

    def _load_config(self, model_path: Path) -> dict:
        """Load model configuration"""
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def _detect_architecture(self) -> str:
        """Detect model architecture"""
        if "architectures" in self.base_config:
            arch = self.base_config["architectures"][0]
            if "llama" in arch.lower():
                return "LLaMA"
            elif "mistral" in arch.lower():
                return "Mistral"
            elif "qwen" in arch.lower():
                return "Qwen"
            elif "gemma" in arch.lower():
                return "Gemma"
            elif "granite" in arch.lower():
                return "Granite"
            else:
                return arch
        return "Unknown"

    def _verify_compatibility(self):
        """Verify all models have compatible architectures"""
        base_arch = self.base_config.get("architectures", [None])[0]
        for i, config in enumerate(self.sft_configs):
            sft_arch = config.get("architectures", [None])[0]
            if base_arch != sft_arch:
                raise ValueError(f"Architecture mismatch: Base model ({base_arch}) vs SFT model {i} ({sft_arch})")

    def _get_weight_files(self, model_path: Path) -> List[Path]:
        """Get all safetensors weight files for a model"""
        weight_files = []
        
        # Check for index file
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            weight_files = [model_path / filename for filename in set(index_data["weight_map"].values())]
        else:
            # Single file case
            single_file = model_path / "model.safetensors"
            if single_file.exists():
                weight_files = [single_file]
            else:
                # Check for pytorch_model.bin files
                bin_files = list(model_path.glob("pytorch_model*.bin"))
                if bin_files:
                    weight_files = bin_files
        
        return sorted(weight_files)

    def _load_weights(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """Load all weights from a model"""
        weights = {}
        weight_files = self._get_weight_files(model_path)
        
        print(f"Loading weights from {model_path.name}: {len(weight_files)} files")
        
        for weight_file in tqdm(weight_files, desc=f"Loading {model_path.name}"):
            if weight_file.suffix == '.safetensors':
                with safe_open(weight_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            elif weight_file.suffix == '.bin':
                checkpoint = torch.load(weight_file, map_location='cpu')
                weights.update(checkpoint)
        
        return weights

    def _get_layer_groups(self, weight_keys: List[str]) -> Dict[str, List[str]]:
        """Group weights by layer number and component type"""
        layer_groups = defaultdict(list)
        
        for key in weight_keys:
            # Extract layer number and component type
            if 'layers.' in key:
                # Pattern: model.layers.X.component.weight
                match = re.search(r'layers\.(\d+)\.([^.]+)', key)
                if match:
                    layer_num = int(match.group(1))
                    component = match.group(2)
                    layer_groups[f"layer_{layer_num:02d}_{component}"].append(key)
            elif any(x in key for x in ['embed', 'norm', 'lm_head']):
                # Handle special layers
                if 'embed' in key:
                    layer_groups['embeddings'].append(key)
                elif 'norm' in key:
                    layer_groups['norm'].append(key)
                elif 'lm_head' in key:
                    layer_groups['lm_head'].append(key)
        
        return dict(layer_groups)

    def _calculate_layer_differences(self, base_weights: Dict[str, torch.Tensor], 
                                   sft_weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Calculate various difference metrics for each layer group"""
        common_keys = set(base_weights.keys()) & set(sft_weights.keys())
        layer_groups = self._get_layer_groups(list(common_keys))
        
        layer_stats = {}
        
        for group_name, keys in tqdm(layer_groups.items(), desc="Calculating layer differences"):
            group_stats = {
                'l2_norm': 0.0,
                'l1_norm': 0.0,
                'max_abs_diff': 0.0,
                'mean_abs_diff': 0.0,
                'std_diff': 0.0,
                'cosine_sim': 0.0,
                'param_count': 0
            }
            
            all_base_weights = []
            all_sft_weights = []
            
            for key in keys:
                if key in base_weights and key in sft_weights:
                    base_tensor = base_weights[key].float()
                    sft_tensor = sft_weights[key].float()
                    
                    if base_tensor.shape != sft_tensor.shape:
                        continue
                    
                    diff = sft_tensor - base_tensor
                    
                    # Accumulate statistics
                    group_stats['l2_norm'] += torch.norm(diff, p=2).item() ** 2
                    group_stats['l1_norm'] += torch.norm(diff, p=1).item()
                    group_stats['max_abs_diff'] = max(group_stats['max_abs_diff'], torch.max(torch.abs(diff)).item())
                    group_stats['param_count'] += diff.numel()
                    
                    # Flatten for overall statistics
                    all_base_weights.extend(base_tensor.flatten().tolist())
                    all_sft_weights.extend(sft_tensor.flatten().tolist())
            
            if all_base_weights and all_sft_weights:
                all_base = np.array(all_base_weights)
                all_sft = np.array(all_sft_weights)
                all_diff = all_sft - all_base
                
                group_stats['l2_norm'] = np.sqrt(group_stats['l2_norm'])
                group_stats['mean_abs_diff'] = np.mean(np.abs(all_diff))
                group_stats['std_diff'] = np.std(all_diff)
                
                # Cosine similarity
                if np.linalg.norm(all_base) > 0 and np.linalg.norm(all_sft) > 0:
                    group_stats['cosine_sim'] = np.dot(all_base, all_sft) / (np.linalg.norm(all_base) * np.linalg.norm(all_sft))
                
                layer_stats[group_name] = group_stats
        
        return layer_stats

    def analyze_all_models(self):
        """Main analysis function"""
        print("\n🚀 Starting multi-model analysis...")
        
        # Load base model weights
        print("\n📥 Loading base model weights...")
        base_weights = self._load_weights(self.base_model_path)
        
        # Load all SFT model weights and calculate differences
        all_model_stats = {}
        all_model_weights = {}
        
        for i, sft_path in enumerate(self.sft_model_paths):
            model_name = sft_path.name
            print(f"\n📥 Loading SFT model weights: {model_name}")
            sft_weights = self._load_weights(sft_path)
            all_model_weights[model_name] = sft_weights
            
            print(f"🧮 Calculating differences for {model_name}...")
            model_stats = self._calculate_layer_differences(base_weights, sft_weights)
            all_model_stats[model_name] = model_stats
        
        # Generate visualizations
        print("\n🎨 Generating visualizations...")
        self._plot_layer_by_layer_comparison(all_model_stats)
        self._plot_distribution_comparison(base_weights, all_model_weights)
        self._plot_comparative_heatmap(all_model_stats)
        self._generate_summary_tables(all_model_stats)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")

    def _plot_layer_by_layer_comparison(self, all_model_stats: Dict[str, Dict[str, Dict[str, float]]]):
        """Plot layer-by-layer weight differences for all models in one plot"""
        print("📊 Creating layer-by-layer comparison plot...")
        
        # Prepare data
        metrics = ['l2_norm', 'mean_abs_diff', 'cosine_sim']
        metric_labels = ['L2 Norm', 'Mean Absolute Difference', 'Cosine Similarity']
        
        # Get common layers across all models
        all_layers = set()
        for model_stats in all_model_stats.values():
            all_layers.update(model_stats.keys())
        all_layers = sorted(list(all_layers))
        
        # Create subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        model_names = list(all_model_stats.keys())
        x_positions = np.arange(len(all_layers))
        width = 0.8 / len(model_names)
        
        for metric_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[metric_idx]
            
            for model_idx, model_name in enumerate(model_names):
                values = []
                for layer in all_layers:
                    if layer in all_model_stats[model_name]:
                        values.append(all_model_stats[model_name][layer][metric])
                    else:
                        values.append(0)
                
                offset = (model_idx - len(model_names)/2 + 0.5) * width
                bars = ax.bar(x_positions + offset, values, width, 
                            label=model_name, color=COLORS[model_idx % len(COLORS)], alpha=0.8)
            
            ax.set_xlabel('Layer')
            ax.set_ylabel(label)
            ax.set_title(f'{label} by Layer Across All Models')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([layer.replace('layer_', '').replace('_', '\n') for layer in all_layers], 
                              rotation=45, ha='right', fontsize=8)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "layer_analysis" / "layer_by_layer_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_distribution_comparison(self, base_weights: Dict[str, torch.Tensor], 
                                    all_model_weights: Dict[str, Dict[str, torch.Tensor]]):
        """Plot weight difference distributions for each model"""
        print("📊 Creating distribution comparison plots...")
        
        # Sample weights for distribution analysis (to avoid memory issues)
        sample_size = 100000
        
        for model_name, sft_weights in all_model_weights.items():
            print(f"  Processing distributions for {model_name}...")
            
            # Collect weight differences
            all_diffs = []
            common_keys = set(base_weights.keys()) & set(sft_weights.keys())
            
            for key in list(common_keys)[:20]:  # Limit to first 20 tensors for efficiency
                if key in base_weights and key in sft_weights:
                    base_tensor = base_weights[key].float()
                    sft_tensor = sft_weights[key].float()
                    
                    if base_tensor.shape == sft_tensor.shape:
                        diff = (sft_tensor - base_tensor).flatten()
                        all_diffs.extend(diff.tolist())
            
            if len(all_diffs) > sample_size:
                all_diffs = np.random.choice(all_diffs, sample_size, replace=False)
            
            # Create distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Weight Difference Distributions: {model_name}', fontsize=16)
            
            # Histogram
            axes[0, 0].hist(all_diffs, bins=100, alpha=0.7, density=True, color=COLORS[0])
            axes[0, 0].set_title('Weight Difference Histogram')
            axes[0, 0].set_xlabel('Weight Difference')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Log-scale histogram
            abs_diffs = np.abs(all_diffs)
            abs_diffs = abs_diffs[abs_diffs > 0]  # Remove zeros for log scale
            if len(abs_diffs) > 0:
                axes[0, 1].hist(abs_diffs, bins=100, alpha=0.7, density=True, color=COLORS[1])
                axes[0, 1].set_xscale('log')
                axes[0, 1].set_yscale('log')
                axes[0, 1].set_title('Absolute Weight Difference (Log Scale)')
                axes[0, 1].set_xlabel('|Weight Difference|')
                axes[0, 1].set_ylabel('Density')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Box plot by magnitude
            diff_magnitudes = np.abs(all_diffs)
            magnitude_bins = np.percentile(diff_magnitudes, [0, 25, 50, 75, 90, 95, 99, 100])
            binned_diffs = []
            bin_labels = []
            
            for i in range(len(magnitude_bins) - 1):
                mask = (diff_magnitudes >= magnitude_bins[i]) & (diff_magnitudes < magnitude_bins[i + 1])
                if np.sum(mask) > 0:
                    binned_diffs.append(np.array(all_diffs)[mask])
                    bin_labels.append(f'{magnitude_bins[i]:.2e}-{magnitude_bins[i+1]:.2e}')
            
            if binned_diffs:
                axes[1, 0].boxplot(binned_diffs, labels=bin_labels)
                axes[1, 0].set_title('Weight Differences by Magnitude Range')
                axes[1, 0].set_xlabel('Magnitude Range')
                axes[1, 0].set_ylabel('Weight Difference')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # Q-Q plot against normal distribution
            from scipy import stats
            stats.probplot(all_diffs, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot vs Normal Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "distribution_analysis" / f"{model_name}_distributions.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_comparative_heatmap(self, all_model_stats: Dict[str, Dict[str, Dict[str, float]]]):
        """Create comparative heatmap across all models and layers"""
        print("📊 Creating comparative heatmap...")
        
        # Prepare data for heatmap
        metrics = ['l2_norm', 'mean_abs_diff', 'cosine_sim']
        model_names = list(all_model_stats.keys())
        
        # Get all unique layers
        all_layers = set()
        for model_stats in all_model_stats.values():
            all_layers.update(model_stats.keys())
        all_layers = sorted(list(all_layers))
        
        for metric in metrics:
            # Create matrix
            data_matrix = np.zeros((len(model_names), len(all_layers)))
            
            for i, model_name in enumerate(model_names):
                for j, layer in enumerate(all_layers):
                    if layer in all_model_stats[model_name]:
                        data_matrix[i, j] = all_model_stats[model_name][layer][metric]
            
            # Create heatmap
            plt.figure(figsize=(max(12, len(all_layers) * 0.5), max(8, len(model_names) * 0.5)))
            
            # Handle different scales for different metrics
            if metric == 'cosine_sim':
                vmin, vmax = 0, 1
                cmap = 'RdYlBu_r'
            else:
                vmin, vmax = None, None
                cmap = 'viridis'
            
            sns.heatmap(data_matrix, 
                       xticklabels=[layer.replace('layer_', '').replace('_', '\n') for layer in all_layers],
                       yticklabels=model_names,
                       annot=True, fmt='.3f', cmap=cmap, 
                       vmin=vmin, vmax=vmax, cbar_kws={'label': metric})
            
            plt.title(f'{metric.replace("_", " ").title()} Across Models and Layers')
            plt.xlabel('Layer')
            plt.ylabel('Model')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / "comparative_analysis" / f"{metric}_heatmap.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _generate_summary_tables(self, all_model_stats: Dict[str, Dict[str, Dict[str, float]]]):
        """Generate summary tables and statistics"""
        print("📋 Generating summary tables...")
        
        # Overall model comparison
        model_summary = []
        for model_name, model_stats in all_model_stats.items():
            total_params = sum(stats['param_count'] for stats in model_stats.values())
            avg_l2 = np.mean([stats['l2_norm'] for stats in model_stats.values()])
            avg_l1 = np.mean([stats['l1_norm'] for stats in model_stats.values()])
            avg_cosine = np.mean([stats['cosine_sim'] for stats in model_stats.values()])
            max_diff = max([stats['max_abs_diff'] for stats in model_stats.values()])
            
            model_summary.append({
                'Model': model_name,
                'Total Parameters': total_params,
                'Average L2 Norm': avg_l2,
                'Average L1 Norm': avg_l1,
                'Average Cosine Similarity': avg_cosine,
                'Max Absolute Difference': max_diff
            })
        
        model_df = pd.DataFrame(model_summary)
        model_df.to_csv(self.output_dir / "summary_tables" / "model_comparison.csv", index=False)
        
        # Layer-wise detailed comparison
        all_layers = set()
        for model_stats in all_model_stats.values():
            all_layers.update(model_stats.keys())
        all_layers = sorted(list(all_layers))
        
        layer_details = []
        for layer in all_layers:
            for model_name, model_stats in all_model_stats.items():
                if layer in model_stats:
                    stats = model_stats[layer]
                    layer_details.append({
                        'Model': model_name,
                        'Layer': layer,
                        'L2 Norm': stats['l2_norm'],
                        'L1 Norm': stats['l1_norm'],
                        'Mean Abs Diff': stats['mean_abs_diff'],
                        'Std Diff': stats['std_diff'],
                        'Cosine Similarity': stats['cosine_sim'],
                        'Max Abs Diff': stats['max_abs_diff'],
                        'Parameter Count': stats['param_count']
                    })
        
        layer_df = pd.DataFrame(layer_details)
        layer_df.to_csv(self.output_dir / "summary_tables" / "layer_detailed_comparison.csv", index=False)
        
        # Generate summary report
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write("Multi-Model Weight Difference Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Base Model: {self.base_model_path.name}\n")
            f.write(f"SFT Models: {[p.name for p in self.sft_model_paths]}\n")
            f.write(f"Architecture: {self.architecture}\n\n")
            
            f.write("Model Comparison Summary:\n")
            f.write("-" * 30 + "\n")
            for _, row in model_df.iterrows():
                f.write(f"\nModel: {row['Model']}\n")
                f.write(f"  Total Parameters: {row['Total Parameters']:,}\n")
                f.write(f"  Average L2 Norm: {row['Average L2 Norm']:.6f}\n")
                f.write(f"  Average Cosine Similarity: {row['Average Cosine Similarity']:.6f}\n")
                f.write(f"  Max Absolute Difference: {row['Max Absolute Difference']:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Weight Difference Analyzer")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Path to the base model directory")
    parser.add_argument("--sft_models", type=str, nargs='+', required=True,
                       help="Paths to SFT model directories")
    parser.add_argument("--output", type=str, default="multi_model_analysis",
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Verify paths exist
    base_path = Path(args.base_model)
    if not base_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_path}")
    
    sft_paths = [Path(p) for p in args.sft_models]
    for sft_path in sft_paths:
        if not sft_path.exists():
            raise FileNotFoundError(f"SFT model path does not exist: {sft_path}")
    
    # Create analyzer and run analysis
    analyzer = MultiModelWeightAnalyzer(args.base_model, args.sft_models, args.output)
    analyzer.analyze_all_models()

if __name__ == "__main__":
    main()
