#!/usr/bin/env python3
"""
Gap Correlation Analyzer

Computes gaps (SFT - Base) and analyzes Pearson correlations between SFT models.
Focuses on how similarly different SFT models deviate from the base model.

Usage:
    python gap_correlation_analyzer.py --base_model path/to/base --sft_models path/to/sft1 path/to/sft2 path/to/sft3
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

class GapCorrelationAnalyzer:
    def __init__(self, base_model_path: str, sft_model_paths: List[str], output_dir: str = "gap_correlation_analysis", 
                 filter_small_gaps: bool = False, gap_threshold: float = 0.5):
        self.base_model_path_str = base_model_path
        self.sft_model_paths = [Path(p) for p in sft_model_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Gap filtering parameters
        self.filter_small_gaps = filter_small_gaps
        self.gap_threshold = gap_threshold
        
        print(f"🔗 Gap Correlation Analysis:")
        print(f"   Base Model: {base_model_path}")
        print(f"   SFT Models: {[p.name for p in self.sft_model_paths]}")
        print(f"   Output: {self.output_dir}")
        if filter_small_gaps:
            print(f"   🎯 Gap Filtering: Enabled (threshold: {gap_threshold*100:.1f}% of max magnitude)")
        else:
            print(f"   🎯 Gap Filtering: Disabled")
        
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

    def get_layer_label(self, layer_num: int) -> str:
        """Get human-readable label for layer number"""
        if layer_num == -1:
            return 'emb'
        elif layer_num == 998:
            return 'final_norm'
        elif layer_num == 999:
            return 'lm_head'
        else:
            return f'layer_{layer_num}'

    def _filter_gap_values(self, gap: np.ndarray) -> np.ndarray:
        """Filter small magnitude gap values, setting them to zero"""
        if not self.filter_small_gaps:
            return gap
        
        # Compute threshold as percentage of maximum absolute value
        max_abs_gap = np.max(np.abs(gap))
        threshold = self.gap_threshold * max_abs_gap
        
        # Create filtered gap: keep values with |gap| >= threshold, set others to 0
        filtered_gap = np.where(np.abs(gap) >= threshold, gap, 0.0)
        
        # Statistics for monitoring
        original_nonzero = np.count_nonzero(gap)
        filtered_nonzero = np.count_nonzero(filtered_gap)
        filtered_percentage = (original_nonzero - filtered_nonzero) / original_nonzero * 100 if original_nonzero > 0 else 0
        
        return filtered_gap

    def compute_gaps_and_correlations(self, base_weights: dict, sft_weights_list: List[dict]) -> Dict:
        """Compute gaps (SFT - Base) and their correlations"""
        print("🔗 Computing gaps and correlations between SFT models...")
        
        # Get common weight names across all models
        common_weights = set(base_weights.keys())
        for sft_weights in sft_weights_list:
            common_weights &= set(sft_weights.keys())
        
        print(f"   📊 Found {len(common_weights)} common weight tensors")
        
        # Statistics for gap filtering
        total_filtered_values = 0
        total_original_values = 0
        
        # Compute gaps for each SFT model
        sft_gaps = []
        for i, sft_weights in enumerate(sft_weights_list):
            model_name = self.sft_model_paths[i].name
            print(f"   📊 Computing gaps for {model_name}...")
            
            gaps = {}
            for weight_name in common_weights:
                layer_num, component, param_type = self.parse_layer_info(weight_name)
                if layer_num is not None and param_type == 'weight':
                    base_weight = base_weights[weight_name].float()
                    sft_weight = sft_weights[weight_name].float()
                    
                    # Compute raw gap
                    raw_gap = (sft_weight - base_weight).flatten().numpy()
                    
                    # Apply filtering if enabled
                    filtered_gap = self._filter_gap_values(raw_gap)
                    
                    # Update statistics
                    if self.filter_small_gaps:
                        total_original_values += len(raw_gap)
                        total_filtered_values += len(raw_gap) - np.count_nonzero(filtered_gap)
                    
                    gaps[weight_name] = {
                        'gap': filtered_gap,
                        'layer_num': layer_num,
                        'component': component,
                        'shape': list(base_weight.shape)
                    }
            
            sft_gaps.append({
                'model_idx': i,
                'model_name': model_name,
                'gaps': gaps
            })
        
        # Print filtering statistics
        if self.filter_small_gaps and total_original_values > 0:
            filtered_percentage = (total_filtered_values / total_original_values) * 100
            print(f"   🎯 Gap Filtering Statistics:")
            print(f"       Original values: {total_original_values:,}")
            print(f"       Filtered to zero: {total_filtered_values:,} ({filtered_percentage:.1f}%)")
            print(f"       Remaining non-zero: {total_original_values - total_filtered_values:,} ({100-filtered_percentage:.1f}%)")
        
        # Compute layer-wise correlations
        layer_correlations = self._compute_layer_correlations(sft_gaps)
        
        # Compute overall correlations
        overall_correlations = self._compute_overall_correlations(sft_gaps)
        
        return {
            'sft_gaps': sft_gaps,
            'layer_correlations': layer_correlations,
            'overall_correlations': overall_correlations,
            'model_names': [gap['model_name'] for gap in sft_gaps]
        }

    def _compute_layer_correlations(self, sft_gaps: List[Dict]) -> Dict:
        """Compute correlations layer by layer"""
        print("   🔗 Computing layer-wise correlations...")
        
        layer_correlations = {}
        
        # Get all layers
        all_layers = set()
        for gap_data in sft_gaps:
            for weight_name, gap_info in gap_data['gaps'].items():
                all_layers.add(gap_info['layer_num'])
        
        for layer_num in sorted(all_layers):
            # Include all layers including special ones (-1: embedding, 998: final_norm, 999: lm_head)
            # Skip only truly invalid layers
            if layer_num is None:
                continue
                
            layer_correlations[layer_num] = {}
            
            # Collect gaps for this layer from all models
            layer_gaps = []
            for gap_data in sft_gaps:
                layer_gap_combined = []
                for weight_name, gap_info in gap_data['gaps'].items():
                    if gap_info['layer_num'] == layer_num:
                        layer_gap_combined.extend(gap_info['gap'])
                layer_gaps.append({
                    'model_name': gap_data['model_name'],
                    'model_idx': gap_data['model_idx'],
                    'combined_gap': np.array(layer_gap_combined)
                })
            
            # Compute pairwise correlations for this layer
            n_models = len(layer_gaps)
            correlation_matrix = np.zeros((n_models, n_models))
            correlations = {}
            
            for i in range(n_models):
                for j in range(i, n_models):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                        continue
                    
                    model1_name = layer_gaps[i]['model_name']
                    model2_name = layer_gaps[j]['model_name']
                    gap1 = layer_gaps[i]['combined_gap']
                    gap2 = layer_gaps[j]['combined_gap']
                    
                    if len(gap1) > 0 and len(gap2) > 0 and len(gap1) == len(gap2):
                        try:
                            corr_coef, p_value = pearsonr(gap1, gap2)
                            correlation_matrix[i, j] = corr_coef
                            correlation_matrix[j, i] = corr_coef
                            
                            correlations[f"{model1_name}_vs_{model2_name}"] = {
                                'correlation': corr_coef,
                                'p_value': p_value,
                                'n_samples': len(gap1)
                            }
                        except:
                            correlation_matrix[i, j] = 0.0
                            correlation_matrix[j, i] = 0.0
            
            layer_correlations[layer_num] = {
                'correlations': correlations,
                'correlation_matrix': correlation_matrix,
                'model_names': [lg['model_name'] for lg in layer_gaps]
            }
        
        return layer_correlations

    def _compute_overall_correlations(self, sft_gaps: List[Dict]) -> Dict:
        """Compute overall correlations across all layers"""
        print("   🔗 Computing overall correlations...")
        
        correlations = {}
        n_models = len(sft_gaps)
        correlation_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i, n_models):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    continue
                
                model1_name = sft_gaps[i]['model_name']
                model2_name = sft_gaps[j]['model_name']
                
                # Collect all gap values for correlation
                gaps1_all = []
                gaps2_all = []
                
                for weight_name in sft_gaps[i]['gaps'].keys():
                    if weight_name in sft_gaps[j]['gaps']:
                        gap1 = sft_gaps[i]['gaps'][weight_name]['gap']
                        gap2 = sft_gaps[j]['gaps'][weight_name]['gap']
                        
                        gaps1_all.extend(gap1)
                        gaps2_all.extend(gap2)
                
                # Compute Pearson correlation
                if len(gaps1_all) > 1:
                    corr_coef, p_value = pearsonr(gaps1_all, gaps2_all)
                    correlation_matrix[i, j] = corr_coef
                    correlation_matrix[j, i] = corr_coef
                    
                    correlations[f"{model1_name}_vs_{model2_name}"] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n_samples': len(gaps1_all)
                    }
                    
                    print(f"     📈 {model1_name} vs {model2_name}: r={corr_coef:.4f} (p={p_value:.4f})")
        
        return {
            'correlations': correlations,
            'correlation_matrix': correlation_matrix,
            'model_names': [gap['model_name'] for gap in sft_gaps]
        }

    def create_layer_correlation_plot(self, layer_correlations: Dict, overall_correlations: Dict):
        """Create layer-by-layer correlation analysis plot"""
        print("📊 Creating layer-by-layer correlation analysis...")
        
        if len(layer_correlations) == 0:
            print("   ⚠️ No layer correlations to plot")
            return
        
        # Extract correlation data by layer (include special layers)
        all_layers = sorted(layer_correlations.keys())
        regular_layers = [l for l in all_layers if 0 <= l <= 50]
        special_layers = [l for l in all_layers if l < 0 or l > 50]
        layers = special_layers + regular_layers  # Show special layers first
        model_names = overall_correlations['model_names']
        n_models = len(model_names)
        
        if n_models < 2:
            print("   ⚠️ Need at least 2 SFT models for correlation analysis")
            return
        
        # Create one plot per model pair
        model_pairs = []
        for i in range(n_models):
            for j in range(i+1, n_models):
                model_pairs.append((i, j, model_names[i], model_names[j]))
        
        n_pairs = len(model_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_pairs > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for pair_idx, (i, j, model1, model2) in enumerate(model_pairs):
            if pair_idx >= len(axes):
                break
                
            ax = axes[pair_idx]
            
            # Extract correlations for this pair across layers
            correlations_by_layer = []
            p_values_by_layer = []
            
            for layer_num in layers:
                layer_corr_data = layer_correlations[layer_num]
                pair_key = f"{model1}_vs_{model2}"
                if pair_key in layer_corr_data['correlations']:
                    correlations_by_layer.append(layer_corr_data['correlations'][pair_key]['correlation'])
                    p_values_by_layer.append(layer_corr_data['correlations'][pair_key]['p_value'])
                else:
                    correlations_by_layer.append(0.0)
                    p_values_by_layer.append(1.0)
            
            # Create x-axis positions (use index for special layers, actual numbers for regular layers)
            x_positions = list(range(len(layers)))
            x_labels = [self.get_layer_label(layer) for layer in layers]
            
            # Plot correlation by layer
            line = ax.plot(x_positions, correlations_by_layer, 'o-', linewidth=2, markersize=6,
                          color=COLORS[pair_idx % len(COLORS)], label=f'{model1} vs {model2}')
            
            # Color points by significance
            colors = []
            for p_val in p_values_by_layer:
                if p_val < 0.001:
                    colors.append('darkgreen')
                elif p_val < 0.01:
                    colors.append('green')  
                elif p_val < 0.05:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            # Add significance markers
            for x_pos, corr, color in zip(x_positions, correlations_by_layer, colors):
                ax.scatter(x_pos, corr, c=color, s=50, alpha=0.8, edgecolors='white', linewidth=1)
            
            # Set x-axis labels
            ax.set_xticks(x_positions[::max(1, len(x_positions)//10)])  # Show every nth label to avoid crowding
            ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), max(1, len(x_labels)//10))], 
                              rotation=45, ha='right', fontsize=8)
            
            # Add overall correlation as horizontal line
            overall_pair_key = f"{model1}_vs_{model2}"
            if overall_pair_key in overall_correlations['correlations']:
                overall_corr = overall_correlations['correlations'][overall_pair_key]['correlation']
                ax.axhline(y=overall_corr, color=COLORS[pair_idx % len(COLORS)], 
                          linestyle='--', alpha=0.7, label=f'Overall: {overall_corr:.3f}')
            
            ax.set_title(f'{model1}\nvs\n{model2}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Layer Number')
            ax.set_ylabel('Pearson Correlation')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(model_pairs), len(axes)):
            axes[idx].set_visible(False)
        
        # Add significance legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='p < 0.001'),
            Patch(facecolor='green', label='p < 0.01'),
            Patch(facecolor='orange', label='p < 0.05'), 
            Patch(facecolor='red', label='p ≥ 0.05')
        ]
        
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
        
        filter_note = f"\nGap Filtering: {self.gap_threshold*100:.0f}% threshold" if self.filter_small_gaps else ""
        plt.suptitle(f'Layer-by-Layer Gap Correlations\n(How similarly do models deviate by layer?){filter_note}\nIncludes: emb=embedding, final_norm=final normalization, lm_head=language model head', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # Save plot in PDF format
        output_path = self.output_dir / 'layer_by_layer_correlations.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer-by-layer correlation plot (PDF)")

    def create_overall_correlation_plot(self, overall_correlations: Dict):
        """Create overall correlation analysis plot"""
        print("📊 Creating overall correlation analysis...")
        
        correlations = overall_correlations['correlations']
        correlation_matrix = overall_correlations['correlation_matrix']
        model_names = overall_correlations['model_names']
        n_models = len(model_names)
        
        if n_models < 2:
            print("   ⚠️ Need at least 2 SFT models for correlation analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Correlation Matrix Heatmap
        ax1 = axes[0, 0]
        im = ax1.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(n_models))
        ax1.set_yticks(range(n_models))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_yticklabels(model_names)
        ax1.set_title('Overall Gap Correlation Matrix\n(Pearson r)', fontweight='bold')
        
        # Add correlation values as text
        for i in range(n_models):
            for j in range(n_models):
                text = ax1.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                               ha="center", va="center", 
                               color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                               fontweight='bold')
        
        plt.colorbar(im, ax=ax1, label='Pearson Correlation')
        
        # 2. Correlation Distribution
        ax2 = axes[0, 1]
        corr_values = [data['correlation'] for data in correlations.values()]
        
        if corr_values:
            ax2.hist(corr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(corr_values), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(corr_values):.3f}')
            ax2.set_xlabel('Pearson Correlation')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Gap Correlations', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Correlation Significance
        ax3 = axes[1, 0]
        model_pairs = []
        correlations_list = []
        colors = []
        
        for key, data in correlations.items():
            model_pairs.append(key.replace('_vs_', '\nvs\n'))
            correlations_list.append(data['correlation'])
            # Color by significance
            if data['p_value'] < 0.001:
                colors.append('darkgreen')
            elif data['p_value'] < 0.01:
                colors.append('green')
            elif data['p_value'] < 0.05:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax3.bar(range(len(correlations_list)), correlations_list, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(model_pairs)))
        ax3.set_xticklabels(model_pairs, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Pearson Correlation')
        ax3.set_title('Pairwise Gap Correlations\n(Color: significance level)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add significance legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='p < 0.001'),
            Patch(facecolor='green', label='p < 0.01'),
            Patch(facecolor='orange', label='p < 0.05'),
            Patch(facecolor='red', label='p ≥ 0.05')
        ]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # 4. Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        for key, data in correlations.items():
            models = key.split('_vs_')
            significance = '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else 'ns'
            table_data.append([
                f"{models[0][:10]}...",
                f"{models[1][:10]}...",
                f"{data['correlation']:.4f}",
                f"{data['p_value']:.4f}",
                significance,
                f"{data['n_samples']:,}"
            ])
        
        if table_data:
            table = ax4.table(cellText=table_data,
                             colLabels=['Model 1', 'Model 2', 'Correlation', 'p-value', 'Sig', 'N'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            
            # Color rows by significance
            for i, (_, data) in enumerate(correlations.items()):
                if data['p_value'] < 0.001:
                    color = '#e8f5e8'  # Light green
                elif data['p_value'] < 0.01:
                    color = '#f0f8f0'  # Very light green
                elif data['p_value'] < 0.05:
                    color = '#fff8e8'  # Light orange
                else:
                    color = '#ffe8e8'  # Light red
                
                for j in range(6):
                    table[(i+1, j)].set_facecolor(color)
        
        ax4.set_title('Gap Correlation Summary\n(*** p<0.001, ** p<0.01, * p<0.05)', 
                     fontweight='bold', pad=20)
        
        filter_note = f" (Gap Filtering: {self.gap_threshold*100:.0f}% threshold)" if self.filter_small_gaps else ""
        plt.suptitle(f'Overall SFT Gap Correlation Analysis{filter_note}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot in PDF format
        output_path = self.output_dir / 'overall_correlations.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved overall correlation analysis plot (PDF)")

    def save_results(self, results: Dict):
        """Save correlation results to files"""
        print("💾 Saving correlation results...")
        
        # Save overall correlations
        overall_df = pd.DataFrame([
            {
                'model_1': key.split('_vs_')[0],
                'model_2': key.split('_vs_')[1],
                'correlation': data['correlation'],
                'p_value': data['p_value'],
                'n_samples': data['n_samples'],
                'significant': data['p_value'] < 0.05,
                'significance_level': '***' if data['p_value'] < 0.001 else '**' if data['p_value'] < 0.01 else '*' if data['p_value'] < 0.05 else 'ns'
            }
            for key, data in results['overall_correlations']['correlations'].items()
        ])
        
        overall_df.to_csv(self.output_dir / 'overall_correlations.csv', index=False)
        
        # Save layer-wise correlations (detailed format)
        layer_corr_data = []
        for layer_num, layer_data in results['layer_correlations'].items():
            for key, corr_data in layer_data['correlations'].items():
                layer_corr_data.append({
                    'layer_num': layer_num,
                    'layer_label': self.get_layer_label(layer_num),
                    'model_1': key.split('_vs_')[0],
                    'model_2': key.split('_vs_')[1],
                    'correlation': corr_data['correlation'],
                    'p_value': corr_data['p_value'],
                    'n_samples': corr_data['n_samples'],
                    'significant': corr_data['p_value'] < 0.05,
                    'significance_level': '***' if corr_data['p_value'] < 0.001 else '**' if corr_data['p_value'] < 0.01 else '*' if corr_data['p_value'] < 0.05 else 'ns'
                })
        
        layer_df = pd.DataFrame(layer_corr_data)
        layer_df.to_csv(self.output_dir / 'layer_by_layer_correlations.csv', index=False)
        
        # Save correlation matrix format for easy viewing
        model_names = results['overall_correlations']['model_names']
        layers = sorted(results['layer_correlations'].keys())  # Include all layers including special ones
        
        # Create correlation matrix CSV for each model pair
        if len(model_names) >= 2:
            matrix_data = []
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Only upper triangle
                        row_data = {'model_pair': f"{model1}_vs_{model2}"}
                        
                        # Add overall correlation
                        pair_key = f"{model1}_vs_{model2}"
                        if pair_key in results['overall_correlations']['correlations']:
                            row_data['overall_correlation'] = results['overall_correlations']['correlations'][pair_key]['correlation']
                            row_data['overall_p_value'] = results['overall_correlations']['correlations'][pair_key]['p_value']
                        else:
                            row_data['overall_correlation'] = np.nan
                            row_data['overall_p_value'] = np.nan
                        
                        # Add layer-wise correlations
                        for layer_num in layers:
                            layer_label = self.get_layer_label(layer_num)
                            if layer_num in results['layer_correlations']:
                                layer_corrs = results['layer_correlations'][layer_num]['correlations']
                                if pair_key in layer_corrs:
                                    row_data[f'{layer_label}_corr'] = layer_corrs[pair_key]['correlation']
                                    row_data[f'{layer_label}_pval'] = layer_corrs[pair_key]['p_value']
                                else:
                                    row_data[f'{layer_label}_corr'] = np.nan
                                    row_data[f'{layer_label}_pval'] = np.nan
                            else:
                                row_data[f'{layer_label}_corr'] = np.nan
                                row_data[f'{layer_label}_pval'] = np.nan
                        
                        matrix_data.append(row_data)
            
            matrix_df = pd.DataFrame(matrix_data)
            matrix_df.to_csv(self.output_dir / 'correlation_matrix_by_layer.csv', index=False)
            
            print(f"   💾 Saved 3 correlation CSV files:")
            print(f"       - overall_correlations.csv")
            print(f"       - layer_by_layer_correlations.csv") 
            print(f"       - correlation_matrix_by_layer.csv")
        else:
            print(f"   💾 Saved 2 correlation CSV files:")
            print(f"       - overall_correlations.csv")
            print(f"       - layer_by_layer_correlations.csv")

    def run_analysis(self):
        """Run the complete gap correlation analysis"""
        print("\n🚀 Starting gap correlation analysis...")
        
        # Load weights
        print("📥 Loading base model weights...")
        base_weights = self.load_weights(self.base_model_path, self.base_weight_map)
        
        print("📥 Loading SFT model weights...")
        sft_weights_list = []
        for i, (sft_path, sft_map) in enumerate(zip(self.sft_model_paths, self.sft_weight_maps)):
            print(f"   Loading SFT model {i+1}/{len(self.sft_model_paths)}: {sft_path.name}")
            sft_weights = self.load_weights(sft_path, sft_map)
            sft_weights_list.append(sft_weights)
        
        # Compute gaps and correlations
        results = self.compute_gaps_and_correlations(base_weights, sft_weights_list)
        
        # Create visualizations
        self.create_layer_correlation_plot(results['layer_correlations'], results['overall_correlations'])
        self.create_overall_correlation_plot(results['overall_correlations'])
        
        # Save results
        self.save_results(results)
        
        print("\n✅ Gap correlation analysis complete!")
        
        # Count generated files
        plot_files = len(list(self.output_dir.glob('*.pdf')))
        csv_files = len(list(self.output_dir.glob('*.csv')))
        
        print(f"📊 Generated:")
        print(f"   - {plot_files} correlation plots (PDF format)")
        print(f"   - {csv_files} CSV files with detailed results")

def main():
    parser = argparse.ArgumentParser(description='Gap correlation analysis between SFT models')
    parser.add_argument('--base_model', type=str, required=True, 
                       help='Path to base model')
    parser.add_argument('--sft_models', type=str, nargs='+', required=True,
                       help='Paths to SFT models (space-separated)')
    parser.add_argument('--output', type=str, default='gap_correlation_analysis', 
                       help='Output directory')
    parser.add_argument('--filter_small_gaps', action='store_true',
                       help='Filter small magnitude gap values (default: False)')
    parser.add_argument('--gap_threshold', type=float, default=0.5,
                       help='Threshold for filtering small gap values as fraction of max magnitude (default: 0.5 = 50%%)')
    
    args = parser.parse_args()
    
    if len(args.sft_models) < 2:
        print("⚠️ Please provide at least 2 SFT models for correlation analysis")
        return
    
    # Initialize analyzer
    analyzer = GapCorrelationAnalyzer(args.base_model, args.sft_models, args.output, 
                                     filter_small_gaps=args.filter_small_gaps, 
                                     gap_threshold=args.gap_threshold)
    
    # Run analysis
    analyzer.run_analysis()
    
    print(f"\n🎯 Gap correlation analysis complete! Check {args.output}/ for results")

if __name__ == "__main__":
    main() 