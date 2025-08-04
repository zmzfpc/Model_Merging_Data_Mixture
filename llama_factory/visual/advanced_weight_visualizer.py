#!/usr/bin/env python3
"""
Advanced Weight Visualization Analyzer

Comprehensive visualization of model weights before/after SFT using advanced methods:
- Histograms & KDE overlays
- 2D scatter plots & Hexbin plots
- Spectral analysis (SVD, eigenvalues)
- Weight distribution analysis
- Statistical manifold analysis

Usage:
    python advanced_weight_visualizer.py --model1 path/to/model1 --model2 path/to/model2
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
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set plotting style with vibrant colors
plt.style.use('default')
sns.set_style("whitegrid")
# Define custom color palette
COLORS = {
    'model1': '#1f77b4',  # Blue
    'model2': '#ff7f0e',  # Orange  
    'diff': '#2ca02c',    # Green
    'kde1': '#d62728',    # Red
    'kde2': '#9467bd',    # Purple
    'highlight': '#ffff00', # Yellow
    'reference': '#17becf'  # Cyan
}
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

class AdvancedWeightVisualizer:
    def __init__(self, model1_path: str, model2_path: str, output_dir: str = "advanced_weight_analysis"):
        self.model1_path = Path(model1_path)
        self.model2_path = Path(model2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different analysis types
        (self.output_dir / "histograms_kde").mkdir(exist_ok=True)
        (self.output_dir / "scatter_hexbin").mkdir(exist_ok=True)
        (self.output_dir / "spectral_analysis").mkdir(exist_ok=True)
        (self.output_dir / "manifold_analysis").mkdir(exist_ok=True)
        (self.output_dir / "interactive_plots").mkdir(exist_ok=True)
        
        print(f"🔬 Advanced Weight Analysis:")
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

    def create_histogram_kde_analysis(self, weights1: dict, weights2: dict):
        """Create histogram and KDE overlay analysis"""
        print("📊 Creating histogram & KDE analysis...")
        
        # Organize weights by layer and component
        layer_weights = defaultdict(list)
        for weight_name in set(weights1.keys()) & set(weights2.keys()):
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 30 and 
                param_type == 'weight' and weights1[weight_name].numel() > 100):
                layer_weights[layer_num].append((weight_name, component))
        
        # Create analysis for selected layers
        sample_layers = sorted(layer_weights.keys())[::2]  # Every 4th layer
        
        for layer_num in sample_layers:
            self._create_layer_histogram_kde(layer_num, layer_weights[layer_num], weights1, weights2)

    def _create_layer_histogram_kde(self, layer_num: int, layer_weights: List, weights1: dict, weights2: dict):
        """Create histogram and KDE plots for a single layer"""
        
        if not layer_weights:
            return
        
        n_weights = len(layer_weights)
        n_cols = min(3, n_weights)
        n_rows = (n_weights + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
        if n_weights == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_weights > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (weight_name, component) in enumerate(layer_weights):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            w1 = weights1[weight_name].flatten().float().numpy()
            w2 = weights2[weight_name].flatten().float().numpy()
            
            # Sample large arrays for performance
            if len(w1) > 50000:
                idx = np.random.choice(len(w1), 50000, replace=False)
                w1 = w1[idx]
                w2 = w2[idx]
            
            # Create histograms with distinct colors
            ax.hist(w1, bins=50, alpha=0.7, density=True, color=COLORS['model1'], label='Model 1', edgecolor='white', linewidth=0.5)
            ax.hist(w2, bins=50, alpha=0.7, density=True, color=COLORS['model2'], label='Model 2', edgecolor='white', linewidth=0.5)
            
            # Add KDE overlays
            try:
                kde1 = gaussian_kde(w1)
                kde2 = gaussian_kde(w2)
                
                x_range = np.linspace(min(w1.min(), w2.min()), max(w1.max(), w2.max()), 200)
                ax.plot(x_range, kde1(x_range), color=COLORS['kde1'], linewidth=3, label='KDE Model 1', linestyle='-')
                ax.plot(x_range, kde2(x_range), color=COLORS['kde2'], linewidth=3, label='KDE Model 2', linestyle='-')
            except:
                pass  # Skip KDE if computation fails
            
            ax.set_title(f'{component.upper()}: {weight_name.split(".")[-2]}')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistical comparison
            ks_stat, ks_p = stats.ks_2samp(w1, w2)
            wasserstein_dist = stats.wasserstein_distance(w1, w2)
            
            stats_text = f'KS stat: {ks_stat:.4f}\n'
            stats_text += f'KS p-val: {ks_p:.4f}\n'
            stats_text += f'Wasserstein: {wasserstein_dist:.4f}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(layer_weights), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Layer {layer_num} - Histogram & KDE Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "histograms_kde" / f'layer_{layer_num:02d}_hist_kde.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved layer {layer_num} histogram & KDE plot")

    def create_scatter_hexbin_analysis(self, weights1: dict, weights2: dict):
        """Create 2D scatter and hexbin plots"""
        print("🔶 Creating scatter & hexbin analysis...")
        
        # Collect weight pairs for different components
        component_pairs = defaultdict(list)
        
        for weight_name in set(weights1.keys()) & set(weights2.keys()):
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 30 and 
                param_type == 'weight' and weights1[weight_name].numel() > 100):
                
                w1 = weights1[weight_name].flatten().float().numpy()
                w2 = weights2[weight_name].flatten().float().numpy()
                
                # Sample for performance
                if len(w1) > 10000:
                    idx = np.random.choice(len(w1), 10000, replace=False)
                    w1 = w1[idx]
                    w2 = w2[idx]
                
                component_pairs[component].extend(list(zip(w1, w2)))
        
        # Create scatter/hexbin plots for each component
        for component, pairs in component_pairs.items():
            if len(pairs) > 1000:  # Only create plots for components with enough data
                self._create_component_scatter_hexbin(component, pairs)

    def _create_component_scatter_hexbin(self, component: str, weight_pairs: List):
        """Create scatter and hexbin plots for a component"""
        
        # Convert to numpy arrays
        pairs = np.array(weight_pairs)
        w1_vals = pairs[:, 0]
        w2_vals = pairs[:, 1]
        
        # Sample for plotting performance
        if len(w1_vals) > 50000:
            idx = np.random.choice(len(w1_vals), 50000, replace=False)
            w1_vals = w1_vals[idx]
            w2_vals = w2_vals[idx]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Regular scatter plot with vibrant colors
        ax1.scatter(w1_vals, w2_vals, alpha=0.4, s=1, c=COLORS['model1'], edgecolors='none')
        ax1.plot([w1_vals.min(), w1_vals.max()], [w1_vals.min(), w1_vals.max()], 
                color=COLORS['reference'], linestyle='--', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Model 1 Weights')
        ax1.set_ylabel('Model 2 Weights')
        ax1.set_title('Scatter Plot')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation
        correlation = np.corrcoef(w1_vals, w2_vals)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Hexbin plot with better colormap
        hb = ax2.hexbin(w1_vals, w2_vals, gridsize=50, cmap='plasma', mincnt=1, alpha=0.8)
        ax2.plot([w1_vals.min(), w1_vals.max()], [w1_vals.min(), w1_vals.max()], 
                color=COLORS['reference'], linestyle='--', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Model 1 Weights')
        ax2.set_ylabel('Model 2 Weights')
        ax2.set_title('Hexbin Plot')
        plt.colorbar(hb, ax=ax2, label='Count')
        
        # 3. Density scatter with KDE
        try:
            xy = np.vstack([w1_vals, w2_vals])
            density = gaussian_kde(xy)(xy)
            
            scatter = ax3.scatter(w1_vals, w2_vals, c=density, s=2, cmap='viridis', alpha=0.8)
            ax3.plot([w1_vals.min(), w1_vals.max()], [w1_vals.min(), w1_vals.max()], 
                    color=COLORS['reference'], linestyle='--', linewidth=2, alpha=0.8)
            ax3.set_xlabel('Model 1 Weights')
            ax3.set_ylabel('Model 2 Weights')
            ax3.set_title('Density Scatter (KDE)')
            plt.colorbar(scatter, ax=ax3, label='Density')
        except:
            ax3.text(0.5, 0.5, 'KDE calculation failed', transform=ax3.transAxes, ha='center')
        
        # 4. Residual plot with distinct colors
        residuals = w2_vals - w1_vals
        ax4.scatter(w1_vals, residuals, alpha=0.4, s=1, c=COLORS['diff'], edgecolors='none')
        ax4.axhline(y=0, color=COLORS['reference'], linestyle='--', linewidth=2, alpha=0.8)
        ax4.set_xlabel('Model 1 Weights')
        ax4.set_ylabel('Residuals (Model2 - Model1)')
        ax4.set_title('Residual Plot')
        ax4.grid(True, alpha=0.3)
        
        # Add residual statistics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ax4.text(0.05, 0.95, f'RMSE: {rmse:.6f}\nMAE: {mae:.6f}', 
                transform=ax4.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle(f'{component.upper()} Component - Scatter & Hexbin Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "scatter_hexbin" / f'{component}_scatter_hexbin.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🔶 Saved {component} scatter & hexbin plot")

    def create_spectral_analysis(self, weights1: dict, weights2: dict):
        """Create spectral analysis using SVD and eigenvalue decomposition"""
        print("🌈 Creating spectral analysis...")
        
        # Collect weight matrices for spectral analysis
        spectral_data = []
        total_weights = 0
        filtered_weights = 0
        
        for weight_name in set(weights1.keys()) & set(weights2.keys()):
            total_weights += 1
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 30 and 
                param_type == 'weight' and len(weights1[weight_name].shape) >= 2):  # Allow any multi-dimensional
                
                w1 = weights1[weight_name].float().numpy()
                w2 = weights2[weight_name].float().numpy()
                
                # Make 2D if needed (flatten extra dimensions)
                if len(w1.shape) > 2:
                    w1 = w1.reshape(w1.shape[0], -1)
                    w2 = w2.reshape(w2.shape[0], -1)
                
                # Be more permissive with size limit
                if w1.size < 5000000:  # 5M parameters instead of 1M
                    spectral_data.append((layer_num, component, weight_name, w1, w2))
                    filtered_weights += 1
        
        print(f"   🔍 Found {filtered_weights}/{total_weights} matrices suitable for spectral analysis")
        
        if len(spectral_data) == 0:
            print("   ⚠️ No matrices found for spectral analysis")
            return
        
        # Sample layers for analysis
        sample_data = spectral_data[::max(1, len(spectral_data)//12)]  # Sample ~12 matrices
        
        for layer_num, component, weight_name, w1, w2 in sample_data:
            self._create_matrix_spectral_analysis(layer_num, component, weight_name, w1, w2)
        
        # Create overall spectral summary
        self._create_spectral_summary(spectral_data)

    def _create_matrix_spectral_analysis(self, layer_num: int, component: str, weight_name: str, w1: np.ndarray, w2: np.ndarray):
        """Create spectral analysis for a single matrix"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Singular values comparison
        try:
            u1, s1, v1 = np.linalg.svd(w1, full_matrices=False)
            u2, s2, v2 = np.linalg.svd(w2, full_matrices=False)
            
            axes[0, 0].semilogy(s1, color=COLORS['model1'], label='Model 1', linewidth=3, marker='o', markersize=2)
            axes[0, 0].semilogy(s2, color=COLORS['model2'], label='Model 2', linewidth=3, marker='s', markersize=2)
            axes[0, 0].set_title('Singular Values')
            axes[0, 0].set_xlabel('Index')
            axes[0, 0].set_ylabel('Singular Value (log scale)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Spectral norm comparison
            spec_norm_1 = s1[0]
            spec_norm_2 = s2[0]
            axes[0, 0].text(0.05, 0.95, f'Spec norm 1: {spec_norm_1:.4f}\nSpec norm 2: {spec_norm_2:.4f}', 
                           transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'SVD failed: {str(e)}', ha='center', va='center')
        
        # 2. Rank analysis
        try:
            rank1 = np.linalg.matrix_rank(w1)
            rank2 = np.linalg.matrix_rank(w2)
            
            # Effective rank (based on singular value decay)
            s1_normalized = s1 / s1[0]
            s2_normalized = s2 / s2[0]
            
            eff_rank_1 = np.sum(s1_normalized > 0.01)  # Threshold at 1%
            eff_rank_2 = np.sum(s2_normalized > 0.01)
            
            rank_data = {
                'Model': ['Model 1', 'Model 2'],
                'Full Rank': [rank1, rank2],
                'Effective Rank': [eff_rank_1, eff_rank_2]
            }
            
            x_pos = np.arange(2)
            width = 0.35
            
            axes[0, 1].bar(x_pos - width/2, [rank1, rank2], width, label='Full Rank', 
                           color=[COLORS['model1'], COLORS['model2']], alpha=0.8, edgecolor='white', linewidth=1)
            axes[0, 1].bar(x_pos + width/2, [eff_rank_1, eff_rank_2], width, label='Effective Rank', 
                           color=[COLORS['kde1'], COLORS['kde2']], alpha=0.8, edgecolor='white', linewidth=1)
            axes[0, 1].set_title('Rank Analysis')
            axes[0, 1].set_xlabel('Model')
            axes[0, 1].set_ylabel('Rank')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(['Model 1', 'Model 2'])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Rank analysis failed: {str(e)}', ha='center', va='center')
        
        # 3. Condition number
        try:
            cond1 = np.linalg.cond(w1)
            cond2 = np.linalg.cond(w2)
            
            axes[0, 2].bar(['Model 1', 'Model 2'], [cond1, cond2], alpha=0.8, 
                           color=[COLORS['model1'], COLORS['model2']], edgecolor='white', linewidth=1)
            axes[0, 2].set_title('Condition Number')
            axes[0, 2].set_ylabel('Condition Number (log scale)')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Add values as text
            for i, (model, cond) in enumerate(zip(['Model 1', 'Model 2'], [cond1, cond2])):
                axes[0, 2].text(i, cond * 1.1, f'{cond:.2e}', ha='center', va='bottom')
                
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'Condition number failed: {str(e)}', ha='center', va='center')
        
        # 4. Frobenius norm comparison
        frob1 = np.linalg.norm(w1, 'fro')
        frob2 = np.linalg.norm(w2, 'fro')
        frob_diff = np.linalg.norm(w2 - w1, 'fro')
        
        axes[1, 0].bar(['Model 1', 'Model 2', 'Difference'], [frob1, frob2, frob_diff], 
                      alpha=0.8, color=[COLORS['model1'], COLORS['model2'], COLORS['diff']], 
                      edgecolor='white', linewidth=1)
        axes[1, 0].set_title('Frobenius Norms')
        axes[1, 0].set_ylabel('Frobenius Norm')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add values as text
        for i, (label, val) in enumerate(zip(['Model 1', 'Model 2', 'Difference'], [frob1, frob2, frob_diff])):
            axes[1, 0].text(i, val * 1.05, f'{val:.4f}', ha='center', va='bottom')
        
        # 5. Weight matrix difference heatmap (sampled)
        diff_matrix = w2 - w1
        if diff_matrix.size > 10000:
            # Sample the matrix
            step_r = max(1, diff_matrix.shape[0] // 100)
            step_c = max(1, diff_matrix.shape[1] // 100)
            diff_matrix_sample = diff_matrix[::step_r, ::step_c]
        else:
            diff_matrix_sample = diff_matrix
        
        im = axes[1, 1].imshow(diff_matrix_sample, cmap='RdBu_r', aspect='auto',
                              vmin=-np.abs(diff_matrix_sample).max(), 
                              vmax=np.abs(diff_matrix_sample).max())
        axes[1, 1].set_title('Weight Difference Matrix')
        axes[1, 1].set_xlabel('Input Dimension')
        axes[1, 1].set_ylabel('Output Dimension')
        plt.colorbar(im, ax=axes[1, 1])
        
        # 6. Eigenvalue analysis (for square matrices or use covariance)
        try:
            if w1.shape[0] == w1.shape[1] and w1.shape[0] < 1000:  # Square and not too large
                eig1 = np.linalg.eigvals(w1)
                eig2 = np.linalg.eigvals(w2)
            else:
                # Use covariance for non-square matrices
                cov1 = np.cov(w1)
                cov2 = np.cov(w2)
                eig1 = np.linalg.eigvals(cov1)
                eig2 = np.linalg.eigvals(cov2)
            
            # Sort eigenvalues
            eig1 = np.sort(np.real(eig1))[::-1]
            eig2 = np.sort(np.real(eig2))[::-1]
            
            axes[1, 2].plot(eig1, color=COLORS['model1'], label='Model 1', linewidth=3, marker='o', markersize=2)
            axes[1, 2].plot(eig2, color=COLORS['model2'], label='Model 2', linewidth=3, marker='s', markersize=2)
            axes[1, 2].set_title('Eigenvalues')
            axes[1, 2].set_xlabel('Index')
            axes[1, 2].set_ylabel('Eigenvalue')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f'Eigenvalue analysis failed: {str(e)}', ha='center', va='center')
        
        plt.suptitle(f'Spectral Analysis: Layer {layer_num} {component.upper()} - {weight_name.split(".")[-2]}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "spectral_analysis" / f'layer_{layer_num:02d}_{component}_spectral.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🌈 Saved layer {layer_num} {component} spectral analysis")

    def _create_spectral_summary(self, spectral_data: List):
        """Create overall spectral analysis summary"""
        
        # Collect spectral properties
        layer_nums = []
        components = []
        spec_norms_1 = []
        spec_norms_2 = []
        cond_nums_1 = []
        cond_nums_2 = []
        frob_norms_1 = []
        frob_norms_2 = []
        
        for layer_num, component, weight_name, w1, w2 in spectral_data:
            try:
                u1, s1, v1 = np.linalg.svd(w1, full_matrices=False)
                u2, s2, v2 = np.linalg.svd(w2, full_matrices=False)
                
                layer_nums.append(layer_num)
                components.append(component)
                spec_norms_1.append(s1[0])
                spec_norms_2.append(s2[0])
                cond_nums_1.append(np.linalg.cond(w1))
                cond_nums_2.append(np.linalg.cond(w2))
                frob_norms_1.append(np.linalg.norm(w1, 'fro'))
                frob_norms_2.append(np.linalg.norm(w2, 'fro'))
                
            except:
                continue
        
        if not layer_nums:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Spectral norms by layer with distinct colors
        axes[0, 0].plot(layer_nums, spec_norms_1, color=COLORS['model1'], marker='o', 
                       label='Model 1', markersize=6, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        axes[0, 0].plot(layer_nums, spec_norms_2, color=COLORS['model2'], marker='s', 
                       label='Model 2', markersize=6, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        axes[0, 0].set_title('Spectral Norms Across Layers')
        axes[0, 0].set_xlabel('Layer Number')
        axes[0, 0].set_ylabel('Spectral Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Condition numbers by layer with distinct colors
        axes[0, 1].semilogy(layer_nums, cond_nums_1, color=COLORS['model1'], marker='o', 
                           label='Model 1', markersize=6, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        axes[0, 1].semilogy(layer_nums, cond_nums_2, color=COLORS['model2'], marker='s', 
                           label='Model 2', markersize=6, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        axes[0, 1].set_title('Condition Numbers Across Layers')
        axes[0, 1].set_xlabel('Layer Number')
        axes[0, 1].set_ylabel('Condition Number (log scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Frobenius norms by layer with distinct colors
        axes[1, 0].plot(layer_nums, frob_norms_1, color=COLORS['model1'], marker='o', 
                       label='Model 1', markersize=6, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        axes[1, 0].plot(layer_nums, frob_norms_2, color=COLORS['model2'], marker='s', 
                       label='Model 2', markersize=6, linewidth=2, markerfacecolor='white', markeredgewidth=2)
        axes[1, 0].set_title('Frobenius Norms Across Layers')
        axes[1, 0].set_xlabel('Layer Number')
        axes[1, 0].set_ylabel('Frobenius Norm')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Component comparison
        component_types = list(set(components))
        spec_norm_ratios = np.array(spec_norms_2) / np.array(spec_norms_1)
        
        comp_ratios = defaultdict(list)
        for comp, ratio in zip(components, spec_norm_ratios):
            comp_ratios[comp].append(ratio)
        
        comp_means = [np.mean(comp_ratios[comp]) for comp in component_types]
        comp_stds = [np.std(comp_ratios[comp]) for comp in component_types]
        
        # Use different colors for different components
        colors = [COLORS['model1'], COLORS['model2'], COLORS['diff']][:len(component_types)]
        
        axes[1, 1].bar(component_types, comp_means, yerr=comp_stds, capsize=5, alpha=0.8,
                      color=colors, edgecolor='white', linewidth=1)
        axes[1, 1].axhline(y=1.0, color=COLORS['reference'], linestyle='--', linewidth=2, alpha=0.8, label='No change')
        axes[1, 1].set_title('Spectral Norm Ratios by Component')
        axes[1, 1].set_ylabel('Spectral Norm Ratio (Model2/Model1)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Spectral Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / "spectral_analysis" / 'spectral_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🌈 Saved spectral analysis summary")

    def create_manifold_analysis(self, weights1: dict, weights2: dict):
        """Create manifold analysis using PCA and t-SNE"""
        print("🗺️ Creating manifold analysis...")
        
        # Collect weight vectors for manifold analysis
        weight_vectors_1 = []
        weight_vectors_2 = []
        labels = []
        
        for weight_name in set(weights1.keys()) & set(weights2.keys()):
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 30 and 
                param_type == 'weight' and weights1[weight_name].numel() > 100):
                
                w1 = weights1[weight_name].flatten().float().numpy()
                w2 = weights2[weight_name].flatten().float().numpy()
                
                # Sample or use PCA for dimensionality reduction if too large
                if len(w1) > 1000:
                    # Use PCA to reduce to manageable size
                    try:
                        pca_temp = PCA(n_components=min(100, len(w1)))
                        w1_reduced = pca_temp.fit_transform(w1.reshape(1, -1)).flatten()
                        w2_reduced = pca_temp.transform(w2.reshape(1, -1)).flatten()
                        weight_vectors_1.append(w1_reduced)
                        weight_vectors_2.append(w2_reduced)
                    except:
                        # Fallback to sampling
                        idx = np.random.choice(len(w1), min(100, len(w1)), replace=False)
                        weight_vectors_1.append(w1[idx])
                        weight_vectors_2.append(w2[idx])
                else:
                    weight_vectors_1.append(w1)
                    weight_vectors_2.append(w2)
                
                labels.append(f"L{layer_num}_{component}")
        
        if len(weight_vectors_1) < 3:
            print("   ⚠️ Not enough weight matrices for manifold analysis")
            return
        
        # Pad vectors to same length
        max_len = max(len(v) for v in weight_vectors_1 + weight_vectors_2)
        
        def pad_vector(v, target_len):
            if len(v) < target_len:
                return np.pad(v, (0, target_len - len(v)), 'constant')
            else:
                return v[:target_len]
        
        weight_vectors_1 = [pad_vector(v, max_len) for v in weight_vectors_1]
        weight_vectors_2 = [pad_vector(v, max_len) for v in weight_vectors_2]
        
        # Combine data
        all_vectors = np.array(weight_vectors_1 + weight_vectors_2)
        all_labels = labels + [f"{l}_M2" for l in labels]
        model_labels = ['Model 1'] * len(weight_vectors_1) + ['Model 2'] * len(weight_vectors_2)
        
        self._create_pca_analysis(all_vectors, all_labels, model_labels)
        self._create_tsne_analysis(all_vectors, all_labels, model_labels)

    def _create_pca_analysis(self, vectors: np.ndarray, labels: List[str], model_labels: List[str]):
        """Create PCA analysis"""
        
        try:
            pca = PCA(n_components=min(10, vectors.shape[0], vectors.shape[1]))
            vectors_pca = pca.fit_transform(vectors)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. PCA scatter plot (first 2 components)
            model1_mask = np.array(model_labels) == 'Model 1'
            model2_mask = np.array(model_labels) == 'Model 2'
            
            axes[0, 0].scatter(vectors_pca[model1_mask, 0], vectors_pca[model1_mask, 1], 
                              alpha=0.8, label='Model 1', color=COLORS['model1'], s=50, edgecolors='white', linewidth=0.5)
            axes[0, 0].scatter(vectors_pca[model2_mask, 0], vectors_pca[model2_mask, 1], 
                              alpha=0.8, label='Model 2', color=COLORS['model2'], s=50, edgecolors='white', linewidth=0.5)
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
            axes[0, 0].set_title('PCA: Weight Matrix Embeddings')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Explained variance ratio with gradient colors
            bars = axes[0, 1].bar(range(len(pca.explained_variance_ratio_)), 
                                 pca.explained_variance_ratio_, alpha=0.8, 
                                 color=plt.cm.viridis(np.linspace(0, 1, len(pca.explained_variance_ratio_))),
                                 edgecolor='white', linewidth=0.5)
            axes[0, 1].set_xlabel('Principal Component')
            axes[0, 1].set_ylabel('Explained Variance Ratio')
            axes[0, 1].set_title('PCA Explained Variance')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Cumulative explained variance with vibrant colors
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            axes[1, 0].plot(range(len(cumvar)), cumvar, color=COLORS['model1'], marker='o', 
                           linewidth=3, markersize=6, markerfacecolor='white', markeredgewidth=2)
            axes[1, 0].axhline(y=0.95, color=COLORS['reference'], linestyle='--', linewidth=2, alpha=0.8, label='95% variance')
            axes[1, 0].set_xlabel('Number of Components')
            axes[1, 0].set_ylabel('Cumulative Explained Variance')
            axes[1, 0].set_title('Cumulative Explained Variance')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Component loadings heatmap (first few components)
            n_components_show = min(5, pca.components_.shape[0])
            n_features_show = min(50, pca.components_.shape[1])
            
            im = axes[1, 1].imshow(pca.components_[:n_components_show, :n_features_show], 
                                  aspect='auto', cmap='RdBu_r')
            axes[1, 1].set_xlabel('Feature Index')
            axes[1, 1].set_ylabel('Principal Component')
            axes[1, 1].set_title('PCA Component Loadings')
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.suptitle('PCA Analysis of Weight Matrices', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.output_dir / "manifold_analysis" / 'pca_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   🗺️ Saved PCA analysis")
            
        except Exception as e:
            print(f"   ⚠️ PCA analysis failed: {str(e)}")

    def _create_tsne_analysis(self, vectors: np.ndarray, labels: List[str], model_labels: List[str]):
        """Create t-SNE analysis"""
        
        try:
            # Use PCA first to reduce dimensionality for t-SNE
            if vectors.shape[1] > 50:
                pca = PCA(n_components=50)
                vectors_reduced = pca.fit_transform(vectors)
            else:
                vectors_reduced = vectors
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
            vectors_tsne = tsne.fit_transform(vectors_reduced)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 1. t-SNE by model
            model1_mask = np.array(model_labels) == 'Model 1'
            model2_mask = np.array(model_labels) == 'Model 2'
            
            axes[0].scatter(vectors_tsne[model1_mask, 0], vectors_tsne[model1_mask, 1], 
                           alpha=0.8, label='Model 1', color=COLORS['model1'], s=60, edgecolors='white', linewidth=0.5)
            axes[0].scatter(vectors_tsne[model2_mask, 0], vectors_tsne[model2_mask, 1], 
                           alpha=0.8, label='Model 2', color=COLORS['model2'], s=60, edgecolors='white', linewidth=0.5)
            axes[0].set_xlabel('t-SNE Component 1')
            axes[0].set_ylabel('t-SNE Component 2')
            axes[0].set_title('t-SNE: Weight Matrix Embeddings by Model')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. t-SNE by component type with distinct colors
            component_types = list(set(l.split('_')[1] for l in labels))
            comp_colors = [COLORS['model1'], COLORS['model2'], COLORS['diff'], COLORS['kde1'], COLORS['kde2']]
            
            for i, comp_type in enumerate(component_types):
                mask = np.array([comp_type in l for l in labels])
                if np.any(mask):
                    color = comp_colors[i % len(comp_colors)]
                    axes[1].scatter(vectors_tsne[mask, 0], vectors_tsne[mask, 1], 
                                   alpha=0.8, label=comp_type, s=60, color=color, 
                                   edgecolors='white', linewidth=0.5)
            
            axes[1].set_xlabel('t-SNE Component 1')
            axes[1].set_ylabel('t-SNE Component 2')
            axes[1].set_title('t-SNE: Weight Matrix Embeddings by Component')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle('t-SNE Analysis of Weight Matrices', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.output_dir / "manifold_analysis" / 'tsne_analysis.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   🗺️ Saved t-SNE analysis")
            
        except Exception as e:
            print(f"   ⚠️ t-SNE analysis failed: {str(e)}")

    def create_interactive_plots(self, weights1: dict, weights2: dict):
        """Create interactive plots using Plotly"""
        print("🎮 Creating interactive plots...")
        
        # Collect data for interactive plots
        plot_data = []
        
        for weight_name in set(weights1.keys()) & set(weights2.keys()):
            layer_num, component, param_type = self.parse_layer_info(weight_name)
            if (layer_num is not None and 0 <= layer_num <= 30 and 
                param_type == 'weight' and weights1[weight_name].numel() > 100):
                
                w1 = weights1[weight_name].flatten().float().numpy()
                w2 = weights2[weight_name].flatten().float().numpy()
                
                # Sample for interactive plotting
                if len(w1) > 5000:
                    idx = np.random.choice(len(w1), 5000, replace=False)
                    w1 = w1[idx]
                    w2 = w2[idx]
                
                plot_data.append({
                    'layer': layer_num,
                    'component': component,
                    'weight_name': weight_name,
                    'w1': w1,
                    'w2': w2
                })
        
        if plot_data:
            self._create_interactive_scatter(plot_data)
            self._create_interactive_histogram(plot_data)

    def _create_interactive_scatter(self, plot_data: List[Dict]):
        """Create interactive scatter plot"""
        
        try:
            fig = make_subplots(rows=1, cols=1, 
                               subplot_titles=['Interactive Weight Comparison'])
            
            for i, data in enumerate(plot_data[:10]):  # Limit to first 10 for performance
                fig.add_trace(
                    go.Scatter(
                        x=data['w1'],
                        y=data['w2'],
                        mode='markers',
                        name=f"L{data['layer']}_{data['component']}",
                        marker=dict(size=2, opacity=0.6),
                        hovertemplate=f"<b>Layer {data['layer']} {data['component']}</b><br>" +
                                    "Model 1: %{x:.6f}<br>" +
                                    "Model 2: %{y:.6f}<extra></extra>"
                    )
                )
            
            # Add diagonal line
            all_w1 = np.concatenate([d['w1'] for d in plot_data[:10]])
            all_w2 = np.concatenate([d['w2'] for d in plot_data[:10]])
            min_val, max_val = min(all_w1.min(), all_w2.min()), max(all_w1.max(), all_w2.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Correlation',
                    line=dict(dash='dash', color='red')
                )
            )
            
            fig.update_layout(
                title='Interactive Weight Value Comparison',
                xaxis_title='Model 1 Weights',
                yaxis_title='Model 2 Weights',
                height=600,
                showlegend=True
            )
            
            output_path = self.output_dir / "interactive_plots" / 'interactive_scatter.html'
            fig.write_html(str(output_path))
            
            print(f"   🎮 Saved interactive scatter plot")
            
        except Exception as e:
            print(f"   ⚠️ Interactive scatter plot failed: {str(e)}")

    def _create_interactive_histogram(self, plot_data: List[Dict]):
        """Create interactive histogram comparison"""
        
        try:
            fig = make_subplots(rows=2, cols=2, 
                               subplot_titles=['Model 1 Distributions', 'Model 2 Distributions',
                                             'Difference Distributions', 'Component Comparison'])
            
            # Sample a few components for visualization
            sample_data = plot_data[:6]
            
            for data in sample_data:
                # Model 1 histogram
                fig.add_trace(
                    go.Histogram(
                        x=data['w1'],
                        name=f"L{data['layer']}_{data['component']}_M1",
                        opacity=0.7,
                        nbinsx=50
                    ),
                    row=1, col=1
                )
                
                # Model 2 histogram
                fig.add_trace(
                    go.Histogram(
                        x=data['w2'],
                        name=f"L{data['layer']}_{data['component']}_M2",
                        opacity=0.7,
                        nbinsx=50
                    ),
                    row=1, col=2
                )
                
                # Difference histogram
                diff = data['w2'] - data['w1']
                fig.add_trace(
                    go.Histogram(
                        x=diff,
                        name=f"L{data['layer']}_{data['component']}_diff",
                        opacity=0.7,
                        nbinsx=50
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title='Interactive Weight Distribution Analysis',
                height=800,
                showlegend=True
            )
            
            output_path = self.output_dir / "interactive_plots" / 'interactive_histograms.html'
            fig.write_html(str(output_path))
            
            print(f"   🎮 Saved interactive histogram plot")
            
        except Exception as e:
            print(f"   ⚠️ Interactive histogram plot failed: {str(e)}")

    def run_analysis(self):
        """Run the complete advanced analysis"""
        print("\n🚀 Starting advanced weight analysis...")
        
        # Load weights
        print("📥 Loading model weights...")
        weights1 = self.load_weights(self.model1_path, self.weight_map1)
        weights2 = self.load_weights(self.model2_path, self.weight_map2)
        
        # Run different analysis methods
        self.create_histogram_kde_analysis(weights1, weights2)
        self.create_scatter_hexbin_analysis(weights1, weights2)
        self.create_spectral_analysis(weights1, weights2)
        self.create_manifold_analysis(weights1, weights2)
        self.create_interactive_plots(weights1, weights2)
        
        print("\n✅ Advanced analysis complete!")
        
        # Count generated files
        hist_plots = len(list((self.output_dir / 'histograms_kde').glob('*.png')))
        scatter_plots = len(list((self.output_dir / 'scatter_hexbin').glob('*.png')))
        spectral_plots = len(list((self.output_dir / 'spectral_analysis').glob('*.png')))
        manifold_plots = len(list((self.output_dir / 'manifold_analysis').glob('*.png')))
        interactive_plots = len(list((self.output_dir / 'interactive_plots').glob('*.html')))
        
        print(f"📊 Generated:")
        print(f"   - {hist_plots} histogram & KDE plots")
        print(f"   - {scatter_plots} scatter & hexbin plots")
        print(f"   - {spectral_plots} spectral analysis plots")
        print(f"   - {manifold_plots} manifold analysis plots")
        print(f"   - {interactive_plots} interactive HTML plots")

def main():
    parser = argparse.ArgumentParser(description='Advanced weight visualization analysis')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--output', type=str, default='advanced_weight_analysis', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdvancedWeightVisualizer(args.model1, args.model2, args.output)
    
    # Run analysis
    analyzer.run_analysis()
    
    print(f"\n🎯 Analysis complete! Check {args.output}/ for results")

if __name__ == "__main__":
    main() 