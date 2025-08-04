#!/usr/bin/env python3
"""
Comprehensive Plotter for Multi-Model Analysis - Optimized for AAAI 2026

Creates publication-quality individual PDF visualizations for weight difference analysis:
- Layer-by-layer weight differences (5 PDFs)
- Special layers analysis (5 PDFs) 
- Comparative boxplots (3 PDFs)
- Interactive plots (1 HTML)

AAAI 2026 Publication Features:
- Academic figure sizing (7x4.5 inches for line plots, 8x5 for boxplots)
- Publication-quality fonts (Times New Roman, 12pt base)
- High DPI output (300 DPI) suitable for print
- Professional color palette that works in both color and grayscale
- Distinct line styles and markers for accessibility
- Optimized legend positioning to prevent content overlap
- Subtle grid lines and professional formatting
- White backgrounds with black borders

Model Name Labels:
- sft_ct → "SFT for Code Sum"
- sft_4o → "SFT for Code Gen" 
- dare → "DARE"
- ties → "TIES"
- linear → "LINEAR"
- della → "DELLA"
- Data mixture indicators: "1e5/5e6 Data Mixture" (no parentheses)

Technical Features:
- Mean values only (no standard deviation error bars for clarity)
- Individual PDF files for each subfigure
- Consistent color scheme across all plots
- Professional typography and spacing

Usage:
    python comprehensive_plotter.py --data_file layer_summary.csv --output_dir plots/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


plt.style.use('default')
sns.set_style("whitegrid")


plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
})

# Define color palette optimized for publication (works in grayscale)
COLORS = [
    '#1f77b4',  # Blue
    '#d62728',  # Red  
    '#2ca02c',  # Green
    '#ff7f0e',  # Orange
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]

# Define line styles for better grayscale compatibility
LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

class ComprehensivePlotter:
    def __init__(self, output_dir: str = "comprehensive_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "layer_analysis").mkdir(exist_ok=True)
        (self.output_dir / "distribution_analysis").mkdir(exist_ok=True)
        (self.output_dir / "comprehensive_plots").mkdir(exist_ok=True)
        (self.output_dir / "interactive_plots").mkdir(exist_ok=True)
        (self.output_dir / "individual_pdfs").mkdir(exist_ok=True)
        
        print(f"🎨 Comprehensive Plotter initialized")
        print(f"   Output directory: {self.output_dir}")

    def clean_model_name(self, model_name: str) -> str:
        """Clean and format model names with proper labels"""
        name = model_name.lower()
        
        # Handle model types
        if 'sft_ct' in name:
            base = "SFT for Code Sum"
        elif 'sft_4o' in name:
            base = "SFT for Code Gen"  
        elif 'dare' in name:
            base = "DARE"
        elif 'ties' in name:
            base = "TIES"
        elif 'linear' in name:
            base = "LINEAR"
        elif 'della' in name:
            base = "DELLA"
        elif '1e5' == name or '5e6' == name:
            base = "Data Mixture"

        else:
            base = model_name
            
        return base

    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load layer summary data from CSV file"""
        print(f"Loading data from {data_file}...")
        
        df = pd.read_csv(data_file)
        
        # Check required columns
        required_cols = ['layer_num', 'model_name', 'l2_norm_mean', 'relative_diff_mean', 
                        'cosine_similarity_mean', 'mean_abs_diff_mean']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print(f"Loaded {len(df)} records")
        print(f"Found {df['model_name'].nunique()} unique models")
        print(f"Layer range: {df['layer_num'].min()} to {df['layer_num'].max()}")
        
        return df

    def create_layer_by_layer_plot(self, layer_summary: pd.DataFrame):
        """Create individual PDF files for each layer-by-layer analysis plot"""
        print(" Creating individual layer-by-layer analysis PDFs...")
        
        # Filter to only regular transformer layers (0-50), exclude special layers
        regular_layers_data = layer_summary[
            (layer_summary['layer_num'] >= 0) & 
            (layer_summary['layer_num'] <= 50)
        ].copy()
        
        if len(regular_layers_data) == 0:
            print("No regular transformer layers found for plotting")
            return

        # Add model_idx if not present and clean model names
        if 'model_idx' not in regular_layers_data.columns:
            model_names = regular_layers_data['model_name'].unique()
            model_map = {name: idx for idx, name in enumerate(model_names)}
            regular_layers_data['model_idx'] = regular_layers_data['model_name'].map(model_map)
        
        # Clean model names
        regular_layers_data['clean_model_name'] = regular_layers_data['model_name'].apply(self.clean_model_name)
        
        # Get unique models and layers
        models = regular_layers_data[['model_idx', 'model_name', 'clean_model_name']].drop_duplicates().sort_values('model_idx')
        layers = sorted(regular_layers_data['layer_num'].unique())
        
        print(f"   📊 Plotting {len(layers)} layers: {min(layers)} to {max(layers)}")
        
        # 1. L2 Norm by Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))  # Slightly larger to accommodate legend
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            
            ax.plot(model_data['layer_num'], model_data['l2_norm_mean'], 
                    marker=MARKERS[i % len(MARKERS)], 
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    linewidth=2, markersize=5, markeredgewidth=1, markeredgecolor='white',
                    label=model['clean_model_name'],
                    color=COLORS[i % len(COLORS)])
        

        ax.set_xlabel('Layer Number')
        ax.set_ylabel('L2 Norm')
        
        # Position legend outside plot area to avoid overlap
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                 fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(left=min(layers)-0.5, right=max(layers)+0.5)
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'l2_norm_by_layer.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: l2_norm_by_layer.pdf")
        
        # 2. Relative Difference by Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax.plot(model_data['layer_num'], model_data['relative_diff_mean'], 
                    marker=MARKERS[i % len(MARKERS)], 
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    linewidth=2, markersize=5, markeredgewidth=1, markeredgecolor='white',
                    label=model['clean_model_name'],
                    color=COLORS[i % len(COLORS)])
        

        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Relative Difference')
        
        # Position legend outside plot area to avoid overlap
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                 fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(left=min(layers)-0.5, right=max(layers)+0.5)
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'relative_diff_by_layer.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: relative_diff_by_layer.pdf")
        
        # 3. Cosine Similarity by Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax.plot(model_data['layer_num'], model_data['cosine_similarity_mean'], 
                    marker=MARKERS[i % len(MARKERS)], 
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    linewidth=2, markersize=5, markeredgewidth=1, markeredgecolor='white',
                    label=model['clean_model_name'],
                    color=COLORS[i % len(COLORS)])
        

        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Cosine Similarity')
        
        # Position legend outside plot area to avoid overlap
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                 fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(left=min(layers)-0.5, right=max(layers)+0.5)
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'cosine_similarity_by_layer.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f" Saved: cosine_similarity_by_layer.pdf")
        
        # 4. Mean Absolute Difference by Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = regular_layers_data[regular_layers_data['model_idx'] == model['model_idx']]
            ax.plot(model_data['layer_num'], model_data['mean_abs_diff_mean'], 
                    marker=MARKERS[i % len(MARKERS)], 
                    linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    linewidth=2, markersize=5, markeredgewidth=1, markeredgecolor='white',
                    label=model['clean_model_name'],
                    color=COLORS[i % len(COLORS)])
        

        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Mean Absolute Difference')
        
        # Position legend outside plot area to avoid overlap
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                 fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xlim(left=min(layers)-0.5, right=max(layers)+0.5)
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'mean_abs_diff_by_layer.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f" Saved: mean_abs_diff_by_layer.pdf")
        
        # 5. Heatmap of L2 norms
        fig, ax = plt.subplots(figsize=(8, 4))  # Double column width for heatmap
        
        # Create pivot table for heatmap (regular layers only) with clean names
        try:
            pivot_data = regular_layers_data.pivot(index='clean_model_name', columns='layer_num', values='l2_norm_mean')
            
            # Use a publication-friendly colormap
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='Blues', interpolation='nearest')
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index)
            
            # Set layer number ticks at reasonable intervals
            layer_cols = pivot_data.columns
            if len(layer_cols) > 10:
                tick_step = max(1, len(layer_cols) // 10)
                tick_positions = range(0, len(layer_cols), tick_step)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([layer_cols[i] for i in tick_positions])
            else:
                ax.set_xticks(range(len(layer_cols)))
                ax.set_xticklabels(layer_cols)
                
            ax.set_xlabel('Layer Number')
            ax.set_ylabel('Model')
            
            # Add colorbar with better formatting
            cbar = plt.colorbar(im, ax=ax, label='L2 Norm')
            cbar.ax.tick_params(labelsize=10)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Heatmap unavailable:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'l2_norm_heatmap.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"   📊 Saved: l2_norm_heatmap.pdf")

    def create_special_layers_plot(self, layer_summary: pd.DataFrame):
        """Create individual PDF files for special layers analysis (embedding, final norm, LM head)"""
        print("Creating individual special layers analysis PDFs...")
        
        # Filter to only special layers
        special_layers_data = layer_summary[
            (layer_summary['layer_num'] == -1) |   # Embedding
            (layer_summary['layer_num'] == 998) |  # Final norm
            (layer_summary['layer_num'] == 999)    # LM head
        ].copy()
        
        if len(special_layers_data) == 0:
            print("   ⚠️ No special layers found for plotting")
            return

        # Add model_idx if not present and clean model names
        if 'model_idx' not in special_layers_data.columns:
            model_names = special_layers_data['model_name'].unique()
            model_map = {name: idx for idx, name in enumerate(model_names)}
            special_layers_data['model_idx'] = special_layers_data['model_name'].map(model_map)
        
        # Clean model names
        special_layers_data['clean_model_name'] = special_layers_data['model_name'].apply(self.clean_model_name)
        
        # Create layer name mapping for better readability
        layer_names = {-1: 'Embedding', 998: 'Final Norm', 999: 'LM Head'}
        special_layers_data['layer_name'] = special_layers_data['layer_num'].map(layer_names)
        
        # Get unique models
        models = special_layers_data[['model_idx', 'model_name', 'clean_model_name']].drop_duplicates().sort_values('model_idx')
        special_layers = sorted(special_layers_data['layer_num'].unique())
        
        print(f"   🎯 Plotting special layers: {[layer_names.get(l, f'Layer {l}') for l in special_layers]}")
        
        # 1. L2 Norm by Special Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x_pos = range(len(special_layers))
        width = 0.8 / len(models)
        
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = special_layers_data[special_layers_data['model_idx'] == model['model_idx']]
            values = []
            for layer_num in special_layers:
                layer_data = model_data[model_data['layer_num'] == layer_num]
                if len(layer_data) > 0:
                    values.append(layer_data['l2_norm_mean'].iloc[0])
                else:
                    values.append(0)
            
            ax.bar([x + i * width for x in x_pos], values, width, 
                   label=model['clean_model_name'],
                   color=COLORS[i % len(COLORS)], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
        

        ax.set_xlabel('Layer Type')
        ax.set_ylabel('L2 Norm')
        ax.set_xticks([x + width * (len(models) - 1) / 2 for x in x_pos])
        ax.set_xticklabels([layer_names.get(l, f'Layer {l}') for l in special_layers])
        
        # Position legend at bottom to avoid overlap with bars
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=min(3, len(models)), 
                 frameon=True, fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'special_layers_l2_norm.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: special_layers_l2_norm.pdf")
        
        # 2. Relative Difference by Special Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = special_layers_data[special_layers_data['model_idx'] == model['model_idx']]
            values = []
            for layer_num in special_layers:
                layer_data = model_data[model_data['layer_num'] == layer_num]
                if len(layer_data) > 0:
                    values.append(layer_data['relative_diff_mean'].iloc[0])
                else:
                    values.append(0)
            
            ax.bar([x + i * width for x in x_pos], values, width,
                   label=model['clean_model_name'],
                   color=COLORS[i % len(COLORS)], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
        

        ax.set_xlabel('Layer Type')
        ax.set_ylabel('Relative Difference')
        ax.set_xticks([x + width * (len(models) - 1) / 2 for x in x_pos])
        ax.set_xticklabels([layer_names.get(l, f'Layer {l}') for l in special_layers])
        
        # Position legend at bottom to avoid overlap with bars
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=min(3, len(models)), 
                 frameon=True, fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'special_layers_relative_diff.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: special_layers_relative_diff.pdf")
        
        # 3. Cosine Similarity by Special Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = special_layers_data[special_layers_data['model_idx'] == model['model_idx']]
            values = []
            for layer_num in special_layers:
                layer_data = model_data[model_data['layer_num'] == layer_num]
                if len(layer_data) > 0:
                    values.append(layer_data['cosine_similarity_mean'].iloc[0])
                else:
                    values.append(1.0)  # Default cosine similarity
            
            ax.bar([x + i * width for x in x_pos], values, width,
                   label=model['clean_model_name'],
                   color=COLORS[i % len(COLORS)], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
        

        ax.set_xlabel('Layer Type')
        ax.set_ylabel('Cosine Similarity')
        ax.set_xticks([x + width * (len(models) - 1) / 2 for x in x_pos])
        ax.set_xticklabels([layer_names.get(l, f'Layer {l}') for l in special_layers])
        
        # Position legend at bottom to avoid overlap with bars
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=min(3, len(models)), 
                 frameon=True, fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'special_layers_cosine_sim.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: special_layers_cosine_sim.pdf")
        
        # 4. Mean Absolute Difference by Special Layer
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (_, model) in enumerate(models.iterrows()):
            model_data = special_layers_data[special_layers_data['model_idx'] == model['model_idx']]
            values = []
            for layer_num in special_layers:
                layer_data = model_data[model_data['layer_num'] == layer_num]
                if len(layer_data) > 0:
                    values.append(layer_data['mean_abs_diff_mean'].iloc[0])
                else:
                    values.append(0)
            
            ax.bar([x + i * width for x in x_pos], values, width,
                   label=model['clean_model_name'],
                   color=COLORS[i % len(COLORS)], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
        

        ax.set_xlabel('Layer Type')
        ax.set_ylabel('Mean Absolute Difference')
        ax.set_xticks([x + width * (len(models) - 1) / 2 for x in x_pos])
        ax.set_xticklabels([layer_names.get(l, f'Layer {l}') for l in special_layers])
        
        # Position legend at bottom to avoid overlap with bars
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=min(3, len(models)), 
                 frameon=True, fancybox=False, shadow=False, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'special_layers_mean_abs_diff.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"   🎯 Saved: special_layers_mean_abs_diff.pdf")
        
        # 5. Special layers heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Create pivot for heatmap with clean names
        heatmap_data = []
        for _, model in models.iterrows():
            row_data = []
            for layer_num in special_layers:
                model_data = special_layers_data[
                    (special_layers_data['model_idx'] == model['model_idx']) &
                    (special_layers_data['layer_num'] == layer_num)
                ]
                if len(model_data) > 0:
                    row_data.append(model_data['l2_norm_mean'].iloc[0])
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        if heatmap_data:
            # Use publication-friendly colormap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='Blues', interpolation='nearest')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([model['clean_model_name'] for _, model in models.iterrows()])
            ax.set_xticks(range(len(special_layers)))
            ax.set_xticklabels([layer_names.get(l, f'Layer {l}') for l in special_layers])

            
            # Add text annotations with better contrast
            for i in range(len(models)):
                for j in range(len(special_layers)):
                    value = heatmap_data[i][j]
                    # Choose text color based on cell intensity
                    text_color = 'white' if value > (max(max(row) for row in heatmap_data) * 0.5) else 'black'
                    text = ax.text(j, i, f'{value:.2f}',
                                   ha="center", va="center", color=text_color, fontsize=10)
            
            cbar = plt.colorbar(im, ax=ax, label='L2 Norm')
            cbar.ax.tick_params(labelsize=10)
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'special_layers_heatmap.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"   🎯 Saved: special_layers_heatmap.pdf")

    def create_comparative_boxplot(self, layer_summary: pd.DataFrame):
        """Create individual PDF files for comparative boxplots"""
        print("📦 Creating individual comparative boxplot PDFs...")
        
        # Filter to only regular transformer layers (0-50)
        regular_layers_data = layer_summary[
            (layer_summary['layer_num'] >= 0) & 
            (layer_summary['layer_num'] <= 50)
        ].copy()
        
        if len(regular_layers_data) == 0:
            print("   ⚠️ No regular transformer layers found for boxplot")
            return

        # Add model_idx if not present and clean model names
        if 'model_idx' not in regular_layers_data.columns:
            model_names = regular_layers_data['model_name'].unique()
            model_map = {name: idx for idx, name in enumerate(model_names)}
            regular_layers_data['model_idx'] = regular_layers_data['model_name'].map(model_map)
        
        # Clean model names
        regular_layers_data['clean_model_name'] = regular_layers_data['model_name'].apply(self.clean_model_name)
        
        # 1. L2 Norm boxplot
        fig, ax = plt.subplots(figsize=(8, 5))  # Wider for better label spacing
        clean_models = []
        l2_data = []
        for model_name in regular_layers_data['model_name'].unique():
            model_data = regular_layers_data[regular_layers_data['model_name'] == model_name]
            clean_models.append(model_data['clean_model_name'].iloc[0])
            l2_data.append(model_data['l2_norm_mean'].values)
        
        bp1 = ax.boxplot(l2_data, labels=clean_models, patch_artist=True)
        for i, patch in enumerate(bp1['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Style whiskers, caps, and medians
        for whisker in bp1['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
        for cap in bp1['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)
        for median in bp1['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        

        ax.set_ylabel('L2 Norm')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'l2_norm_boxplot.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"   📦 Saved: l2_norm_boxplot.pdf")
        
        # 2. Relative Difference boxplot
        fig, ax = plt.subplots(figsize=(8, 5))  # Wider for better label spacing
        rel_data = []
        for model_name in regular_layers_data['model_name'].unique():
            model_data = regular_layers_data[regular_layers_data['model_name'] == model_name]
            rel_data.append(model_data['relative_diff_mean'].values)
        
        bp2 = ax.boxplot(rel_data, labels=clean_models, patch_artist=True)
        for i, patch in enumerate(bp2['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Style whiskers, caps, and medians
        for whisker in bp2['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
        for cap in bp2['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)
        for median in bp2['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        

        ax.set_ylabel('Relative Difference')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'relative_diff_boxplot.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"   📦 Saved: relative_diff_boxplot.pdf")
        
        # 3. Cosine Similarity boxplot
        fig, ax = plt.subplots(figsize=(8, 5))  # Wider for better label spacing
        cos_data = []
        for model_name in regular_layers_data['model_name'].unique():
            model_data = regular_layers_data[regular_layers_data['model_name'] == model_name]
            cos_data.append(model_data['cosine_similarity_mean'].values)
        
        bp3 = ax.boxplot(cos_data, labels=clean_models, patch_artist=True)
        for i, patch in enumerate(bp3['boxes']):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        
        # Style whiskers, caps, and medians
        for whisker in bp3['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
        for cap in bp3['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)
        for median in bp3['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        

        ax.set_ylabel('Cosine Similarity')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout(pad=0.5)
        output_path = self.output_dir / "individual_pdfs" / 'cosine_sim_boxplot.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"   📦 Saved: cosine_sim_boxplot.pdf")
        


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

            # Add model_idx if not present
            if 'model_idx' not in regular_layers_data.columns:
                model_names = regular_layers_data['model_name'].unique()
                model_map = {name: idx for idx, name in enumerate(model_names)}
                regular_layers_data['model_idx'] = regular_layers_data['model_name'].map(model_map)
                
            fig = make_subplots(
                rows=2, cols=2,
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
                
                # Relative Difference
                fig.add_trace(
                    go.Scatter(x=model_data['layer_num'], y=model_data['relative_diff_mean'],
                              mode='lines+markers', name=f"{model['model_name']} - RelDiff",
                              line=dict(color=color), showlegend=False),
                    row=1, col=2
                )
                
                # Cosine Similarity
                fig.add_trace(
                    go.Scatter(x=model_data['layer_num'], y=model_data['cosine_similarity_mean'],
                              mode='lines+markers', name=f"{model['model_name']} - CosSim",
                              line=dict(color=color), showlegend=False),
                    row=2, col=1
                )
                
                # Mean Absolute Difference
                fig.add_trace(
                    go.Scatter(x=model_data['layer_num'], y=model_data['mean_abs_diff_mean'],
                              mode='lines+markers', name=f"{model['model_name']} - MAD",
                              line=dict(color=color), showlegend=False),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                showlegend=True
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Layer Number", row=1, col=1)
            fig.update_xaxes(title_text="Layer Number", row=1, col=2)
            fig.update_xaxes(title_text="Layer Number", row=2, col=1)
            fig.update_xaxes(title_text="Layer Number", row=2, col=2)
            
            fig.update_yaxes(title_text="L2 Norm", row=1, col=1)
            fig.update_yaxes(title_text="Relative Difference", row=1, col=2)
            fig.update_yaxes(title_text="Cosine Similarity", row=2, col=1)
            fig.update_yaxes(title_text="Mean Abs Difference", row=2, col=2)
            
            output_path = self.output_dir / "interactive_plots" / 'interactive_analysis.html'
            fig.write_html(str(output_path))
            
            print(f"   🎮 Saved interactive plot")
            
        except Exception as e:
            print(f"   ⚠️ Interactive plot failed: {str(e)}")

    def create_all_plots(self, layer_summary: pd.DataFrame):
        """Create all individual PDF plots"""
        print("\n🎨 Creating all individual PDF plots...")
        
        self.create_layer_by_layer_plot(layer_summary)
        self.create_special_layers_plot(layer_summary)
        self.create_comparative_boxplot(layer_summary)
        self.create_interactive_plot(layer_summary)
        
        print("\n✅ All individual PDF plots created!")
        
        # Count generated files
        pdf_plots = len(list((self.output_dir / 'individual_pdfs').glob('*.pdf')))
        interactive_plots = len(list((self.output_dir / 'interactive_plots').glob('*.html')))
        
        print(f" Generated:")
        print(f"   - {pdf_plots} individual PDF plots")
        print(f"   - {interactive_plots} interactive plots")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive plotter for multi-model analysis')
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Path to layer summary CSV file')
    parser.add_argument('--output_dir', type=str, default='comprehensive_plots', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"Data file not found: {args.data_file}")
        return
    
    # Initialize plotter
    plotter = ComprehensivePlotter(args.output_dir)
    
    # Load data
    layer_summary = plotter.load_data(args.data_file)
    if layer_summary is None:
        print("Failed to load data")
        return
    
    # Create all plots
    plotter.create_all_plots(layer_summary)
    
    print(f"\n Plotting complete! Check {args.output_dir}/ for results")

if __name__ == "__main__":
    main() 