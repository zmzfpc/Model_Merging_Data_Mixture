#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_layer_label(layer_num):
    """Get human-readable label for layer number"""
    if layer_num == -1:
        return 'emb'
    elif layer_num == 998:
        return 'final_norm'
    elif layer_num == 999:
        return 'lm_head'
    else:
        return f'layer_{layer_num}'

def get_dataset_color(dataset):
    """Get AAAI publication-quality colors for dataset families"""
    # ColorBrewer-inspired palette with similar colors within families
    # Colors work well in grayscale and are colorblind-friendly
    color_map = {
        # Qwen Family (Blue variants) - Both use blue tones
        'qwc7': '#2166ac',    # Dark blue for Qwen Coder 7B  
        'qwc15': '#4393c3',   # Medium blue for Qwen Coder 1.5B
        
        # Deepseek Family (Red variants) - Both use red tones  
        'dsc7': '#d73027',    # Dark red for Deepseek Coder 7B
        'dsc13': '#f46d43'    # Light red for Deepseek Coder 1.3B
    }
    return color_map.get(dataset, '#636363')  # Professional gray if not found

def get_dataset_colors(datasets):
    """Get list of colors for multiple datasets"""
    return [get_dataset_color(dataset) for dataset in datasets]

def get_dataset_style(dataset):
    """Get line style and marker for AAAI publication quality"""
    # Different line styles and markers for grayscale compatibility
    style_map = {
        'qwc7': {'linestyle': '-', 'marker': 'o'},      # Solid line, circle
        'qwc15': {'linestyle': '--', 'marker': 's'},    # Dashed line, square  
        'dsc7': {'linestyle': '-.', 'marker': '^'},     # Dash-dot line, triangle
        'dsc13': {'linestyle': ':', 'marker': 'D'}      # Dotted line, diamond
    }
    return style_map.get(dataset, {'linestyle': '-', 'marker': 'o'})

def setup_publication_style():
    """Set up matplotlib for AAAI publication quality"""
    plt.rcParams.update({
        'font.size': 18,           # Increased to 18 for maximum readability
        'axes.titlesize': 20,      # Increased to 20
        'axes.labelsize': 18,      # Increased to 18
        'xtick.labelsize': 16,     # Increased to 16
        'ytick.labelsize': 16,     # Increased to 16
        'legend.fontsize': 16,     # Increased to 16
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.linewidth': 0.8,
        'grid.alpha': 0.3,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'legend.fancybox': False
    })

def plot_regular_layers():
    """Plot correlations for regular transformer layers (0-50)"""
    
    # Read the combined correlation data
    df = pd.read_csv('combined_ct_vs_4o_correlations_0.3.csv')
    
    # Filter to regular layers only (0-50)
    regular_df = df[(df['layer_num'] >= 0) & (df['layer_num'] <= 50)].copy()
    
    if len(regular_df) == 0:
        print("No regular layers found in data")
        return
    
    # Create mapping for better labels
    dataset_labels = {
        'qwc7': 'Qwen Coder 7B',
        'qwc15': 'Qwen Coder 1.5B',
        'dsc7': 'Deepseek Coder 7B', 
        'dsc13': 'Deepseek Coder 1.3B'
    }
    
    # Add readable dataset labels
    regular_df['dataset_label'] = regular_df['dataset'].map(dataset_labels)
    
    # Set up AAAI publication style
    setup_publication_style()
    
    # Plot 1: Line plot showing correlations across regular layers
    fig, ax = plt.subplots(figsize=(10, 5))  # AAAI column width friendly
    
    for dataset in sorted(regular_df['dataset'].unique()):
        dataset_data = regular_df[regular_df['dataset'] == dataset]
        color = get_dataset_color(dataset)
        style = get_dataset_style(dataset)
        ax.plot(dataset_data['layer_num'], dataset_data['correlation'], 
                color=color, linewidth=2, markersize=5, 
                linestyle=style['linestyle'], marker=style['marker'],
                markeredgewidth=0.5, markeredgecolor='white',
                label=dataset_labels[dataset])
    
    ax.set_xlabel('Layer Number', fontsize=18)
    ax.set_ylabel('Correlation Coefficient', fontsize=18)

    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(regular_df['layer_num']) + 0.5)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('ct_vs_4o_regular_layers_plot.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('ct_vs_4o_regular_layers_plot.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Regular layers plot saved")

def plot_special_layers():
    """Plot correlations for special layers (embedding, final norm, lm head)"""
    
    # Read the combined correlation data
    df = pd.read_csv('combined_ct_vs_4o_correlations_0.3.csv')
    
    # Filter to special layers only (-1, 998, 999)
    special_df = df[df['layer_num'].isin([-1, 998, 999])].copy()
    
    if len(special_df) == 0:
        print("No special layers found in data")
        return
    
    # Create mapping for better labels
    dataset_labels = {
        'qwc7': 'Qwen Coder 7B',
        'qwc15': 'Qwen Coder 1.5B',
        'dsc7': 'Deepseek Coder 7B', 
        'dsc13': 'Deepseek Coder 1.3B'
    }
    
    # Add readable dataset labels and layer labels
    special_df['dataset_label'] = special_df['dataset'].map(dataset_labels)
    special_df['layer_label'] = special_df['layer_num'].apply(get_layer_label)
    
    # Set up AAAI publication style
    setup_publication_style()
    
    # Get unique datasets and special layers
    datasets = sorted(special_df['dataset'].unique())
    special_layers = sorted(special_df['layer_num'].unique())
    
    # Create bar plot for special layers
    fig, ax = plt.subplots(figsize=(8, 5))  # AAAI column width friendly
    
    x_pos = np.arange(len(special_layers))
    width = 0.7 / len(datasets)
    
    for i, dataset in enumerate(datasets):
        dataset_data = special_df[special_df['dataset'] == dataset]
        values = []
        
        for layer_num in special_layers:
            layer_data = dataset_data[dataset_data['layer_num'] == layer_num]
            if len(layer_data) > 0:
                values.append(layer_data['correlation'].iloc[0])
            else:
                values.append(0)
        
        color = get_dataset_color(dataset)
        ax.bar([x + i * width for x in x_pos], values, width, 
               label=dataset_labels[dataset], color=color, alpha=0.85,
               edgecolor='black', linewidth=0.5)
    
    # Set labels
    ax.set_xlabel('Special Layer Type', fontsize=18)
    ax.set_ylabel('Correlation Coefficient', fontsize=18)

    ax.set_xticks([x + width * (len(datasets) - 1) / 2 for x in x_pos])
    ax.set_xticklabels([get_layer_label(layer) for layer in special_layers])
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('ct_vs_4o_special_layers_plot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ct_vs_4o_special_layers_plot.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Special layers plot saved")

def plot_boxplot_distribution():
    """Create box plot showing distribution of correlations by dataset"""
    
    df = pd.read_csv('combined_ct_vs_4o_correlations_0.3.csv')
    
    # Create mapping for better labels
    dataset_labels = {
        'qwc7': 'Qwen Coder 7B',
        'qwc15': 'Qwen Coder 1.5B',
        'dsc7': 'Deepseek Coder 7B', 
        'dsc13': 'Deepseek Coder 1.3B'
    }
    
    # Set up AAAI publication style
    setup_publication_style()
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(8, 5))  # AAAI column width friendly
    
    box_data = []
    box_labels = []
    datasets = []
    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        box_data.append(dataset_data['correlation'].values)
        box_labels.append(dataset_labels[dataset])
        datasets.append(dataset)
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    widths=0.6)  # Slightly narrower boxes for cleaner look
    
    # Color the boxes with custom colors and professional styling
    for i, patch in enumerate(bp['boxes']):
        color = get_dataset_color(datasets[i])
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Style whiskers, caps, medians, and fliers
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(0.8)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(0.8)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    for flier in bp['fliers']:
        flier.set_markerfacecolor('gray')
        flier.set_markeredgecolor('black')
        flier.set_markersize(4)
    
    ax.set_ylabel('Correlation Coefficient', fontsize=18)

    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Rotate x-axis labels if needed
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    plt.savefig('ct_vs_4o_boxplot_distribution.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('ct_vs_4o_boxplot_distribution.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("✅ Boxplot distribution saved")

def create_combined_heatmap():
    """Create a heatmap visualization of correlations for all layers"""
    
    df = pd.read_csv('combined_ct_vs_4o_correlations_0.3.csv')
    
    # Separate regular and special layers
    regular_df = df[(df['layer_num'] >= 0) & (df['layer_num'] <= 50)].copy()
    special_df = df[df['layer_num'].isin([-1, 998, 999])].copy()
    
    # Add layer labels
    special_df['layer_label'] = special_df['layer_num'].apply(get_layer_label)
    
    # Create dataset labels
    dataset_labels = {
        'qwc7': 'Qwen Coder 7B',
        'qwc15': 'Qwen Coder 1.5B',
        'dsc7': 'Deepseek Coder 7B', 
        'dsc13': 'Deepseek Coder 1.3B'
    }
    
    # Create heatmap for regular layers
    if len(regular_df) > 0:
        heatmap_data_regular = regular_df.pivot(index='layer_num', columns='dataset', values='correlation')
        heatmap_data_regular.columns = [dataset_labels[col] for col in heatmap_data_regular.columns]
        
        fig, ax = plt.subplots(figsize=(10, 12))  # AAAI column width friendly
        
        sns.heatmap(heatmap_data_regular, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='Blues',  # More professional colormap
                    cbar_kws={'label': 'Correlation Coefficient'},
                    annot_kws={'size': 15, 'weight': 'normal'},
                    ax=ax)
        

        ax.set_xlabel('Dataset', fontsize=18)
        ax.set_ylabel('Layer Number', fontsize=18)
        
        plt.tight_layout()
        plt.savefig('ct_vs_4o_regular_layers_heatmap.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig('ct_vs_4o_regular_layers_heatmap.pdf', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()
        print("✅ Regular layers heatmap saved")
    
    # Create heatmap for special layers
    if len(special_df) > 0:
        # Use layer labels instead of numbers for special layers
        heatmap_data_special = special_df.pivot(index='layer_label', columns='dataset', values='correlation')
        heatmap_data_special.columns = [dataset_labels[col] for col in heatmap_data_special.columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))  # AAAI column width friendly
        
        sns.heatmap(heatmap_data_special, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='Blues',  # More professional colormap
                    cbar_kws={'label': 'Correlation Coefficient'},
                    annot_kws={'size': 16, 'weight': 'normal'},
                    ax=ax)
        

        ax.set_xlabel('Dataset', fontsize=18)
        ax.set_ylabel('Special Layer Type', fontsize=18)
        
        plt.tight_layout()
        plt.savefig('ct_vs_4o_special_layers_heatmap.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig('ct_vs_4o_special_layers_heatmap.pdf', bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.show()
        print("✅ Special layers heatmap saved")

def print_summary_statistics():
    """Print comprehensive summary statistics"""
    
    df = pd.read_csv('combined_ct_vs_4o_correlations_0.3.csv')
    
    # Create mapping for better labels
    dataset_labels = {
        'qwc7': 'Qwen Coder 7B',
        'qwc15': 'Qwen Coder 1.5B',
        'dsc7': 'Deepseek Coder 7B', 
        'dsc13': 'Deepseek Coder 1.3B'
    }
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"{'Dataset':<20} {'Total':<8} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Regular':<8} {'Special':<8}")
    print("-"*80)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_data = df[df['dataset'] == dataset]
        regular_data = dataset_data[(dataset_data['layer_num'] >= 0) & (dataset_data['layer_num'] <= 50)]
        special_data = dataset_data[dataset_data['layer_num'].isin([-1, 998, 999])]
        
        print(f"{dataset_labels[dataset]:<20} {len(dataset_data):<8} {dataset_data['correlation'].mean():<10.4f} "
              f"{dataset_data['correlation'].std():<10.4f} {dataset_data['correlation'].min():<8.4f} "
              f"{dataset_data['correlation'].max():<8.4f} {len(regular_data):<8} {len(special_data):<8}")
    
    # Special layers breakdown
    special_df = df[df['layer_num'].isin([-1, 998, 999])].copy()
    if len(special_df) > 0:
        print("\n" + "="*60)
        print("SPECIAL LAYERS BREAKDOWN")
        print("="*60)
        special_df['layer_label'] = special_df['layer_num'].apply(get_layer_label)
        
        for layer_num in sorted(special_df['layer_num'].unique()):
            layer_label = get_layer_label(layer_num)
            layer_data = special_df[special_df['layer_num'] == layer_num]
            print(f"\n{layer_label.upper()} (Layer {layer_num}):")
            print(f"{'Dataset':<20} {'Correlation':<12}")
            print("-"*32)
            for dataset in sorted(layer_data['dataset'].unique()):
                dataset_layer = layer_data[layer_data['dataset'] == dataset]
                if len(dataset_layer) > 0:
                    corr = dataset_layer['correlation'].iloc[0]
                    print(f"{dataset_labels[dataset]:<20} {corr:<12.4f}")
    
    print("\nNote: All correlations are between Code Sum (ct) and Code Gen (4o) models")
    print("Higher correlations indicate more similar representations between the two model types")

if __name__ == "__main__":
    print("🎨 Creating individual correlation plots...")
    
    print("\n1. Plotting regular transformer layers...")
    plot_regular_layers()
    
    print("\n2. Plotting special layers (embedding, final norm, lm head)...")
    plot_special_layers()
    
    print("\n3. Creating boxplot distribution...")
    plot_boxplot_distribution()
    
    print("\n4. Creating heatmaps...")
    create_combined_heatmap()
    
    print("\n5. Generating summary statistics...")
    print_summary_statistics()
    
    print("\n✅ All plots saved as individual files:")
    print("📊 Regular Layers:")
    print("  - ct_vs_4o_regular_layers_plot.png/pdf")
    print("  - ct_vs_4o_regular_layers_heatmap.png/pdf")
    print("🎯 Special Layers:")
    print("  - ct_vs_4o_special_layers_plot.png/pdf") 
    print("  - ct_vs_4o_special_layers_heatmap.png/pdf")
    print("📦 Distribution:")
    print("  - ct_vs_4o_boxplot_distribution.png/pdf") 