#!/usr/bin/env python3
"""
Generate a comprehensive analysis report comparing the two SFT checkpoints.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_analysis_data():
    """Load all analysis results"""
    results_dir = Path("analysis_results")
    
    # Load CSV files
    overall = pd.read_csv(results_dir / "overall_summary.csv")
    layers = pd.read_csv(results_dir / "layer_summary.csv")
    components = pd.read_csv(results_dir / "component_summary.csv")
    detailed = pd.read_csv(results_dir / "detailed_weight_differences.csv")
    
    # Load JSON
    with open(results_dir / "layer_analysis.json", 'r') as f:
        layer_json = json.load(f)
    
    return overall, layers, components, detailed, layer_json

def create_insights_report(overall, layers, components, detailed):
    """Generate key insights from the analysis"""
    
    insights = []
    
    # Overall insights
    total_params = int(overall[overall['Metric'] == 'Total Parameters']['Value'].iloc[0].replace(',', ''))
    mean_cosine_sim = float(overall[overall['Metric'] == 'Mean Cosine Similarity']['Value'].iloc[0])
    mean_rel_diff = float(overall[overall['Metric'] == 'Mean Relative Difference']['Value'].iloc[0])
    
    insights.append("=== KEY INSIGHTS ===\n")
    
    # Model similarity
    if mean_cosine_sim > 0.99:
        insights.append("🔍 **High Model Similarity**: The two checkpoints are highly similar with average cosine similarity of {:.6f}".format(mean_cosine_sim))
    else:
        insights.append("🔍 **Moderate Model Similarity**: The two checkpoints show moderate similarity with average cosine similarity of {:.6f}".format(mean_cosine_sim))
    
    # Relative differences
    if mean_rel_diff < 0.01:
        insights.append("✅ **Small Weight Changes**: Average relative difference of {:.6f} indicates fine-tuning preserved most original weights".format(mean_rel_diff))
    else:
        insights.append("⚠️ **Significant Weight Changes**: Average relative difference of {:.6f} indicates substantial adaptation".format(mean_rel_diff))
    
    # Layer analysis
    layer_diffs = layers[layers['Layer'].str.contains('Layer')]['Mean L2 Diff'].astype(float)
    min_layer = layer_diffs.idxmin()
    max_layer = layer_diffs.idxmax()
    
    insights.append("\n=== LAYER-WISE PATTERNS ===")
    insights.append("🎯 **Most Changed Layer**: {} with L2 diff of {:.6f}".format(
        layers.iloc[max_layer]['Layer'], layer_diffs.iloc[max_layer]))
    insights.append("🎯 **Least Changed Layer**: {} with L2 diff of {:.6f}".format(
        layers.iloc[min_layer]['Layer'], layer_diffs.iloc[min_layer]))
    
    # Check for gradient pattern
    middle_layers = layer_diffs.iloc[len(layer_diffs)//4:3*len(layer_diffs)//4]
    early_layers = layer_diffs.iloc[:len(layer_diffs)//4]
    late_layers = layer_diffs.iloc[3*len(layer_diffs)//4:]
    
    if middle_layers.mean() > early_layers.mean() and middle_layers.mean() > late_layers.mean():
        insights.append("📊 **U-shaped Pattern**: Middle layers show more changes than early/late layers")
    elif early_layers.mean() > middle_layers.mean() > late_layers.mean():
        insights.append("📊 **Decreasing Pattern**: Earlier layers changed more than later layers")
    elif late_layers.mean() > middle_layers.mean() > early_layers.mean():
        insights.append("📊 **Increasing Pattern**: Later layers changed more than earlier layers")
    
    # Component analysis
    mlp_components = components[components['Component'].str.contains('mlp')]
    attn_components = components[components['Component'].str.contains('self_attn')]
    norm_components = components[components['Component'].str.contains('layernorm')]
    
    insights.append("\n=== COMPONENT-WISE PATTERNS ===")
    insights.append("🧠 **MLP vs Attention**:")
    insights.append("   - MLP mean L2 diff: {:.6f}".format(mlp_components['Mean L2 Diff'].astype(float).mean()))
    insights.append("   - Attention mean L2 diff: {:.6f}".format(attn_components['Mean L2 Diff'].astype(float).mean()))
    insights.append("   - LayerNorm mean L2 diff: {:.6f}".format(norm_components['Mean L2 Diff'].astype(float).mean()))
    
    # Embedding vs LM Head
    embedding_diff = float(layers[layers['Layer'] == 'Embedding']['Mean L2 Diff'].iloc[0])
    lm_head_diff = float(layers[layers['Layer'] == 'LM Head']['Mean L2 Diff'].iloc[0])
    
    insights.append("\n=== SPECIAL LAYERS ===")
    insights.append("📝 **Token Embeddings**: L2 diff of {:.6f}".format(embedding_diff))
    insights.append("📝 **LM Head (Output)**: L2 diff of {:.6f}".format(lm_head_diff))
    
    if lm_head_diff > embedding_diff:
        insights.append("   → Output layer changed more than input embeddings")
    else:
        insights.append("   → Input embeddings changed more than output layer")
    
    # Task specialization insights
    insights.append("\n=== TASK SPECIALIZATION INSIGHTS ===")
    
    # Check if certain components show high variance
    component_std = components.groupby(components['Component'].str.split('.').str[0])['Mean L2 Diff'].agg(['mean', 'std'])
    high_variance_components = component_std[component_std['std'] > component_std['std'].mean()]
    
    if len(high_variance_components) > 0:
        insights.append("🎯 **Variable Adaptation**: Some component types show high variance in changes:")
        for comp in high_variance_components.index:
            insights.append("   - {}: std = {:.6f}".format(comp, high_variance_components.loc[comp, 'std']))
    
    return "\n".join(insights)

def create_advanced_visualizations(layers, components, detailed):
    """Create additional visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Layer progression analysis
    ax1 = fig.add_subplot(gs[0, 0])
    layer_data = layers[layers['Layer'].str.contains('Layer')].copy()
    layer_nums = [int(x.split()[1]) for x in layer_data['Layer']]
    layer_data['Layer_Num'] = layer_nums
    layer_data = layer_data.sort_values('Layer_Num')
    
    ax1.plot(layer_data['Layer_Num'], layer_data['Mean L2 Diff'].astype(float), 'o-', linewidth=2, markersize=6)
    ax1.fill_between(layer_data['Layer_Num'], layer_data['Mean L2 Diff'].astype(float), alpha=0.3)
    ax1.set_title('Layer-wise L2 Differences', fontweight='bold')
    ax1.set_xlabel('Layer Number')
    ax1.set_ylabel('Mean L2 Difference')
    ax1.grid(True, alpha=0.3)
    
    # 2. Component comparison (grouped bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Group components by type
    mlp_comps = components[components['Component'].str.contains('mlp')]['Mean L2 Diff'].astype(float)
    attn_comps = components[components['Component'].str.contains('self_attn')]['Mean L2 Diff'].astype(float)
    norm_comps = components[components['Component'].str.contains('layernorm')]['Mean L2 Diff'].astype(float)
    
    comp_means = [mlp_comps.mean(), attn_comps.mean(), norm_comps.mean()]
    comp_stds = [mlp_comps.std(), attn_comps.std(), norm_comps.std()]
    comp_names = ['MLP', 'Attention', 'LayerNorm']
    
    bars = ax2.bar(comp_names, comp_means, yerr=comp_stds, capsize=5, alpha=0.7, 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('Component Type Comparison', fontweight='bold')
    ax2.set_ylabel('Mean L2 Difference')
    
    # Add value labels on bars
    for bar, mean in zip(bars, comp_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean:.3f}', ha='center', va='bottom')
    
    # 3. Relative vs Absolute differences scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(detailed['l2_norm_diff'], detailed['relative_diff'], alpha=0.6, s=30)
    ax3.set_xlabel('L2 Norm Difference')
    ax3.set_ylabel('Relative Difference')
    ax3.set_title('Absolute vs Relative Changes', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(detailed['l2_norm_diff'], detailed['relative_diff'])[0,1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    # 4. Distribution of cosine similarities
    ax4 = fig.add_subplot(gs[1, 0])
    cos_sims = detailed['cosine_sim']
    ax4.hist(cos_sims, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(cos_sims.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {cos_sims.mean():.4f}')
    ax4.axvline(cos_sims.median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {cos_sims.median():.4f}')
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Weight Similarities', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameter count vs changes
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(detailed['num_params'], detailed['l2_norm_diff'], alpha=0.6, s=30)
    ax5.set_xlabel('Number of Parameters')
    ax5.set_ylabel('L2 Difference')
    ax5.set_title('Parameter Count vs Changes', fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 6. Layer depth vs changes heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Create a matrix showing layer vs component type changes
    layer_comp_matrix = []
    layer_nums = []
    comp_types = ['self_attn', 'mlp', 'layernorm']
    
    for i in range(28):  # 28 layers
        layer_weights = detailed[detailed['weight_name'].str.contains(f'model.layers.{i}.')]
        row = []
        for comp_type in comp_types:
            comp_weights = layer_weights[layer_weights['weight_name'].str.contains(comp_type)]
            if len(comp_weights) > 0:
                row.append(comp_weights['l2_norm_diff'].mean())
            else:
                row.append(0)
        layer_comp_matrix.append(row)
        layer_nums.append(i)
    
    im = ax6.imshow(np.array(layer_comp_matrix).T, aspect='auto', cmap='viridis')
    ax6.set_xticks(range(0, 28, 4))
    ax6.set_xticklabels(range(0, 28, 4))
    ax6.set_yticks(range(len(comp_types)))
    ax6.set_yticklabels(comp_types)
    ax6.set_xlabel('Layer Number')
    ax6.set_title('Layer vs Component Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax6, label='Mean L2 Difference')
    
    # 7. Weight magnitude analysis
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.scatter(detailed['l2_norm_w1'], detailed['l2_norm_w2'], alpha=0.6, s=30)
    ax7.plot([detailed['l2_norm_w1'].min(), detailed['l2_norm_w1'].max()], 
             [detailed['l2_norm_w1'].min(), detailed['l2_norm_w1'].max()], 
             'r--', label='Perfect Match')
    ax7.set_xlabel('Checkpoint 1 Weight Norm')
    ax7.set_ylabel('Checkpoint 2 Weight Norm')
    ax7.set_title('Weight Magnitude Comparison', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Distribution comparison
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(detailed['l2_norm_w1'], bins=30, alpha=0.5, label='Checkpoint 1', color='blue')
    ax8.hist(detailed['l2_norm_w2'], bins=30, alpha=0.5, label='Checkpoint 2', color='red')
    ax8.set_xlabel('Weight Norm')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Weight Norm Distributions', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary statistics
    summary_stats = {
        'Metric': [
            'Total Weights Compared',
            'Avg L2 Difference',
            'Avg Relative Diff',
            'Avg Cosine Similarity',
            'Max L2 Difference',
            'Min Cosine Similarity'
        ],
        'Value': [
            f'{len(detailed):,}',
            f'{detailed["l2_norm_diff"].mean():.6f}',
            f'{detailed["relative_diff"].mean():.6f}',
            f'{detailed["cosine_sim"].mean():.6f}',
            f'{detailed["l2_norm_diff"].max():.6f}',
            f'{detailed["cosine_sim"].min():.6f}'
        ]
    }
    
    # Create table
    table_data = list(zip(summary_stats['Metric'], summary_stats['Value']))
    table = ax9.table(cellText=table_data, 
                      colLabels=['Metric', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.suptitle('Comprehensive Checkpoint Analysis - Advanced Visualizations', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('analysis_results/advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report():
    """Generate the complete analysis report"""
    
    print("Loading analysis data...")
    overall, layers, components, detailed, layer_json = load_analysis_data()
    
    print("Generating insights report...")
    insights = create_insights_report(overall, layers, components, detailed)
    
    print("Creating advanced visualizations...")
    create_advanced_visualizations(layers, components, detailed)
    
    # Save the insights report
    with open('analysis_results/insights_report.txt', 'w') as f:
        f.write("COMPREHENSIVE CHECKPOINT ANALYSIS REPORT\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoints Compared:\n")
        f.write("- Checkpoint 1: sft_4o_sol_5e6 (GPT-4 Omni Solutions, 5e-6 LR)\n")
        f.write("- Checkpoint 2: sft_ct_1e6 (Custom Task, 1e-6 LR)\n\n")
        f.write(insights)
        f.write("\n\n" + "="*50 + "\n")
        f.write("For detailed numerical results, see:\n")
        f.write("- overall_summary.csv\n")
        f.write("- layer_summary.csv\n")
        f.write("- component_summary.csv\n")
        f.write("- detailed_weight_differences.csv\n")
        f.write("\nFor visualizations, see:\n")
        f.write("- checkpoint_analysis.png\n")
        f.write("- component_analysis.png\n")
        f.write("- advanced_analysis.png\n")
    
    print("\n" + "="*60)
    print(insights)
    print("="*60)
    print("\nComplete analysis saved to 'analysis_results/insights_report.txt'")
    print("Advanced visualizations saved to 'analysis_results/advanced_analysis.png'")

if __name__ == "__main__":
    generate_comprehensive_report() 