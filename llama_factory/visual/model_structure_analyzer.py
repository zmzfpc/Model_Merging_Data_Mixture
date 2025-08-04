#!/usr/bin/env python3
"""
Model Structure Analyzer

Analyzes the weight structure of transformer models, showing:
- Layer count and organization
- Component breakdown (attention, MLP, normalization)
- Parameter counts and distributions
- Weight tensor shapes and sizes
- Hierarchical structure visualization

Usage:
    python model_structure_analyzer.py --model_path path/to/model
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
from typing import Dict, Tuple, List, Optional
import re
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_style("whitegrid")

class ModelStructureAnalyzer:
    def __init__(self, model_path: str, output_dir: str = "model_structure_analysis"):
        self.model_path_str = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔍 Model Structure Analysis:")
        print(f"   Model Path: {model_path}")
        print(f"   Output: {self.output_dir}")
        
        # Determine if model is from HuggingFace or local path
        self.is_hf_model = not Path(model_path).exists()
        if self.is_hf_model:
            print(f"   📥 Detected HuggingFace model: {model_path}")
            self.model_path = self._download_hf_model(model_path)
        else:
            self.model_path = Path(model_path)
        
        # Load configuration and weight mapping
        self.config = self._load_config()
        self.weight_map = self._load_weight_map()
        self.architecture = self._detect_architecture()
        
        print(f"📐 Architecture: {self.architecture}")
        print(f"🔢 Config Layers: {self.config.get('num_hidden_layers', 'unknown')}")

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

    def _load_config(self) -> dict:
        """Load model configuration"""
        config_path = self.model_path / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_weight_map(self) -> dict:
        """Load weight mapping from safetensors index or single file"""
        index_path = self.model_path / "model.safetensors.index.json"
        
        if index_path.exists():
            # Sharded model with index
            with open(index_path, 'r') as f:
                return json.load(f)["weight_map"]
        else:
            # Single safetensors file
            single_file_path = self.model_path / "model.safetensors"
            if single_file_path.exists():
                with safe_open(single_file_path, framework="pt", device="cpu") as f:
                    weight_map = {}
                    for key in f.keys():
                        weight_map[key] = "model.safetensors"
                    return weight_map
            else:
                raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {self.model_path}")

    def _detect_architecture(self) -> str:
        """Detect model architecture"""
        if "architectures" in self.config:
            return self.config["architectures"][0]
        elif "model_type" in self.config:
            return self.config["model_type"]
        else:
            return "unknown"

    def parse_weight_info(self, weight_name: str) -> Dict:
        """Parse detailed information from weight name"""
        info = {
            'name': weight_name,
            'layer_num': None,
            'component': 'other',
            'subcomponent': 'unknown',
            'param_type': weight_name.split('.')[-1],
            'is_special': False
        }
        
        # Special layers
        if 'embed_tokens' in weight_name or 'embed_positions' in weight_name:
            info.update({'layer_num': -1, 'component': 'embedding', 'subcomponent': 'tokens', 'is_special': True})
        elif 'lm_head' in weight_name or 'output' in weight_name:
            info.update({'layer_num': 999, 'component': 'lm_head', 'subcomponent': 'projection', 'is_special': True})
        elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
            if 'model.norm' in weight_name:
                info.update({'layer_num': 998, 'component': 'final_norm', 'subcomponent': 'norm', 'is_special': True})
        
        # Regular layers
        layer_match = re.search(r'layers?\.(\d+)\.', weight_name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            info['layer_num'] = layer_num
            
            # Attention components
            if 'self_attn' in weight_name or 'attention' in weight_name:
                info['component'] = 'attention'
                if 'q_proj' in weight_name or 'query' in weight_name:
                    info['subcomponent'] = 'query'
                elif 'k_proj' in weight_name or 'key' in weight_name:
                    info['subcomponent'] = 'key'
                elif 'v_proj' in weight_name or 'value' in weight_name:
                    info['subcomponent'] = 'value'
                elif 'o_proj' in weight_name or 'dense' in weight_name:
                    info['subcomponent'] = 'output'
                else:
                    info['subcomponent'] = 'other_attn'
            
            # MLP components
            elif 'mlp' in weight_name or 'feed_forward' in weight_name:
                info['component'] = 'mlp'
                if 'gate_proj' in weight_name or 'w1' in weight_name:
                    info['subcomponent'] = 'gate'
                elif 'up_proj' in weight_name or 'w3' in weight_name:
                    info['subcomponent'] = 'up'
                elif 'down_proj' in weight_name or 'w2' in weight_name:
                    info['subcomponent'] = 'down'
                elif 'intermediate' in weight_name:
                    info['subcomponent'] = 'intermediate'
                else:
                    info['subcomponent'] = 'other_mlp'
            
            # Layer normalization
            elif 'layernorm' in weight_name or 'layer_norm' in weight_name:
                info['component'] = 'layernorm'
                if 'input_layernorm' in weight_name or 'ln_1' in weight_name:
                    info['subcomponent'] = 'input_norm'
                elif 'post_attention_layernorm' in weight_name or 'ln_2' in weight_name:
                    info['subcomponent'] = 'post_attn_norm'
                else:
                    info['subcomponent'] = 'other_norm'
        
        return info

    def analyze_structure(self) -> Dict:
        """Analyze the complete model structure"""
        print("🔍 Analyzing model structure...")
        
        structure = {
            'total_params': 0,
            'total_tensors': len(self.weight_map),
            'layers': defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
            'components': defaultdict(lambda: {'count': 0, 'params': 0, 'tensors': []}),
            'special_layers': {},
            'layer_stats': defaultdict(lambda: {'params': 0, 'tensors': 0, 'components': set()}),
            'param_types': defaultdict(lambda: {'count': 0, 'params': 0}),
            'shapes': defaultdict(list)
        }
        
        # Get all weight files
        safetensor_files = set(self.weight_map.values())
        
        for i, file in enumerate(safetensor_files, 1):
            file_path = self.model_path / file
            print(f"   📊 Analyzing {file} ({i}/{len(safetensor_files)})...")
            
            if not file_path.exists():
                print(f"   ⚠️ Warning: {file} not found, skipping...")
                continue
                
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in self.weight_map:
                        tensor = f.get_tensor(key)
                        weight_info = self.parse_weight_info(key)
                        
                        # Basic stats
                        param_count = tensor.numel()
                        structure['total_params'] += param_count
                        
                        # Shape information
                        shape_str = f"{list(tensor.shape)}"
                        structure['shapes'][shape_str].append(key)
                        
                        # Parameter type stats
                        param_type = weight_info['param_type']
                        structure['param_types'][param_type]['count'] += 1
                        structure['param_types'][param_type]['params'] += param_count
                        
                        # Component-level organization
                        component = weight_info['component']
                        structure['components'][component]['count'] += 1
                        structure['components'][component]['params'] += param_count
                        structure['components'][component]['tensors'].append({
                            'name': key,
                            'shape': list(tensor.shape),
                            'params': param_count,
                            'subcomponent': weight_info['subcomponent']
                        })
                        
                        # Layer-specific analysis
                        layer_num = weight_info['layer_num']
                        if layer_num is not None:
                            if weight_info['is_special']:
                                # Special layers
                                structure['special_layers'][layer_num] = structure['special_layers'].get(layer_num, {
                                    'name': component,
                                    'params': 0,
                                    'tensors': []
                                })
                                structure['special_layers'][layer_num]['params'] += param_count
                                structure['special_layers'][layer_num]['tensors'].append({
                                    'name': key,
                                    'shape': list(tensor.shape),
                                    'params': param_count
                                })
                            else:
                                # Regular layers
                                structure['layer_stats'][layer_num]['params'] += param_count
                                structure['layer_stats'][layer_num]['tensors'] += 1
                                structure['layer_stats'][layer_num]['components'].add(component)
                                
                                # Detailed layer structure
                                subcomp = weight_info['subcomponent']
                                if subcomp not in structure['layers'][layer_num][component]:
                                    structure['layers'][layer_num][component][subcomp] = {
                                        'params': 0,
                                        'tensors': [],
                                        'param_types': defaultdict(int)
                                    }
                                
                                structure['layers'][layer_num][component][subcomp]['params'] += param_count
                                structure['layers'][layer_num][component][subcomp]['tensors'].append({
                                    'name': key,
                                    'shape': list(tensor.shape),
                                    'params': param_count
                                })
                                structure['layers'][layer_num][component][subcomp]['param_types'][param_type] += param_count
        
        # Convert defaultdicts to regular dicts for JSON serialization
        structure['layers'] = {k: dict(v) for k, v in structure['layers'].items()}
        for layer_data in structure['layers'].values():
            for comp_data in layer_data.values():
                for subcomp_key in comp_data:
                    if isinstance(comp_data[subcomp_key]['param_types'], defaultdict):
                        comp_data[subcomp_key]['param_types'] = dict(comp_data[subcomp_key]['param_types'])
        
        structure['components'] = dict(structure['components'])
        structure['layer_stats'] = dict(structure['layer_stats'])
        structure['param_types'] = dict(structure['param_types'])
        structure['shapes'] = dict(structure['shapes'])
        
        # Convert sets to lists
        for layer_stat in structure['layer_stats'].values():
            layer_stat['components'] = list(layer_stat['components'])
        
        return structure

    def print_structure_summary(self, structure: Dict):
        """Print comprehensive structure summary"""
        print("\n" + "="*80)
        print("MODEL STRUCTURE SUMMARY")
        print("="*80)
        
        # Basic info
        print(f"📊 Total Parameters: {structure['total_params']:,}")
        print(f"📦 Total Tensors: {structure['total_tensors']:,}")
        print(f"🏗️ Architecture: {self.architecture}")
        
        # Special layers
        if structure['special_layers']:
            print("\n🎯 SPECIAL LAYERS:")
            print("-" * 40)
            for layer_num in sorted(structure['special_layers'].keys()):
                layer_info = structure['special_layers'][layer_num]
                layer_name = {-1: 'Embedding', 998: 'Final Norm', 999: 'LM Head'}.get(layer_num, f'Special {layer_num}')
                print(f"  {layer_name}: {layer_info['params']:,} params ({len(layer_info['tensors'])} tensors)")
        
        # Regular layers summary
        regular_layers = [k for k in structure['layer_stats'].keys() if 0 <= k <= 100]
        if regular_layers:
            print(f"\n🔄 TRANSFORMER LAYERS: {len(regular_layers)} layers")
            print("-" * 40)
            
            total_layer_params = sum(structure['layer_stats'][i]['params'] for i in regular_layers)
            avg_params = total_layer_params // len(regular_layers) if regular_layers else 0
            
            print(f"  Range: Layer {min(regular_layers)} to {max(regular_layers)}")
            print(f"  Total Layer Params: {total_layer_params:,}")
            print(f"  Average per Layer: {avg_params:,}")
            
            # Show first few layers in detail
            print(f"\n  Sample Layer Structure (Layer {regular_layers[0]}):")
            if regular_layers[0] in structure['layers']:
                for component, comp_data in structure['layers'][regular_layers[0]].items():
                    comp_params = sum(subcomp['params'] for subcomp in comp_data.values())
                    print(f"    {component}: {comp_params:,} params")
                    for subcomp, subcomp_data in comp_data.items():
                        print(f"      └─ {subcomp}: {subcomp_data['params']:,} params")
        
        # Component breakdown
        print(f"\n🧩 COMPONENT BREAKDOWN:")
        print("-" * 40)
        for component, comp_data in sorted(structure['components'].items(), key=lambda x: x[1]['params'], reverse=True):
            percentage = (comp_data['params'] / structure['total_params']) * 100
            print(f"  {component}: {comp_data['params']:,} params ({percentage:.1f}%) - {comp_data['count']} tensors")
        
        # Parameter types
        print(f"\n📋 PARAMETER TYPES:")
        print("-" * 40)
        for param_type, type_data in sorted(structure['param_types'].items(), key=lambda x: x[1]['params'], reverse=True):
            percentage = (type_data['params'] / structure['total_params']) * 100
            print(f"  {param_type}: {type_data['params']:,} params ({percentage:.1f}%) - {type_data['count']} tensors")
        
        # Common shapes
        print(f"\n📐 COMMON TENSOR SHAPES:")
        print("-" * 40)
        shape_counts = {shape: len(tensors) for shape, tensors in structure['shapes'].items()}
        for shape, count in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {shape}: {count} tensors")

    def create_structure_visualizations(self, structure: Dict):
        """Create comprehensive structure visualizations"""
        print("📊 Creating structure visualizations...")
        
        # 1. Parameter distribution by component
        self._plot_component_distribution(structure)
        
        # 2. Layer-by-layer parameter analysis
        self._plot_layer_analysis(structure)
        
        # 3. Tensor shape analysis
        self._plot_shape_analysis(structure)
        
        # 4. Architecture overview
        self._plot_architecture_overview(structure)

    def _plot_component_distribution(self, structure: Dict):
        """Plot component parameter distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pie chart
        components = list(structure['components'].keys())
        params = [structure['components'][comp]['params'] for comp in components]
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        
        wedges, texts, autotexts = ax1.pie(params, labels=components, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax1.set_title('Parameter Distribution by Component', fontweight='bold', fontsize=14)
        
        # Bar chart
        ax2.barh(components, params, color=colors)
        ax2.set_xlabel('Parameters')
        ax2.set_title('Parameter Count by Component', fontweight='bold', fontsize=14)
        ax2.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'component_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_layer_analysis(self, structure: Dict):
        """Plot layer-by-layer analysis"""
        regular_layers = sorted([k for k in structure['layer_stats'].keys() if 0 <= k <= 100])
        
        if not regular_layers:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Layer parameter counts
        layer_params = [structure['layer_stats'][i]['params'] for i in regular_layers]
        layer_tensors = [structure['layer_stats'][i]['tensors'] for i in regular_layers]
        
        axes[0, 0].plot(regular_layers, layer_params, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Parameters per Layer', fontweight='bold')
        axes[0, 0].set_xlabel('Layer Number')
        axes[0, 0].set_ylabel('Parameter Count')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Tensor counts per layer
        axes[0, 1].plot(regular_layers, layer_tensors, 'g-s', linewidth=2, markersize=4)
        axes[0, 1].set_title('Tensors per Layer', fontweight='bold')
        axes[0, 1].set_xlabel('Layer Number')
        axes[0, 1].set_ylabel('Tensor Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Component breakdown for sample layers
        sample_layers = regular_layers[::max(1, len(regular_layers)//10)][:10]
        component_data = defaultdict(list)
        
        for layer in sample_layers:
            layer_data = structure['layers'].get(layer, {})
            for component in ['attention', 'mlp', 'layernorm']:
                comp_params = sum(subcomp['params'] for subcomp in layer_data.get(component, {}).values())
                component_data[component].append(comp_params)
        
        x_pos = np.arange(len(sample_layers))
        width = 0.25
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (component, params) in enumerate(component_data.items()):
            axes[1, 0].bar(x_pos + i*width, params, width, label=component, color=colors[i], alpha=0.8)
        
        axes[1, 0].set_title('Component Distribution Across Layers', fontweight='bold')
        axes[1, 0].set_xlabel('Layer Number')
        axes[1, 0].set_ylabel('Parameters')
        axes[1, 0].set_xticks(x_pos + width)
        axes[1, 0].set_xticklabels(sample_layers)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Special layers
        if structure['special_layers']:
            special_names = []
            special_params = []
            name_map = {-1: 'Embedding', 998: 'Final Norm', 999: 'LM Head'}
            
            for layer_num in sorted(structure['special_layers'].keys()):
                special_names.append(name_map.get(layer_num, f'Special {layer_num}'))
                special_params.append(structure['special_layers'][layer_num]['params'])
            
            axes[1, 1].bar(special_names, special_params, color='purple', alpha=0.7)
            axes[1, 1].set_title('Special Layers Parameters', fontweight='bold')
            axes[1, 1].set_ylabel('Parameters')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_shape_analysis(self, structure: Dict):
        """Plot tensor shape analysis"""
        # Get most common shapes
        shape_counts = {shape: len(tensors) for shape, tensors in structure['shapes'].items()}
        top_shapes = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Shape frequency
        shapes, counts = zip(*top_shapes)
        axes[0].barh(range(len(shapes)), counts, color='skyblue', alpha=0.8)
        axes[0].set_yticks(range(len(shapes)))
        axes[0].set_yticklabels(shapes, fontsize=8)
        axes[0].set_xlabel('Number of Tensors')
        axes[0].set_title('Most Common Tensor Shapes', fontweight='bold', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Parameter distribution by shape
        shape_params = []
        for shape in shapes:
            shape_eval = eval(shape)  # Convert string back to list
            param_count = np.prod(shape_eval) if shape_eval else 0
            tensor_count = shape_counts[shape]
            total_params = param_count * tensor_count
            shape_params.append(total_params)
        
        axes[1].barh(range(len(shapes)), shape_params, color='lightcoral', alpha=0.8)
        axes[1].set_yticks(range(len(shapes)))
        axes[1].set_yticklabels(shapes, fontsize=8)
        axes[1].set_xlabel('Total Parameters')
        axes[1].set_title('Parameter Count by Shape', fontweight='bold', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'shape_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_architecture_overview(self, structure: Dict):
        """Plot architecture overview"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a hierarchical view
        y_pos = 0
        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        
        # Special layers first
        if structure['special_layers']:
            for i, (layer_num, layer_info) in enumerate(sorted(structure['special_layers'].items())):
                layer_name = {-1: 'Embedding', 998: 'Final Norm', 999: 'LM Head'}.get(layer_num, f'Special {layer_num}')
                width = layer_info['params'] / structure['total_params'] * 10
                
                rect = plt.Rectangle((0, y_pos), width, 0.5, 
                                   facecolor=colors[i % len(colors)], alpha=0.7, 
                                   edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(width/2, y_pos+0.25, f'{layer_name}\n{layer_info["params"]:,}', 
                       ha='center', va='center', fontsize=8, fontweight='bold')
                y_pos += 0.7
        
        # Regular layers (grouped)
        regular_layers = sorted([k for k in structure['layer_stats'].keys() if 0 <= k <= 100])
        if regular_layers:
            total_layer_params = sum(structure['layer_stats'][i]['params'] for i in regular_layers)
            width = total_layer_params / structure['total_params'] * 10
            
            rect = plt.Rectangle((0, y_pos), width, 1.0, 
                               facecolor='lightblue', alpha=0.7, 
                               edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(width/2, y_pos+0.5, f'Transformer Layers\n{len(regular_layers)} layers\n{total_layer_params:,} params', 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 12)
        ax.set_ylim(-0.5, y_pos + 1.5)
        ax.set_xlabel('Relative Parameter Size')
        ax.set_title('Model Architecture Overview', fontweight='bold', fontsize=16)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_structure_data(self, structure: Dict):
        """Save structure analysis to files"""
        print("💾 Saving structure data...")
        
        # Save complete structure as JSON
        structure_json = structure.copy()
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(self.output_dir / 'model_structure.json', 'w') as f:
            json.dump(structure_json, f, indent=2, default=convert_numpy)
        
        # Create summary CSV
        summary_data = []
        
        # Add special layers
        for layer_num, layer_info in structure['special_layers'].items():
            layer_name = {-1: 'Embedding', 998: 'Final Norm', 999: 'LM Head'}.get(layer_num, f'Special {layer_num}')
            summary_data.append({
                'layer_type': 'special',
                'layer_num': layer_num,
                'layer_name': layer_name,
                'component': layer_info['name'],
                'subcomponent': '',
                'parameters': layer_info['params'],
                'tensors': len(layer_info['tensors'])
            })
        
        # Add regular layers
        regular_layers = sorted([k for k in structure['layer_stats'].keys() if 0 <= k <= 100])
        for layer_num in regular_layers:
            layer_data = structure['layers'].get(layer_num, {})
            for component, comp_data in layer_data.items():
                for subcomponent, subcomp_data in comp_data.items():
                    summary_data.append({
                        'layer_type': 'regular',
                        'layer_num': layer_num,
                        'layer_name': f'Layer {layer_num}',
                        'component': component,
                        'subcomponent': subcomponent,
                        'parameters': subcomp_data['params'],
                        'tensors': len(subcomp_data['tensors'])
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'structure_summary.csv', index=False)
        
        print(f"   💾 Saved structure data:")
        print(f"       - model_structure.json (complete structure)")
        print(f"       - structure_summary.csv (tabular summary)")

    def run_analysis(self):
        """Run complete structure analysis"""
        print("\n🚀 Starting model structure analysis...")
        
        # Analyze structure
        structure = self.analyze_structure()
        
        # Print summary
        self.print_structure_summary(structure)
        
        # Create visualizations
        self.create_structure_visualizations(structure)
        
        # Save data
        self.save_structure_data(structure)
        
        print("\n✅ Model structure analysis complete!")
        
        # Count generated files
        png_files = len(list(self.output_dir.glob('*.png')))
        data_files = len(list(self.output_dir.glob('*.json'))) + len(list(self.output_dir.glob('*.csv')))
        
        print(f"📊 Generated:")
        print(f"   - {png_files} visualization plots")
        print(f"   - {data_files} data files")

def main():
    parser = argparse.ArgumentParser(description='Model structure analysis')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model (local path or HuggingFace model name)')
    parser.add_argument('--output', type=str, default='model_structure_analysis', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ModelStructureAnalyzer(args.model_path, args.output)
    
    # Run analysis
    analyzer.run_analysis()
    
    print(f"\n🎯 Analysis complete! Check {args.output}/ for results")

if __name__ == "__main__":
    main() 