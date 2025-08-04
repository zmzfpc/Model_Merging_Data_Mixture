# Multi-task Code LLMs: Data Mix or Model Merge?

This repository contains the implementation for the paper **"Multi-task Code LLMs: Data Mix or Model Merge?"** submitted to AAAI.

## Overview

This research explores two fundamental approaches for creating multi-task code large language models:
1. **Data Mixing (DM)**: Training models on mixed multi-task datasets
2. **Model Merging (MM)**: Training task-specific models and merging them post-hoc

We provide comprehensive implementations, 28 pre-trained model checkpoints, and evaluation frameworks to reproduce our results.

**Download all 28 pre-trained models from Google Drive:**
[https://drive.google.com/drive/folders/MODEL_CHECKPOINTS_FOLDER_ID](https://drive.google.com/drive/folders/MODEL_CHECKPOINTS_FOLDER_ID)
## Repository Structure

```
├── llama_factory/          # SFT training framework (based on LLaMA-Factory)
│   ├── my_yaml/           # Training configuration files
│   │   ├── 2b/            # 2B parameter model configs
│   │   ├── 8b/            # 8B parameter model configs  
│   │   └── 20b/           # 20B parameter model configs
│   └── job_sh/            # Training job scripts
├── mergekit/              # Model merging toolkit and scripts
│   ├── job_sh/            # Merging job scripts
│   └── my_yaml/           # Merging configuration files
├── eval_hm/               # Evaluation framework
│   ├── job_sh/            # Evaluation job scripts
│   ├── eval_codesum.py    # Code summarization evaluation
│   └── json_to_markdowns.py  # Result processing
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU
- HuggingFace account with access tokens
- Weights & Biases account (optional)

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Model_Merging_Data_Mixture
```

2. **Install LLaMA-Factory:**
```bash
cd llama_factory
pip install -e .
# OR install requirements
pip install -r requirements.txt
```

3. **Install MergeKit:**
```bash
cd mergekit
pip install -e .
```

4. **Set up environment variables:**
```bash
export HF_HOME="/path/to/your/huggingface/cache"
export HF_TOKEN="your_huggingface_token_here"
export TOKENIZERS_PARALLELISM="false"

# Optional: for wandb logging
wandb login your_wandb_token_here
```

## Training Models (SFT)

### Configuration Files

Training configurations are organized in `llama_factory/my_yaml/` by model size:

- **2B models**: `2b/ct/`, `2b/kod/`, `2b/gs_dm/`
- **8B models**: `8b/kod/`, `8b/gs_dm/`  
- **20B models**: `20b/`

### Example Training Commands

1. **Single-task training (Code Summarization):**
```bash
cd llama_factory
llamafactory-cli train my_yaml/2b/ct/dscoder_full_sftct.yaml
```

2. **Single-task training (Code Generation):**
```bash
llamafactory-cli train my_yaml/2b/kod/dscoder_full_sft4o.yaml
```

3. **Multi-task training (Data Mixing):**
```bash
llamafactory-cli train my_yaml/2b/gs_dm/dscoder_full_sftdm.yaml
```

### Using Job Scripts

For batch job submission:
```bash
cd llama_factory/job_sh
# Edit paths in the script before running
bash run_sft_kod_qwc2515.sh
```

**⚠️ Important**: Update all file paths in job scripts to match your environment before running.

## Model Merging

### Merging Methods

We implement several state-of-the-art merging techniques:

- **Linear**: Simple weighted averaging
- **TIES**: Task Interference Elimination via Sparse merging  
- **DARE**: Drop And REscale for efficient merging
- **DELLA**: Depth-Localized LoRA for efficient merging

### Configuration Files

Merging configurations are in `mergekit/my_yaml/` organized by method and model combinations.

### Example Merging Commands

1. **TIES Merging:**
```bash
cd mergekit
mergekit-yaml my_yaml/merge_ties_config.yaml ./output/merged_model
```

2. **DARE Merging:**
```bash
mergekit-yaml my_yaml/merge_dare_config.yaml ./output/merged_model
```

### Using Merging Scripts

```bash
cd mergekit/job_sh
# Example: TIES merging for Qwen2.5-Coder-7B models
bash merge_ties_qwc7.sh
```


## Evaluation

### Supported Benchmarks

1. **Code Summarization**: Evaluate on code-to-docstring generation
2. **Code Generation**: Evaluate on programming problem solving (HumanEval, MBPP)
3. **Code Summarization**: Evaluate on cross-language code Summarization

### Running Evaluations

1. **Code Summarization:**
```bash
cd eval_hm
python eval_codesum.py \
    --model /path/to/your/model \
    --data /path/to/test/data.jsonl \
    --output results.json
```

2. **Code Generation (HumanEval):**
```bash
cd eval_hm/job_sh/g_eval
bash evalplus_sft_model_name.sh
```

3. **Using evaluation job scripts:**
```bash
cd eval_hm
bash sub_s_e.sh  # Submit code summarization evaluation jobs  
bash sub_g_e.sh  # Submit code generation evaluation jobs
```

### Processing Results

Convert summarization evaluation results to markdown format:
```bash
python json_to_markdowns.py \
    --input results/ \
    --output reports/ \
    --prefix experiment_
```

## Model Checkpoints

We provide **28 pre-trained model checkpoints** covering:

- **Base Models**: DeepSeek-Coder (1.3B, 7B), Qwen2.5-Coder (1.5B, 7B)
- **Task Specializations**: Code Generation, Code Summarization
- **Training Approaches**: Single-task SFT, Multi-task Data Mixing (DM)
- **Model Sizes**: 2B and 8B parameter variants

### Download Models

**Download all 28 pre-trained models from Google Drive:**
[https://drive.google.com/drive/folders/MODEL_CHECKPOINTS_FOLDER_ID](https://drive.google.com/drive/folders/MODEL_CHECKPOINTS_FOLDER_ID)

**Important**: Please modify the model paths in configuration files and scripts according to your download location. Update all references to model paths in:
- Job scripts (`llama_factory/job_sh/`, `mergekit/job_sh/`, `eval_hm/job_sh/`)
- Evaluation scripts



## Customization

### Adding New Tasks

1. Create dataset configuration in `llama_factory/data/`
2. Add training YAML in appropriate `my_yaml/` subdirectory
3. Create corresponding job script in `job_sh/`

### Adding New Merging Methods

1. Implement in MergeKit (if not already available)
2. Create configuration YAML in `mergekit/my_yaml/`
3. Add job script in `mergekit/job_sh/`

### Custom Evaluation

1. Add evaluation script in `eval_hm/`
2. Create job script in `eval_hm/job_sh/`
3. Update result processing in `json_to_markdowns.py`

## Reproducing Paper Results

### Full Experimental Pipeline

1. **Train single-task models:**
```bash
# Code Summarization models
bash llama_factory/job_sh/ct/run_sft_ct_*.sh

# Code Generation models  
bash llama_factory/job_sh/kod/run_sft_kod_*.sh
```

2. **Train data mixing models:**
```bash
bash llama_factory/job_sh/gs_dm/run_sft_dm_*.sh
```

3. **Merge single-task models:**
```bash
bash mergekit/job_sh/merge_ties_*.sh
bash mergekit/job_sh/merge_dare_*.sh
bash mergekit/job_sh/merge_della_*.sh
```

4. **Evaluate all models:**
```bash
bash eval_hm/sub_s_e.sh  # Code summarization
bash eval_hm/sub_g_e.sh  # Code generation
```

### Expected Results

Our experiments show that:
- **Data Mixing** generally outperforms model merging for 2B models
- **Larger models** benefit more from model merging strategies

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or use gradient checkpointing
2. **Path errors**: Update all absolute paths in job scripts
3. **Token errors**: Ensure HF_TOKEN is properly set
4. **Evaluation failures**: Check data paths and model compatibility

### Performance Tips

- Use `bf16` training for better performance
- Enable gradient checkpointing for large models
- Use multiple GPUs with DeepSpeed for 8B+ models
- Cache datasets to avoid repeated preprocessing

## Citation

If you use this code or our models, please cite our paper:

```bibtex
@article{anonymous2024multitask,
  title={Multi-task Code LLMs: Data Mix or Model Merge?},
  author={Anonymous},
  journal={AAAI},
  year={2026}
}
```

## Contributing

This repository is released for research purposes. For questions or issues, please open a GitHub issue.

## 📋 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**Note**: This repository contains the implementation for an anonymous submission. All identifying information has been removed for double-blind review.