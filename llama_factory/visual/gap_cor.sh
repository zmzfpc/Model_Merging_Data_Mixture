
export HF_HOME="/path/to/your/project/huggingface"

source /path/to/your/miniconda3/etc/profile.d/conda.sh 

conda activate mergel 

threshold=0.3
echo "threshold: $threshold"

# python gap_correlation_analyzer.py --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
#   --sft_models saves/qwen25c7/best/sft_ct_1e6 saves/qwen25c7/best/sft_4o_sol_5e6 \
#   --output vis/gap_cor_qwc7_{$threshold} --filter_small_gaps --gap_threshold $threshold


# python gap_correlation_analyzer.py --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
#   --sft_models saves/qwen2515c15/best/sft_ct_5e6 \
#   saves/qwen2515c15/best/sft_4o_sol_1e-5 \
#   --output vis/gap_cor_qwc15_{$threshold} --filter_small_gaps --gap_threshold $threshold


# python gap_correlation_analyzer.py --base_model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#   --sft_models saves/dsc7/best/sft_ct_1e6 saves/dsc7/best/sft_4o_sol_5e6 \
#   --output vis/gap_cor_dsc7_{$threshold} --filter_small_gaps --gap_threshold $threshold

# python gap_correlation_analyzer.py --base_model deepseek-ai/deepseek-coder-1.3b-instruct \
#   --sft_models saves/dsc13/best/sft_ct_5e6 saves/dsc13/best/sft_4o_sol_1e5 \
#   --output vis/gap_cor_dsc13_{$threshold} --filter_small_gaps --gap_threshold $threshold


python gap_correlation_analyzer.py --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
  --sft_models saves/qwen25c7/dm/5e6 /path/to/your/project/mergekit/merged_model/della_qwc7_gs \
  --output vis/gap_cor_qwc7_{$threshold}_dm --filter_small_gaps --gap_threshold $threshold


python gap_correlation_analyzer.py --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --sft_models saves/qwen25c15/dm/1e5  \
  /path/to/your/project/mergekit/merged_model/dare_qwc15_gs \
  --output vis/gap_cor_qwc15_{$threshold}_dm --filter_small_gaps --gap_threshold $threshold


python gap_correlation_analyzer.py --base_model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
  --sft_models /path/to/your/project/mergekit/merged_model/dare_dsc7_gs \
  saves/dsc7/dm/5e6 \
  --output vis/gap_cor_dsc7_{$threshold}_dm --filter_small_gaps --gap_threshold $threshold

python gap_correlation_analyzer.py --base_model deepseek-ai/deepseek-coder-1.3b-instruct \
  --sft_models /path/to/your/project/mergekit/merged_model/linear_dsc13_gsn saves/dsc13/dm/1e5 \
  --output vis/gap_cor_dsc13_{$threshold}_dm --filter_small_gaps --gap_threshold $threshold

