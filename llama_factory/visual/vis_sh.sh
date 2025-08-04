# python visualize_model_diff.py   \
#   --model1 saves/qwen2515c15/best/sft_4o_sol_1e-5  \
#    --model2 saves/qwen2515c15/best/sft_ct_5e6  \
#    --output enhance_results_qwc15



# python visualize_model_diff.py   \
#   --model1 saves/qwen2515c15/best/sft_4o_sol_1e-5  \
#    --model2 saves/qwen25c15/dm/1e5  \
#    --output enhance_results_qwc15_dm_g

# python visualize_model_diff.py   \
#   --model1 saves/qwen25c15/dm/1e5   \
#    --model2 saves/qwen2515c15/best/sft_ct_5e6  \
#    --output enhance_results_qwc15_dm_s

# python visualize_model_diff.py   \
#   --model1 saves/qwen25c7/best/sft_4o_sol_5e6  \
#    --model2 saves/qwen25c7/best/sft_ct_1e6  \
#    --output enhance_results_qwc7

# python visualize_model_diff.py   \
#   --model1 saves/qwen25c7/best/sft_4o_sol_5e6  \
#    --model2 saves/qwen25c7/dm/5e6  \
#    --output enhance_results_qwc7_dm_g

# python visualize_model_diff.py   \
#   --model1 saves/qwen25c7/dm/5e6   \
#    --model2 saves/qwen25c7/best/sft_ct_1e6  \
#    --output enhance_results_qwc7_dm_s

# python visualize_model_diff.py   \
#   --model1 saves/dsc13/best/sft_4o_sol_1e5  \
#    --model2 saves/dsc13/best/sft_ct_5e6  \
#    --output enhance_results_dsc13 


# python visualize_model_diff.py   \
#   --model1 saves/dsc13/best/sft_4o_sol_1e5  \
#    --model2 saves/dsc13/dm/1e5  \
#    --output enhance_results_dsc13_dm_g 

# python visualize_model_diff.py   \
#   --model1 saves/dsc13/dm/1e5   \
#    --model2 saves/dsc13/best/sft_ct_5e6  \
#    --output enhance_results_dsc13_dm_s




# python visualize_model_diff.py   \
#   --model1 saves/dsc7/best/sft_4o_sol_5e6  \
#    --model2 saves/dsc7/best/sft_ct_1e6  \
#    --output enhance_results_dsc7 

# python visualize_model_diff.py   \
#   --model1 saves/dsc7/best/sft_4o_sol_5e6  \
#    --model2 saves/dsc7/dm/5e6  \
#    --output enhance_results_dsc7_dm_g

# python visualize_model_diff.py   \
#   --model1 saves/dsc7/dm/5e6   \
#    --model2 saves/dsc7/best/sft_ct_1e6  \
#    --output enhance_results_dsc7_dm_s

export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"

source /dccstor/unified-trans/model_merging/miniconda3/etc/profile.d/conda.sh 

conda activate mergel 


python comprehensive_plotter.py --data_file \
  /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/vis/sft_analysis_dsc13/layer_summary.csv \
  --output_dir my_plots/dsc13

python comprehensive_plotter.py --data_file \
  /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/vis/sft_analysis_dsc7/layer_summary.csv \
  --output_dir my_plots/dsc7

python comprehensive_plotter.py --data_file \
  /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/vis/sft_analysis_qwc7/layer_summary.csv \
  --output_dir my_plots/qwc7

python comprehensive_plotter.py --data_file \
  /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/vis/sft_analysis_qwc15/layer_summary.csv \
  --output_dir my_plots/qwc15

# python multi_sft_analyzer.py --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
#   --sft_models saves/qwen25c7/best/sft_ct_1e6 saves/qwen25c7/best/sft_4o_sol_5e6 saves/qwen25c7/dm/5e6 \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/linear_qwc7_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/ties_qwc7_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/dare_qwc7_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/della_qwc7_gs \
#   --output vis/sft_analysis_qwc7


# python multi_sft_analyzer.py --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
#   --sft_models saves/qwen2515c15/best/sft_ct_5e6 \
#   saves/qwen2515c15/best/sft_4o_sol_1e-5 \
#   saves/qwen25c15/dm/1e5 \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/linear_qwc15_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/ties_qwc15_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/dare_qwc15_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/della_qwc15_gs \
#   --output vis/sft_analysis_qwc15


# python multi_sft_analyzer.py --base_model deepseek-ai/deepseek-coder-1.3b-instruct \
#   --sft_models saves/dsc13/best/sft_ct_5e6 saves/dsc13/best/sft_4o_sol_1e5 saves/dsc13/dm/1e5 \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/linear_dsc13_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/ties_dsc13_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/dare_dsc13_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/della_dsc13_gs \
#   --output vis/sft_analysis_dsc13

# python multi_sft_analyzer.py --base_model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
#   --sft_models saves/dsc7/best/sft_ct_1e6 saves/dsc7/best/sft_4o_sol_5e6 saves/dsc7/dm/5e6 \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/linear_dsc7_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/ties_dsc7_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/dare_dsc7_gs \
#   /dccstor/unified-trans/model_merging/granite33_2/mergekit/merged_model/della_dsc7_gs \
#   --output vis/sft_analysis_dsc7




