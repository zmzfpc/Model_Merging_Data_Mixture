

eval_sh=(
  # "sft_gn8_oc"
  # "sft_dsc67_kod_90k"
  # "sft_dsc13_kod_1e5"
  # "sft_dsc13_kod_5e6"
  
  # "sft_qwen2515_kod"
  # "sft_qwenc257_5e6"
  # "sft_qwenc257_1e7"
  # "sft_qwenc2515_kod_1e5"
  # "sft_qwenc2515_kod_5e6"
  # "sft_dsc7_kod"
  # "dsc7"
  # "sft_gnc3_5e5"
  # "sft_gnc3_1e5" 
  # "sft_gnc3_1e6"
  # "sft_gnc8_1e6"
  # "sft_gnc8_5e6"
  # "sft_lm38_kod"
  # "sft_dsc13_ct"
  # "sft_qwenc2515_ct"
  # "dm_dsc13"
  # "dm_dsc7"
  # "dm_qwc2515"
  "dm_qwc257"

)

for i in "${!eval_sh[@]}"; do
    
    sh_name="${eval_sh[$i]}"
    echo "Submitting job for: $sh_name"

    bsub -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB -J eval_g_${sh_name}_a -M 50G -n 32 \
    -o /dccstor/unified-trans/model_merging/granite33_2/eval_hm/job_out/g_eval/g_${sh_name}_a100.out \
    /dccstor/unified-trans/model_merging/granite33_2/eval_hm/job_sh/g_eval/evalplus_${sh_name}.sh

done

# for i in "${!eval_sh[@]}"; do
    
#     sh_name="${eval_sh[$i]}"
#     echo "Submitting job for: $sh_name"

#     bsub -gpu num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3 -J eval_g_${sh_name}_h -M 50G -n 32 \
#     -o /dccstor/unified-trans/model_merging/granite33_2/eval_hm/job_out/g_eval/g_${sh_name}_h100.out \
#     /dccstor/unified-trans/model_merging/granite33_2/eval_hm/job_sh/g_eval/evalplus_${sh_name}.sh

# done