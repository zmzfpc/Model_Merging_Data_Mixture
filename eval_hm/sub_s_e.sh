

eval_sh=(

  # "cttest_qwc15"
  # "cttest_gnc8"
  # "cttest_dsc7"
  # "cttest_qwc7"
  # "cttest_dsc13"
  # "cttest_gnc3"
  # "cttest_sft_gnc3"
  # "cttest_sft_qwc15"
  # "cttest_sft_qwc15_4o"

  # "cttest_sft_qwc7"
  # "cttest_sft_qwc7_4o"
  #  "cttest_dsc13_4o"
  # "cttest_sft_dsc7"
  # "cttest_sft_gnc3"

  # "cttest_sft_gnc8"
  # "cttest_sft_dsc13"
  # "cttest_dm_dsc13"
  # "cttest_dm_dsc7"
  # "cttest_dm_qwc2515"
  # "cttest_dm_qwc257"
  "cttest_dm_gnc8"
  # "cttest_dm_gnc3"
)

# for i in "${!eval_sh[@]}"; do
    
#     sh_name="${eval_sh[$i]}"
#     echo "Submitting job for: $sh_name"

#     bsub -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB -J eval_g_${sh_name}_a -M 50G -n 32 \
#     -o /path/to/your/project/eval_hm/job_out/s_eval/g_${sh_name}_a100.out \
#     /path/to/your/project/eval_hm/job_sh/s_eval/evalcodesum_${sh_name}.sh

# done

# Which jobs is faster
for i in "${!eval_sh[@]}"; do
    
    sh_name="${eval_sh[$i]}"
    echo "Submitting job for: $sh_name"

    bsub -gpu num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3 -J eval_g_${sh_name}_h -M 50G -n 32 \
    -o /path/to/your/project/eval_hm/job_out/s_eval/g_${sh_name}_h100.out \
    /path/to/your/project/eval_hm/job_sh/s_eval/evalcodesum_${sh_name}.sh

done