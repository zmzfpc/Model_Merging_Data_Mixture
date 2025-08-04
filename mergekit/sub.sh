
eval_sh=(
    "dare_dsc13_d0_3_0_7_ties"
    "dare_dsc7_d0_3_0_7_ties"
    "dare_qwc7_d0_3_0_7_ties"
    "dare_qwc15_d0_3_0_7_ties"
    "dare_dsc13_d0_3_0_7"
    "dare_dsc7_d0_3_0_7"
    "dare_qwc7_d0_3_0_7"
    "dare_qwc15_d0_3_0_7"
    # "linear_dsc13"
    # "linear_dsc7"
    # "linear_qwc7"
    # "linear_qwc15"
    # "della_dsc13"
    # "della_dsc7"
    # "della_qwc7"
    # "della_qwc15"
    # "ties_dsc13"
    # "ties_dsc7"
    # "ties_qwc7"
    # "ties_qwc15"
)

# for i in "${!eval_sh[@]}"; do
    
#     sh_name="${eval_sh[$i]}"
#     echo "Submitting job for: $sh_name"

#     bsub -gpu num=1:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB -J eval_g_${sh_name}_a -M 50G -n 32 \
#     -o /path/to/your/project/eval_hm/job_out/s_eval/g_${sh_name}_a100.out \
#     /path/to/your/project/eval_hm/job_sh/s_eval/evalcodesum_${sh_name}.sh

# done

for i in "${!eval_sh[@]}"; do
    
    sh_name="${eval_sh[$i]}"
    echo "Submitting job for: $sh_name"

    bsub -gpu num=1:mode=exclusive_process:gmodel=NVIDIAH10080GBHBM3 -J eval_g_${sh_name}_h -M 50G -n 32 \
    -o /path/to/your/project/mergekit/job_out/m_${sh_name}_m_h.out \
    /path/to/your/project/mergekit/job_sh/merge_${sh_name}.sh

done
