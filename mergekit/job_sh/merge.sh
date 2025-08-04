cd /dccstor/unified-trans/model_merging/granite33_2/mergekit/




export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

base_model="qw2"

yaml_files=(
    # "ties_${base_model}_gs"
    "dare_${base_model}_gs"
    # "della_${base_model}_gs"
)


for i in "${!yaml_files[@]}"; do
    
    yaml_file="${yaml_files[$i]}"
    echo "Merging model with YAML file: ${yaml_file}"

    source /dccstor/unified-trans/model_merging/granite33_2/mergekit/.venv/bin/activate
    mergekit-yaml my_yaml/${yaml_file}.yml \
        merged_model/${yaml_file} --cuda --trust-remote-code

    source /dccstor/unified-trans/model_merging/granite33_2/evalplus/.venv/bin/activate

    evalplus.evaluate --model  merged_model/${yaml_file} --backend vllm --dataset humaneval --greedy

    echo "Evaluation on HumanEval complete."

    evalplus.evaluate --model  merged_model/${yaml_file} --backend vllm --dataset mbpp --greedy

    echo "Evaluation complete."

done