cd /dccstor/unified-trans/model_merging/granite33_2/mergekit/

source .venv/bin/activate

export HF_ALLOW_CODE_EVAL="1"
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/dccstor/unified-trans/model_merging/granite33_2/huggingface"
export HF_TOKEN="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"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login 077132617119d77cc108c66d1da080152704ee04


base_model="qw2"

yaml_files=(
    # "ties_${base_model}_gs"
    "evl_${base_model}_gs"
    # "della_${base_model}_gs"
)


for i in "${!yaml_files[@]}"; do
    
    yaml_file="${yaml_files[$i]}"
    echo "Merging model with YAML file: ${yaml_file}"
    # mergekit-yaml my_yaml/${yaml_file}.yml merged_model/${yaml_file} --cuda --trust-remote-code
    mergekit-evolve --storage-path merged_model/${yaml_file} my_yaml/${yaml_file}.yml

done