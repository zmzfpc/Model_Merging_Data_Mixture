cd /path/to/your/project/mergekit/

source .venv/bin/activate


export TOKENIZERS_PARALLELISM="false"
export HF_ALLOW_CODE_EVAL="1"
export HF_HOME="/path/to/your/project/huggingface"
export HF_TOKEN=""
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

wandb login 


base_model="qw2"

yaml_files=(
    # "ties_${base_model}_gs"
    "evl_${base_model}_gs_m3"
    # "della_${base_model}_gs"
)


for i in "${!yaml_files[@]}"; do
    
    yaml_file="${yaml_files[$i]}"
    echo "Merging model with YAML file: ${yaml_file}"
    # mergekit-yaml my_yaml/${yaml_file}.yml merged_model/${yaml_file} --cuda --trust-remote-code
    mergekit-evolve --storage-path merged_model/${yaml_file} my_yaml/${yaml_file}.yml --vllm

done