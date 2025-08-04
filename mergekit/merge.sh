cd /path/to/your/project/mergekit/

source .venv/bin/activate


export TOKENIZERS_PARALLELISM="false"
export HF_HOME="/path/to/your/project/huggingface"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

base_model="dsc13"

yaml_files=(
    "linear_${base_model}_gs"
    "ties_${base_model}_gs"
    "dare_${base_model}_gs"
    "della_${base_model}_gs"
)


for i in "${!yaml_files[@]}"; do
    
    yaml_file="${yaml_files[$i]}"
    echo "Merging model with YAML file: ${yaml_file}"
    mergekit-yaml my_yaml/${yaml_file}.yml merged_model/${yaml_file} --cuda --trust-remote-code


done