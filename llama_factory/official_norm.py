import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # or "cpu"
# model_path = "ibm-granite/granite-3b-code-instruct-128k"
model_path = "ibm-granite/granite-8b-code-instruct-4k"
# model_path = "ibm-granite/granite-3.3-2b-instruct"
# model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Method 1: Add special tokens during tokenizer initialization
# You can add custom special tokens like this:
custom_special_tokens = {
    "eos_token": "<|EOT|>",
    "additional_special_tokens": ["<|EOT|>"]
}

# Add the special tokens to the tokenizer
num_added_tokens = tokenizer.add_special_tokens(custom_special_tokens)
print(f"Added {num_added_tokens} new special tokens")
print(f"Set EOS token to: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# Method 2: Add individual special tokens
# tokenizer.add_tokens(["<NEW_TOKEN>"], special_tokens=True)

# Method 3: Set specific special tokens if they don't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token")

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

# IMPORTANT: Resize model embeddings to accommodate new tokens
if num_added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens")
model.eval()

# Print basic special tokens
print("Basic special tokens:")
print(f"BOS token: {tokenizer.bos_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"PAD token: {tokenizer.pad_token}")
print(f"UNK token: {tokenizer.unk_token}")
print(f"SEP token: {tokenizer.sep_token}")
print(f"MASK token: {tokenizer.mask_token}")
print(f"CLS token: {tokenizer.cls_token}")

# Print all special tokens
print("\nAll special tokens:")
print(f"Special tokens map: {tokenizer.special_tokens_map}")
print(f"All special tokens: {tokenizer.all_special_tokens}")
print(f"Special tokens map extended: {tokenizer.all_special_tokens_extended}")

# Print special token IDs
print("\nSpecial token IDs:")
print(f"BOS token ID: {tokenizer.bos_token_id}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"PAD token ID: {tokenizer.pad_token_id}")
print(f"UNK token ID: {tokenizer.unk_token_id}")

# Print additional special tokens if they exist
print(f"\nAdditional special tokens: {tokenizer.additional_special_tokens}")
print(f"Vocab size: {tokenizer.vocab_size}")
print()

# Optional: Save the updated tokenizer with new special tokens
tokenizer.save_pretrained("./updated_model/gnc8")
model.save_pretrained("./updated_model/gnc8")
print("Saved updated tokenizer and model to './updated_model/gnc8'")
# To use the new special tokens in text:
example_text = "This is a <CUSTOM> token and a <SPECIAL> token."
tokens = tokenizer.tokenize(example_text)
print(f"Tokenized example: {tokens}")
print(f"Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}")
print()
# change input text as desired
chat = [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Write a code to find the maximum value in a list of numbers." },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(chat)