#!/usr/bin/env python
"""
Filter NVIDIA/OpenCodeInstruct:
  • Remove rows with average_test_score == 1
    AND an overall perfect (5.0) LLM-judgement.
  • Write the cleaned split(s) to one JSONL file.

Example:
    python filter_opencodeinstruct.py \
          --outfile data/OpenCodeInstruct_filtered.jsonl
"""
import argparse, json
from pathlib import Path
from statistics import mean
from datasets import load_dataset, concatenate_datasets, DatasetDict

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test", "all"],
                   help="Dataset split(s) to keep (default: train).")
    p.add_argument("--outfile", type=Path,
                   default="Kodcode_filtered.jsonl",
                   help="Destination JSONL path.")
    return p.parse_args()

def overall_llm_score(llm_judgement):
    """
    Accepts either a dict or the JSON-encoded string stored in the column.
    Returns the arithmetic mean of all category scores, or None if unavailable.
    """
    if llm_judgement is None:
        return None
    if isinstance(llm_judgement, str):
        llm_judgement = json.loads(llm_judgement)
    scores = [cat.get("score") for cat in llm_judgement.values()
              if isinstance(cat, dict) and "score" in cat]
    return mean(scores) if scores else None

def should_drop(example):

    # if float(example["average_test_score"]) != 1:
    #     print(example["average_test_score"])
    # if overall_llm_score(example["llm_judgement"]) != 5:
    #     print(overall_llm_score(example["llm_judgement"]))

    return (float(example["average_test_score"]) == 1 and overall_llm_score(example["llm_judgement"]) == 5)

def is_rl_ready(example):
    return (example["test_info"] is not None
            and len(example["test_info"]) == 1
            and example["style"] == "instruct"
1)


def main():

            
    args = get_args()


    if args.split == "all":
        ds_dict: DatasetDict = load_dataset("KodCode/KodCode-V1-SFT-R1")      
        dataset = concatenate_datasets(list(ds_dict.values()))
    else:
        dataset = load_dataset("KodCode/KodCode-V1-SFT-R1", split=args.split)

    print(f"Rows before filtering: {len(dataset):,}")


    cleaned = dataset.filter(is_rl_ready, num_proc=8)             
    print(f"Rows after  filtering: {len(cleaned):,}")
    cleaned.to_json(str(args.outfile))                                    
    print("✓ Saved →", args.outfile.resolve())

if __name__ == "__main__":
    main()
