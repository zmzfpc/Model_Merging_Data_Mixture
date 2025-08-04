"""
Convert a code-comment dataset into Alpaca-style Instruct records.

Example:
    python codesum_build_instruct_ct.py \
        --dataset_name google/code_x_glue_ct_code_to_text \
        --lang python java \
        --output_dir ./data/instruct_code_docstring_train/
"""
from datasets import load_dataset, concatenate_datasets, Features, Value
from pathlib import Path
import argparse
import re

COMMENT_RE  = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.S)


def _to_prompt(example, idx):
    code = re.sub(COMMENT_RE, "", example["code"])
    doc = example["docstring"]


    return {
        "id": idx,
        "instruction": (
            "Write a concise doc comment for the function below. "
            "Return only the comment line(s), no additional text."
        ),
        "input": f"```{code}```",
        "output": doc.strip(),
    }


def convert_dataset(dataset_name: str, langs: list[str], output_dir: str):
    splits = ["train","test"]            # CodeXGLUE standard splits:contentReference[oaicite:3]{index=3}
    new_features = Features(
        {
            "id": Value("int32"),
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
        }
    )

    merged = {}                                         
    for split in splits:
        per_lang = []
        for lang in langs:
            ds = load_dataset(dataset_name, name=lang, split=split) 
            # ds = ds.map(_to_prompt, with_indices=True,
                        # features=new_features, desc=f"{lang}-{split}")
            new_ds = ds.map(_to_prompt, with_indices=True)  # ✨ drop “features=” here
            base_cols = ds.column_names 
            new_ds = new_ds.remove_columns(base_cols) 
            per_lang.append(new_ds)
        # stitch python + java together
        merged[split] = concatenate_datasets(per_lang)       

    # # save every split to one Arrow directory
    # Path(output_dir).mkdir(parents=True, exist_ok=True)
    # for split, ds in merged.items():
    #     ds.save_to_disk(Path(output_dir) / split)
    # total = sum(len(ds) for ds in merged.values())
    # print(f"Saved {total:,} examples (Python+Java) to {output_dir}")

    # export each split to JSON-Lines
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total = 0
    for split, ds in merged.items():
        out_path = Path(output_dir) / f"{split}.jsonl"
        ds.to_json(
            path_or_buf=str(out_path),
            lines=True,          # newline-delimited JSON (default)
            orient="records"     # one dict per line
        )
        print(f"Wrote {len(ds):,} examples → {out_path}")
        total += len(ds)
    print(f"Saved {total:,} examples (Python+Java) in JSON-Lines format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        default="google/code_x_glue_ct_code_to_text")
    parser.add_argument("--langs", nargs="+",
                        default=["python", "java"],
                        help="Subset names to merge (default: python java)")
    parser.add_argument("--output_dir", default="instruct_py_java")
    args = parser.parse_args()

    convert_dataset(args.dataset_name, args.langs, args.output_dir)


