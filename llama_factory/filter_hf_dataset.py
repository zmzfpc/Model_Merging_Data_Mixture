"""
filter_hf_dataset.py

Example
-------
python filter_hf_dataset.py \
    --name bigcode/commitpackft \
    --subset python,java \
    --out_dir ./data/filtered_commitpackft
"""

import argparse
from datasets import load_dataset, concatenate_datasets

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True,
                   help="HF dataset name, e.g. codeparrot/github-code")
    # allow many --subset flags; each may itself contain commas
    p.add_argument("--subset", action="append", default=[],
                   help="Config name(s); repeat flag or use comma-sep, e.g. "
                        "--subset python --subset java  OR  --subset python,java")
    p.add_argument("--split", default="train")
    p.add_argument("--filter",  default=None,
                   help='Python expression evaluated for each row as "example"')
    p.add_argument("--columns", default=None)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()

def main():
    args = parse_args()

    # Flatten comma-separated tokens and remove empties
    subsets = [s.strip() for item in args.subset for s in item.split(",") if s.strip()]
    # If user gave none, we treat it like a single "None" config
    subsets = subsets or [None]

    # Load each subset → filtered dataset → stash in list
    all_parts = []
    for cfg in subsets:
        ds = load_dataset(args.name, cfg, split=args.split, trust_remote_code=True)
        if args.columns:
            keep = {c.strip() for c in args.columns.split(",")}
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        if args.filter:
            predicate = compile(args.filter, "<string>", "eval")
            ds = ds.filter(lambda example: eval(predicate, {}, {"example": example}),
                       desc=f"Filtering [{cfg or 'default'}]")
        all_parts.append(ds)

    # Merge the filtered splits (concatenate preserves order)
    merged = concatenate_datasets(all_parts) if len(all_parts) > 1 else all_parts[0]
    print(f"Total rows after merge & filter: {len(merged):,}")

    merged.save_to_disk(args.out_dir)
    print(f"Saved  {args.out_dir}")

if __name__ == "__main__":
    main()
