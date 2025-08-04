# %%

import ast, re, textwrap, inspect

import re
from pathlib import Path   # only needed if you want to read from a file

DOC_PATTERN = re.compile(r'/\*\*.*?\*/', re.S)          # non-greedy .*?  + DOTALL
TAG_PATTERN = re.compile(r'^\s*\*?\s*@', re.M)          # lines that begin with @tag

def extract_javadoc_block(code: str) -> str | None:
    """Return the first /** … */ block verbatim (including the fences)."""
    m = DOC_PATTERN.search(code)
    return m.group(0) if m else None

def extract_functional_description(code: str) -> str | None:
    """
    Return only the free-text description (everything up to the first @tag)
    with leading '* ' stripped.
    """
    block = extract_javadoc_block(code)
    if not block:
        return None

    # Strip the opening /** and closing */
    inner = block.lstrip('/').lstrip('*').rstrip('*/')
    lines = [re.sub(r'^\s*\*\s?', '', ln) for ln in inner.splitlines()]

    desc = []
    for ln in lines:
        if TAG_PATTERN.match(ln):      # stop at @param, @return, etc.
            break
        if ln.strip():                 # skip blank lines
            desc.append(ln.strip())
    return ' '.join(desc)              # → single sentence, join however you like

# ----------------- demo -----------------
sample = """
/**
 * Trims each element of the input array and returns a new array with the trimmed elements.
 * If the input array is empty, returns an empty array.
 *
 * @param array The input array of strings to be trimmed.
 * @return A new array with trimmed elements.
 */
public static String[] trimArrayElements(String[] array) {
  if (Objects.isEmpty(array)) {
    return new String[0];
  }
  // ...
}
"""

full_comment = extract_javadoc_block(sample)
print("=== kept /** ... */ ===")
print(full_comment)

only_description = extract_functional_description(sample)
print("\n=== functional description ===")
print(only_description)



# %%





"""


Example
-------
python codesum_codereval.py /dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/CoderEval/CoderEval4Python.json ./data/coder_eval_py.jsonl
"""
import json
import argparse
from pathlib import Path
import re
from typing import Literal

INSTRUCTION = (
    "Write a concise doc comment for the method below. "
    "Return only the comment lines—no additional text."
)



COMMENT_RE  = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.S)

def convert(input_path: Path, output_path: Path) -> None:
    # 1) read source JSON -----------------------------------------------------
    data = json.loads(input_path.read_text(encoding="utf-8"))
    records = data.get("RECORDS", [])

    # 2) stream-write JSONL ----------------------------------------------------
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(records):
            code_output = re.sub(COMMENT_RE, "", rec['code'].strip())
            example = {
                "id": rec.get("_id", idx),
                "instruction": INSTRUCTION,
                # use fenced block so new-lines survive in plain text prompts
                "input": f"```\n{code_output}\n```",
                # prefer the full docstring when present, else fall back
                "output": (rec.get("docstring") or rec.get("human_label", "")).strip(),
            }
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=Path, help="path to source records.json")
    ap.add_argument("output_jsonl", type=Path, help="destination *.jsonl")
    args = ap.parse_args()
    convert(args.input_json, args.output_jsonl)

if __name__ == "__main__":
    main()
