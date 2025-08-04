#!/usr/bin/env python3
"""
json_to_markdowns.py
--------------------
Convert code-summary records into individual Markdown “chat” files.

Each record must contain:
   "code"        code snippet (optionally wrapped in ``` fences)
   "reference"   ground-truth summary / docstring
   "prediction"  model-generated summary

Usage
=====
# single JSON or a JSON list
python json_to_markdowns.py data.json out_dir

# JSON-Lines file
python json_to_markdowns.py ./data/instruct_code_docstring_train/sftgm_preds_coder_py.jsonl code_sum/codereval_py_sftgm/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/qw_text_preds_coder_py.jsonl code_sum/codereval_py_qw_c/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/sftdsc_preds_coder_py.jsonl code_sum/codereval_py_sftdsc/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/sftdsc_preds_ct.jsonl code_sum/ct_sftdsc/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/dsc_preds_ct.jsonl code_sum/ct_dsc/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/qw_preds_coder_java.jsonl code_sum/codereval_java_qw/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/qw3_preds_ct.jsonl code_sum/qw3_preds_ct.jsonl/ --jsonl

python json_to_markdowns.py ./data/instruct_code_docstring_train/sft_preds_coder_py.jsonl code_sum/codereval_py_sft/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/qwc_preds_coder_java_n.jsonl code_sum/codereval_java_qwc_n/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/sftqw25_preds_coder_py.jsonl code_sum/codereval_py_sftqw25/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/qw25_preds_coder_py.jsonl code_sum/codereval_py_qw25/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/sftdsc_preds_coder_java.jsonl code_sum/codereval_java_sftdsc/ --jsonl
python json_to_markdowns.py ./data/instruct_code_docstring_train/dsc_preds_coder_java.jsonl code_sum/codereval_java_dsc/ --jsonl
Options
-------
  --hide-reference     Omit the ground-truth summary from the output
  --prefix EXPR        Filename prefix (default: "example_")
  --pad N              Zero-pad example numbers to N digits (default: 3)
"""
import argparse, json, re, sys
from pathlib import Path
from typing import Iterable, Dict, Any

def strip_triple_backticks(block: str) -> str:
    block = block.strip()
    if block.startswith("```") and block.endswith("```"):
        return re.sub(r"^```[^\n]*\n|\n```$", "", block, flags=re.S)
    return block


# ─────────────────── Normaliser ──────────────────────────────────
import re, html

_C_TOK_FENCE   = re.compile(r"<think>.*?</think>", re.S)    # remove code fences
_C_TRIPLE_QUOT = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.S)
_C_JDOC        = re.compile(r"/\*\*|/\*|\*/")        # /**  */, /*  */ → ""
_C_STAR        = re.compile(r"^\s*\* ?", re.M)       # leading '*' on each line
_C_WS          = re.compile(r"\s+")                  # collapse whitespace

DOC_PATTERN = re.compile(r'/\*\*.*?\*/', re.S) 
def extract_javadoc_block(code: str) -> str | None:
    m = DOC_PATTERN.search(code)
    return m.group(0) if m else ""

def remove_output_code(text: str) -> str:
    if '"""' in text:  # if triple quotes are present
        try:
            doc = re.search(r'"""(.*?)"""', text, re.DOTALL).group(1)
        except:
            doc = ""

    else:
        doc = text
    # tree = ast.parse(textwrap.dedent(text))                    
    # func = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    # return ast.get_docstring(func)    
    # print(doc)
    return doc


def normalise_docstring(text: str) -> str:
    text = html.unescape(text)               # in case backticks were HTML-escaped
    text = _C_TOK_FENCE.sub("", text)
    # text = _C_TRIPLE_QUOT.sub(" ", text)
    text = text.replace("```", "")
    text = text.replace('"""', "")
    text = text.replace("<code>", "")
    text = text.replace("<\code>", "")
    text = text.replace("<p>", "")
    text = text.replace("<\p>", "")
    text = _C_JDOC.sub(" ", text)
    text = _C_STAR.sub(" ", text)
    text = text.replace("@param", ":param")  # unify param tag
    text = _C_WS.sub(" ", text)              # squash whitespace
    return text.strip().rstrip(".")         

def record_to_markdown(rec: Dict[str, Any], show_reference: bool = True) -> str:
    """Return a Markdown conversation for one record."""
    code_block = re.sub(_C_TRIPLE_QUOT, "",strip_triple_backticks(rec["code"]))
    if "**" in rec["prediction"].strip():
        extract_des = remove_output_code(extract_javadoc_block(rec["prediction"].strip()))
    else:
        extract_des = remove_output_code(rec["prediction"].strip())

    parts = [
        "**User**",
        "Write a concise doc comment for the function below. "
        "Return only the comment line(s), no additional text.",
        f"```\n{code_block}\n```",
        "",
        "**Assistant (prediction)**",
        rec["prediction"].strip(),
        "\n\nNormalised:",
        normalise_docstring(extract_des),

    ]
    if show_reference:
        parts += ["", "**Assistant (reference)**", normalise_docstring(rec["reference"].strip())]
    return "\n".join(parts) + "\n"

def load_records(path: Path, jsonl: bool) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        if jsonl:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:
            data = json.load(f)
            yield from (data if isinstance(data, list) else [data])

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input",  type=Path, help="JSON or JSONL file")
    p.add_argument("output", type=Path, help="Output folder for .md files")
    p.add_argument("--jsonl", action="store_true", help="Treat input as JSON-Lines")
    p.add_argument("--hide-reference", action="store_true")
    p.add_argument("--prefix", default="example_", help="Filename prefix")
    p.add_argument("--pad", type=int, default=3, help="Zero-pad width")
    args = p.parse_args()

    records = list(load_records(args.input, args.jsonl))
    if not records:
        sys.exit("No valid records found.")

    args.output.mkdir(parents=True, exist_ok=True)

    for idx, rec in enumerate(records, 1):
        md_text = record_to_markdown(rec, show_reference=not args.hide_reference)
        fname   = f"{args.prefix}{idx:0{args.pad}d}.md"
        (args.output / fname).write_text(md_text, encoding="utf-8")

    print(f"Wrote {len(records)} markdown file(s) to {args.output.resolve()}")

if __name__ == "__main__":
    main()


# def getmetadata(self, key=None):
#             "\"\"Get the metadata that applies to this element, automatically inherited from parent elements\"\"\"\n       
#     if self.metadata:          
#         d =  self.doc.submetadata[self.metadata]       
#     elif self.parent:
#         d =  self.parent.getmetadata()       
#     elif self.doc:            
#         d =  self.doc.metadata       
#     else:            
#         return None        
#     if key:            
#         return d[key]       
#     else:           
#         return d
