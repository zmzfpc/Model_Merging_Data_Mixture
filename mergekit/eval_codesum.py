#!/usr/bin/env python
"""
Evaluate code-docstring generation with SIDE + classic metrics.
Supports HuggingFace *or* vLLM back-ends, BF16, batching, and result dump.

Example
-------
python eval_codesum.py \
    --model  /path/to/your/project/LLaMA-Factory/saves/qwen2515/full/sft_ct \
    --data  /path/to/your/project/LLaMA-Factory/data/instruct_code_docstring_train/test.jsonl \
    --side   /path/to/your/project/LLaMA-Factory/saves/qwen2515/full/sft_ct \
    --batch_size 64 --bf16 --vllm \
    --save_path ./data/instruct_code_docstring_train/sftqw_preds.jsonl
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path
from typing import List, Dict
import ast, textwrap
import torch, torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu, evaluate
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer as STTokenizer
import nltk
from vllm import LLM, SamplingParams
# try:
#     from vllm import LLM, SamplingParams            # noqa
# except ImportError:
#     LLM = None                                      # resolved at runtime

# ─────────────────── SIDE helpers ─────────────────────────────────

# ─────────────────── Normaliser ──────────────────────────────────
import re, html

_C_TOK_FENCE   = re.compile(r"<think>.*?</think>", re.S)      # remove code fences
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
    """Return a language-agnostic, comment-free one-liner."""
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

def mean_pool(x, m):     # identical to official MPNet SIDE example
    m = m.unsqueeze(-1).expand_as(x).float()
    return torch.nn.functional.normalize((x * m).sum(1) / m.sum(1).clamp(min=1e-9),
                                         p=2, dim=1)

def load_side_encoder(ckpt: Path, device):
    tok = STTokenizer.from_pretrained(ckpt)
    mdl = AutoModel.from_pretrained(ckpt).to(device).eval()
    return tok, mdl

def side_sim(code, summ, tok, mdl, device):
    with torch.no_grad():
        enc = tok([code, summ], padding=True, truncation=True,
                  return_tensors="pt").to(device)
        embs = mean_pool(mdl(**enc).last_hidden_state, enc["attention_mask"])
        return util.cos_sim(embs[0], embs[1]).item()

# ─────────────────── Prompt builder ───────────────────────────────
def chat_prompt(tok, instr: str, code: str, model_name: str) -> str:
    if getattr(tok, "chat_template", None):
        # if  "dsc" not in model_name:
        #     print("Using chat template for tokenization")
        #     return tok.apply_chat_template(
        #         [{"role": "system", "content": instr},
        #          {"role": "user",   "content": code}],
        #         add_generation_prompt=True,
        #         tokenize=False)
        # else:
        #     return tok.apply_chat_template(
        #          [{"role": "user",   "content": f"{instr}\n{code}"}],
        #         add_generation_prompt=True,
        #         tokenize=False)
        return tok.apply_chat_template(
                 [{"role": "user",   "content": f"{instr}\n{code}"}],
                add_generation_prompt=True,
                tokenize=False)
            
    return f"{instr}\n{code}"

# ─────────────────── HF generation ────────────────────────────────
def hf_generate(model, tok, samples, dev, bs, max_new=512,
                temp=0.0, bf16=False):

    preds = []
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if (bf16 and dev.type == "cuda") else torch.no_grad())

    for i in tqdm(range(0, len(samples), bs), desc="HF-generate"):
        batch = samples[i:i+bs]
        prompts = [chat_prompt(tok, s["instruction"], s["input"]) for s in batch]
         # debug: show first prompt
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True).to(dev)
        p_lens = enc["attention_mask"].sum(1)

        with ctx:
            gen = model.generate(**enc,
                                 max_new_tokens=max_new,
                                 do_sample=False,
                                 temperature=temp)

        for seq, pl in zip(gen, p_lens):
            ans = seq[int(pl):]
            preds.append(tok.decode(ans, skip_special_tokens=True).strip())
    return preds

# ─────────────────── vLLM generation ──────────────────────────────
def vllm_generate(engine, tok, samples, max_new=2048, temp=0.0, model_name=None):
    print(model_name)
    prompts = [chat_prompt(tok, s["instruction"], s["input"], model_name) for s in samples]
    print(prompts[:1])
    params  = SamplingParams(max_tokens=max_new, temperature=temp)   # vLLM API
    outs = engine.generate(prompts, params)                          # batched :contentReference[oaicite:1]{index=1}
    return [o.outputs[0].text.strip() for o in outs]

# ─────────────────── Metrics ──────────────────────────────────────
def evaluate_reg(ref, hyp):
    import tempfile
    import os
    
    bleu  = [sentence_bleu([r.split()], h.split()) for r, h in zip(ref, hyp)]
    chrf  = [sacrebleu.sentence_chrf(h, [r], word_order=2).score/100 for r, h in zip(ref, hyp)]

    # Create unique experiment names to avoid parallel conflicts
    pid = os.getpid()
    import time
    timestamp = int(time.time() * 1000000)  # microsecond precision
    
    # Use unique experiment names for parallel safety
    rouge_experiment = f"rouge_exp_{pid}_{timestamp}"
    meteor_experiment = f"meteor_exp_{pid}_{timestamp}"
    
    try:
        rouge = evaluate.load("rouge", experiment_id=rouge_experiment).compute(predictions=hyp, references=ref)
        meteor= evaluate.load("meteor", experiment_id=meteor_experiment).compute(predictions=hyp, references=ref)
        meteor_score = meteor["meteor"]
    except (FileNotFoundError, OSError) as e:
        print(f"Warning: Error computing METEOR metric ({e}), using fallback implementation")
        # Fallback: use nltk METEOR if available, otherwise skip
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score as nltk_meteor
            meteor_scores = [nltk_meteor([r.split()], h.split()) for r, h in zip(ref, hyp)]
            meteor_score = st.mean(meteor_scores)
        except ImportError:
            print("Warning: NLTK METEOR not available, setting METEOR score to 0.0")
            meteor_score = 0.0
        
        # Try ROUGE again with different approach
        try:
            rouge = evaluate.load("rouge", experiment_id=rouge_experiment).compute(predictions=hyp, references=ref)
        except (FileNotFoundError, OSError):
            print("Warning: Error computing ROUGE metric, setting to 0.0")
            rouge = {"rougeL": 0.0}

    return {"BLEU-4": st.mean(bleu), "chrF++": st.mean(chrf),
            "ROUGE-L": rouge["rougeL"], "METEOR": meteor_score
            }

def evaluate_side(ref, hyp, codes, side_ckpt, dev):
    s_tok, s_mdl = load_side_encoder(side_ckpt, dev)
    side  = [side_sim(c, h, s_tok, s_mdl, dev)
             for c, h in tqdm(zip(codes, hyp), total=len(hyp), desc="SIDE")]
    
    side_gt  = [side_sim(c, r, s_tok, s_mdl, dev)
             for c, r in tqdm(zip(codes, ref), total=len(ref), desc="SIDEGT")]
    
    return {"SIDE": st.mean(side), "SIDE_gt": st.mean(side_gt)}

def load_jsonl(path: Path) -> List[Dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]

def dump_jsonl(path: Path, records: List[Dict]):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")




def main():
    import os
    import tempfile
    
    # Setup unique cache directories for parallel execution
    pid = os.getpid()
    import time
    timestamp = int(time.time() * 1000000)
    unique_cache_dir = f"/tmp/eval_cache_{pid}_{timestamp}"
    os.makedirs(unique_cache_dir, exist_ok=True)
    
    # Set environment variables to use unique cache directories
    os.environ["HF_DATASETS_CACHE"] = unique_cache_dir
    os.environ["HF_METRICS_CACHE"] = unique_cache_dir
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--py_clean", action="store_true")
    ap.add_argument("--java_clean", action="store_true")
    ap.add_argument("--save_path", type=str, help="Write preds to this file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    print(args.model.lower())
    print(args.data)
    samples = load_jsonl(Path(args.data))
    refs, codes = [s["output"] for s in samples], [s["input"] for s in samples]

    # tokenizer first (shared by both paths)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    if args.vllm:                                             # ─ vLLM branch
        if LLM is None:
            raise ImportError("pip install vllm-nightly first!")
        engine = LLM(model=args.model,
                     dtype="bfloat16" if args.bf16 else "float16",
                     trust_remote_code=True)          # BF16 kernel support :contentReference[oaicite:2]{index=2}
        preds = vllm_generate(engine, tok, samples, model_name=args.model.lower())
    else:                                                     # ─ HF fallback
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            trust_remote_code=True).to(dev).eval()
        preds = hf_generate(mdl, tok, samples, dev,
                            args.batch_size, bf16=args.bf16)
        # Clean up HF model
        del mdl
    
    # Clean up GPU memory and any remaining objects
    try:
        del engine
    except:
        pass
    torch.cuda.empty_cache()

    # print(refs[:3], preds[:3])  # debug: show first 3 refs and preds
    # print(normalise_docstring(refs[0]))
    # refs  = [normalise_docstring(r) for r in refs]

    # preds = [normalise_docstring(p) for p in preds]

    

    scores = evaluate_reg(refs, preds)
    

    # ─ save predictions if requested ─
    if args.save_path:
        dump_jsonl(Path(args.save_path),
                   [{"code": c, "reference": r, "prediction": p}
                    for c, r, p in zip(codes, refs, preds)])
        print(f"\nPredictions written to {args.save_path}")

    # ─ report metrics ─
    
    print("\n=== Aggregate Scores ===")
    for k, v in scores.items():
        print(f"{k:8s}: {v:.4f}")
    
    nrefs  = [normalise_docstring(r) for r in refs]
    npreds = [normalise_docstring(p) for p in preds]
    scores = evaluate_reg(nrefs, npreds)
    print("\n=== Norm Scores ===")
    for k, v in scores.items():
        print(f"{k:8s}: {v:.4f}")
    
    
    if args.java_clean and "java" in args.data:
        print("java_clean")
        preds = [extract_javadoc_block(p) for p in preds]
    else:

        preds = [extract_javadoc_block(p) if "/**" in p and "*/" in p else p for p in preds]

    

    # if "qwen2.5" in args.model.lower() or (args.py_clean and "py" in args.data):

    #     print("Using Qwen2.5 model, removing output code")
    #     preds = [remove_output_code(p) for p in preds]
    
    preds = [remove_output_code(p) for p in preds]
    
    refs  = [normalise_docstring(r) for r in refs]
    preds = [normalise_docstring(p) for p in preds]
    scores = evaluate_reg(refs, preds)
    print("\n=== Norm/Extract Scores ===")
    for k, v in scores.items():
        print(f"{k:8s}: {v:.4f}")
    # print(args.side)
    # side_scores = evaluate_side(refs, preds, codes, Path(args.side), dev)

    # for k, v in side_scores.items():
    #     print(f"{k:8s}: {v:.4f}")
    
    # Cleanup temporary cache directory
    try:
        import shutil
        shutil.rmtree(unique_cache_dir, ignore_errors=True)
    except:
        pass  # Ignore cleanup errors

if __name__ == "__main__":
    main()
