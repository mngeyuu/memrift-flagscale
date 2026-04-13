#!/usr/bin/env python3
"""MemRift metrics matrix: LoRA PPL/BoolQ/load + disk ratio -> JSONL. See examples/memrift/METRICS.md."""
from __future__ import annotations
import argparse, gc, json, math, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@dataclass
class ModelSpec:
    key: str
    name: str
    hf_path: str
    comp_dir: Path

def default_models():
    r = REPO_ROOT
    return {
        "tinyllama-1.1b": ModelSpec("tinyllama-1.1b", "TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", r / "memrift_weights" / "tinyllama_1b_level18"),
        "llama-3.2-3b": ModelSpec("llama-3.2-3b", "Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", r / "memrift_weights" / "llama32_3b_level18"),
        "mistral-7b": ModelSpec("mistral-7b", "Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", r / "memrift_weights" / "mistral_7b_level18"),
        "llama-3.1-8b": ModelSpec("llama-3.1-8b", "Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", r / "memrift_weights" / "llama31_8b_level18"),
    }

def disk_usage_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def bf16_model_bytes_meta(hf_path: str, trust_remote_code: bool = True) -> Optional[int]:
    try:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype="bfloat16", device_map="meta", trust_remote_code=trust_remote_code)
        n = sum(p.numel() for p in m.parameters())
        del m
        return int(n * 2)
    except Exception as e:
        print(f"[warn] bf16_model_bytes_meta: {e}", file=sys.stderr)
        return None

def build_peft_model(hf_path, lora_r, lora_alpha, device, trust_remote_code=True):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype="bfloat16", device_map=device, trust_remote_code=trust_remote_code)
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(base, cfg)
    tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok

def measure_load_time_peft(hf_path, lora_r, lora_alpha, device, trust_remote_code=True):
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, _ = build_peft_model(hf_path, lora_r, lora_alpha, device, trust_remote_code)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    dev = next(model.parameters()).device
    v = getattr(model.config, "vocab_size", 32000)
    ids = torch.randint(0, min(v, 50000), (1, 32), device=dev, dtype=torch.long)
    with torch.no_grad():
        model(ids)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    del model, _
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"load_to_first_forward_s": t1 - t0, "peak_mem_gb_during_load": peak}

def ppl_wikitext(model, tokenizer, device, max_length, max_samples):
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    import torch
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t["text"] for t in ds if len(t.get("text") or "") > 50][:max_samples]
    if not texts:
        return None
    model.eval()
    tn, tt = 0.0, 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            if enc["input_ids"].numel() < 2:
                continue
            out = model(**enc, labels=enc["input_ids"])
            n = enc["input_ids"].numel()
            tn += float(out.loss.item()) * n
            tt += n
    return math.exp(tn / tt) if tt else None

def boolq_accuracy(model, tokenizer, device, max_samples=64):
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    import torch
    try:
        ds = load_dataset("boolq", split=f"validation[:{max_samples}]")
    except Exception as e:
        print(f"[warn] boolq: {e}", file=sys.stderr)
        return None
    model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for row in ds:
            prefix = f"{row['passage']}\nQuestion: {row['question']}\nAnswer:"
            lab = bool(row["answer"])
            losses = []
            for a in (" yes", " no"):
                enc = tokenizer(prefix + a, return_tensors="pt", truncation=True, max_length=512)
                enc = {k: v.to(device) for k, v in enc.items()}
                if enc["input_ids"].shape[1] < 2:
                    losses.append(1e9)
                    continue
                out = model(**enc, labels=enc["input_ids"])
                losses.append(float(out.loss.item()))
            pred = losses[0] < losses[1]
            if pred == lab:
                c += 1
            t += 1
    return c / t if t else None

def run_one(spec, args, outp):
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rb = {"model_key": spec.key, "model_name": spec.name, "hf_path": spec.hf_path, "device": device, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    bf = bf16_model_bytes_meta(spec.hf_path, args.trust_remote_code)
    cb = disk_usage_bytes(spec.comp_dir)
    r = (cb / bf) if (bf and cb) else None
    with outp.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**rb, "branch": "disk", "compressed_dir": str(spec.comp_dir), "compressed_bytes": cb, "bf16_meta_bytes": bf, "r_compressed_over_bf16": r, "comp_dir_has_index": (spec.comp_dir / "index.json").is_file()}, ensure_ascii=False) + "\n")
    if args.skip_lora_eval:
        return
    try:
        lt = measure_load_time_peft(spec.hf_path, args.lora_r, args.lora_alpha, device, args.trust_remote_code)
    except Exception as e:
        with outp.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**rb, "branch": "lora", "error": str(e)}, ensure_ascii=False) + "\n")
        return
    model, tok = build_peft_model(spec.hf_path, args.lora_r, args.lora_alpha, device, args.trust_remote_code)
    ppl = ppl_wikitext(model, tok, device, args.ppl_max_length, args.ppl_max_samples) if not args.skip_ppl else None
    bq = boolq_accuracy(model, tok, device, args.boolq_max_samples) if not args.skip_boolq else None
    with outp.open("a", encoding="utf-8") as f:
        f.write(json.dumps({**rb, "branch": "lora", **lt, "ppl_wikitext2": ppl, "boolq_acc": bq, "ppl_dataset": "wikitext-2-raw-v1", "ppl_max_samples": args.ppl_max_samples, "ppl_max_length": args.ppl_max_length}, ensure_ascii=False) + "\n")
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    models = default_models()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tinyllama-1.1b"])
    ap.add_argument("--output", type=Path, default=Path("memrift_metrics.jsonl"))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--ppl-max-samples", type=int, default=32)
    ap.add_argument("--ppl-max-length", type=int, default=256)
    ap.add_argument("--boolq-max-samples", type=int, default=64)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--skip-boolq", action="store_true")
    ap.add_argument("--skip-lora-eval", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    args = ap.parse_args()
    keys = list(models.keys()) if "all" in args.models else args.models
    for k in keys:
        if k not in models:
            print("unknown", k, file=sys.stderr)
            return 1
    if args.output.exists():
        args.output.unlink()
    for k in keys:
        print("===", k, "===")
        run_one(models[k], args, args.output)
    print("wrote", args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
