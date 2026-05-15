"""
MemRift-enabled vLLM inference.
Same interface as inference_llm.py but adds:
  - load_format=MemRiftModelLoader  → layer-by-layer weight streaming
  - enforce_eager=True              → required (torch.compile breaks hooks)

Config YAML example:
  llm:
    model: /path/to/mistral-7b
    dtype: bfloat16
    gpu_memory_utilization: 0.4   # only KV cache + 1 layer needed
    max_model_len: 4096
  generate:
    prompts:
      - "Once upon a time"
    sampling:
      max_tokens: 50
      temperature: 0.0

Env vars:
  MEMRIFT_ZSTD_LEVEL       zstd level for compression (default: 3)
  MEMRIFT_DECODE_WORKERS   thread pool size for decompression (default: 16)
  MEMRIFT_PREFETCH_LAYERS  layers to prefetch ahead (default: 1)
"""

import os
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config
from flagscale.compress.memrift.vllm_loader import MemRiftModelLoader


def inference(cfg):
    """MemRift-enabled inference: same API as inference_llm.inference()."""

    # ── Prompts ──────────────────────────────────────────────────────────
    prompts = cfg.generate.get("prompts", [])
    assert prompts, "Please set the prompts in the config yaml."

    # ── LLM config ───────────────────────────────────────────────────────
    llm_cfg = dict(cfg.get("llm", {}))

    # Inject MemRift loader and required settings
    llm_cfg["load_format"] = MemRiftModelLoader
    llm_cfg.setdefault("enforce_eager", True)  # hooks need eager mode
    llm_cfg.setdefault("dtype", "bfloat16")

    print("[MemRift-vLLM] Initializing LLM with MemRiftModelLoader …")
    print(f"[MemRift-vLLM] enforce_eager=True (required for weight streaming)")

    t0 = time.perf_counter()
    llm = LLM(**llm_cfg)
    setup_time = time.perf_counter() - t0
    print(f"[MemRift-vLLM] LLM ready in {setup_time:.1f}s")

    # ── Optional custom tokenizer ─────────────────────────────────────────
    tokenizer_cfg = llm_cfg.get("tokenizer", None)
    if tokenizer_cfg:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg,
                                                  trust_remote_code=True)
        llm.set_tokenizer(tokenizer)

    # ── Sampling params ───────────────────────────────────────────────────
    sampling_cfg = cfg.generate.get("sampling", {})
    assert not sampling_cfg.get("logits_processors", None), \
        "logits_processors is not supported."
    sampling_params = SamplingParams(**sampling_cfg)
    print(f"=> {sampling_params=}")

    # ── Generate ──────────────────────────────────────────────────────────
    inputs = [{"prompt": p} for p in prompts]
    print(f"=> {inputs=}")

    import torch
    torch.cuda.reset_peak_memory_stats()
    t_gen = time.perf_counter()
    outputs = llm.generate(inputs, sampling_params)
    gen_time = time.perf_counter() - t_gen
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    # ── Results ───────────────────────────────────────────────────────────
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    for output in outputs:
        print("*" * 50)
        print(f"{output.prompt=}")
        print(f"{output.outputs[0].text=}")
    print("#" * 50)
    print(f"[MemRift-vLLM] Generated {total_tokens} tokens in {gen_time:.2f}s "
          f"({total_tokens/gen_time:.1f} tok/s)")
    print(f"[MemRift-vLLM] GPU peak memory (generation): {peak_gb:.3f} GB")


if __name__ == "__main__":
    cfg = parse_config()
    inference(cfg)
