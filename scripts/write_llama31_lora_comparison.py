#!/usr/bin/env python3
"""Merge memrift + pure LoRA parse JSON into comparison.json (used by run_llama31_8b_memrift_vs_lora_output.sh)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--train-iters", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--memrift-weight-dir", required=True)
    args = ap.parse_args()

    root: Path = args.out_root
    summary = root / "summary"

    def load_json(name: str) -> dict:
        p = summary / name
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    m = load_json("memrift_metrics.json")
    l = load_json("pure_lora_metrics.json")

    cmp: dict = {
        "train_iters": args.train_iters,
        "model": args.model,
        "memrift_weight_dir": args.memrift_weight_dir,
        "memrift_plus_lora": m,
        "pure_lora": l,
    }
    if m.get("max_allocated_mb") is not None and l.get("max_allocated_mb") is not None:
        cmp["peak_delta_mb_pure_minus_memrift"] = round(
            float(l["max_allocated_mb"]) - float(m["max_allocated_mb"]), 4
        )
    if m.get("elapsed_ms_per_iter_mean") and l.get("elapsed_ms_per_iter_mean"):
        cmp["iter_time_ratio_memrift_over_pure"] = round(
            float(m["elapsed_ms_per_iter_mean"]) / float(l["elapsed_ms_per_iter_mean"]), 6
        )

    summary.mkdir(parents=True, exist_ok=True)
    out = summary / "comparison.json"
    out.write_text(json.dumps(cmp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
