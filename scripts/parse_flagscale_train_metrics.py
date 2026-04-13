#!/usr/bin/env python3
"""Parse FlagScale training host log for peak GPU memory (MB) and iteration time (ms)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_log_text(text: str) -> Dict[str, Any]:
    max_alloc: Optional[float] = None
    for m in re.finditer(r"max allocated:\s*([\d.]+)", text):
        max_alloc = float(m.group(1))

    times: List[float] = []
    for m in re.finditer(r"elapsed time per iteration \(ms\):\s*([\d.]+)", text):
        times.append(float(m.group(1)))

    return {
        "max_allocated_mb": max_alloc,
        "elapsed_ms_per_iter_all": times,
        "elapsed_ms_per_iter_mean": (sum(times) / len(times)) if times else None,
        "elapsed_ms_per_iter_last": times[-1] if times else None,
        "num_time_samples": len(times),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_file", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    text = args.log_file.read_text(encoding="utf-8", errors="replace")
    data = parse_log_text(text)
    data["source_log"] = str(args.log_file)
    out = json.dumps(data, indent=2, ensure_ascii=False)
    print(out)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
