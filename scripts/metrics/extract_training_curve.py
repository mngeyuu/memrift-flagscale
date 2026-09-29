#!/usr/bin/env python3
"""Extract every Megatron training iteration into auditable JSON and CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

LINE = re.compile(
    r"iteration\s+(?P<iteration>\d+)/\s*(?P<total>\d+).*?"
    r"learning rate:\s*(?P<lr>[0-9.E+-]+).*?"
    r"lm loss:\s*(?P<loss>[0-9.E+-]+).*?"
    r"grad norm:\s*(?P<grad_norm>[0-9.E+-]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, default=100)
    args = parser.parse_args()

    rows = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.search(line)
        if match:
            row = {key: int(value) if key in {"iteration", "total"} else float(value)
                   for key, value in match.groupdict().items()}
            row["perplexity"] = math.exp(row["loss"])
            rows.append(row)
    if len(rows) != args.expected_iterations:
        raise ValueError(f"expected {args.expected_iterations} iterations, found {len(rows)} in {args.log}")
    if [row["iteration"] for row in rows] != list(range(1, args.expected_iterations + 1)):
        raise ValueError("training iterations are not exactly 1..expected_iterations")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", "total", "lr", "loss", "perplexity", "grad_norm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} iterations: {args.json} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
