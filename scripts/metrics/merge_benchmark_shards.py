#!/usr/bin/env python3
"""Merge deterministic benchmark shards into one standard result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    variant = payloads[0]["variant"]
    if any(item["variant"] != variant for item in payloads):
        raise ValueError("all shards must use the same variant")

    task_names = set(payloads[0]["tasks"])
    if any(set(item["tasks"]) != task_names for item in payloads):
        raise ValueError("all shards must contain the same tasks")

    tasks = {}
    for task_name in sorted(task_names):
        parts = [item["tasks"][task_name] for item in payloads]
        details = [detail for part in parts for detail in part["details"]]
        sample_ids = [detail["sample_id"] for detail in details]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"duplicate sample ids while merging {task_name}")
        details.sort(key=lambda item: int(item["sample_id"].rsplit(":", 1)[-1]))
        correct = sum(bool(item["correct"]) for item in details)
        tasks[task_name] = {
            "task": task_name,
            "num_samples": len(details),
            "num_correct": correct,
            "accuracy": correct / len(details),
            "scoring": parts[0]["scoring"],
            "details": details,
        }

    output = {
        "variant": variant,
        "tasks": tasks,
        "metadata": {
            **payloads[0].get("metadata", {}),
            "shard_index": None,
            "shard_count": len(payloads),
            "merged_shards": [str(path) for path in args.shards],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    for task_name, result in tasks.items():
        print(f"{variant} {task_name}: {result['num_correct']}/{result['num_samples']} = {result['accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
