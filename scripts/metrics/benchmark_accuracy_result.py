#!/usr/bin/env python3
"""Validate and aggregate pure-LoRA versus MemRift benchmark accuracy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TASKS = ("gsm8k_cot", "hellaswag")


def _validate_task_result(expected_task: str, result: dict[str, Any], label: str) -> float:
    if not isinstance(result, dict):
        raise ValueError(f"{label} task result must be an object")
    if result.get("task") != expected_task:
        raise ValueError(f"{label} task must be {expected_task!r}")
    samples = result.get("num_samples")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError(f"{label} num_samples must be a positive integer")
    accuracy = result.get("accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise ValueError(f"{label} accuracy must be numeric")
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"{label} accuracy must be finite and in [0, 1]")
    return accuracy


def compare_task(
    task: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold: float = 0.01,
) -> dict[str, Any]:
    baseline_accuracy = _validate_task_result(task, baseline, "baseline")
    candidate_accuracy = _validate_task_result(task, candidate, "candidate")
    if baseline_accuracy <= 0.0:
        raise ValueError("baseline accuracy must be positive for relative comparison")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    absolute_drop = baseline_accuracy - candidate_accuracy
    relative_drop = absolute_drop / baseline_accuracy
    return {
        "task": task,
        "pure_lora_accuracy": baseline_accuracy,
        "memrift_accuracy": candidate_accuracy,
        "absolute_drop_percentage_points": absolute_drop * 100.0,
        "relative_accuracy_drop": relative_drop,
        "relative_accuracy_drop_percent": relative_drop * 100.0,
        "threshold_relative_accuracy_drop_percent": threshold * 100.0,
        "pass": relative_drop <= threshold or math.isclose(relative_drop, threshold, abs_tol=1e-12),
    }


def aggregate_results(
    model_key: str,
    model_name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold: float = 0.01,
    tasks: tuple[str, ...] = TASKS,
) -> dict[str, Any]:
    baseline_tasks = baseline.get("tasks") if isinstance(baseline, dict) else None
    candidate_tasks = candidate.get("tasks") if isinstance(candidate, dict) else None
    required = set(tasks)
    if not required:
        raise ValueError("at least one task must be selected")
    if not isinstance(baseline_tasks, dict) or not required.issubset(baseline_tasks):
        raise ValueError(f"baseline tasks must contain {sorted(required)}")
    if not isinstance(candidate_tasks, dict) or not required.issubset(candidate_tasks):
        raise ValueError(f"candidate tasks must contain {sorted(required)}")

    comparisons = {
        task: compare_task(task, baseline_tasks[task], candidate_tasks[task], threshold)
        for task in tasks
    }
    return {
        "metric": "accuracy_loss",
        "model_key": model_key,
        "model_name": model_name,
        "criterion": "relative benchmark accuracy degradation <= 1% for every selected task",
        "selected_tasks": list(tasks),
        "threshold_relative_accuracy_drop": threshold,
        "baseline_pure_lora": baseline,
        "memrift_weight_act_async": candidate,
        "tasks": comparisons,
        "pass": all(item["pass"] for item in comparisons.values()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()

    result = aggregate_results(
        args.model_key,
        args.model_name,
        _read_json(args.baseline),
        _read_json(args.candidate),
        args.threshold,
        tuple(args.tasks),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
