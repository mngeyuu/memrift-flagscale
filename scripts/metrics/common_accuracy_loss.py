#!/usr/bin/env python3
"""Compare final training lm loss for Pure LoRA and MemRift+LoRA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _validated_loss(metrics: dict[str, Any], label: str) -> float:
    if not metrics.get("log_exists"):
        raise ValueError(f"{label} training log is missing")
    if metrics.get("failed_pattern_found"):
        raise ValueError(f"{label} training log contains an error pattern")
    loss = metrics.get("final_lm_loss")
    if not isinstance(loss, (int, float)) or isinstance(loss, bool):
        raise ValueError(f"{label} final_lm_loss must be numeric")
    loss = float(loss)
    if not math.isfinite(loss) or loss <= 0.0:
        raise ValueError(f"{label} final_lm_loss must be finite and positive")
    return loss


def compare_loss(
    model_key: str,
    model_name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold: float = 0.01,
) -> dict[str, Any]:
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    baseline_loss = _validated_loss(baseline, "Pure LoRA")
    candidate_loss = _validated_loss(candidate, "MemRift+LoRA")
    relative_degradation = (candidate_loss - baseline_loss) / baseline_loss
    passed = relative_degradation <= threshold or math.isclose(
        relative_degradation, threshold, abs_tol=1e-12
    )
    return {
        "metric": "accuracy_loss",
        "model_key": model_key,
        "model_name": model_name,
        "measurement": "final training lm loss",
        "criterion": "MemRift weight-compressed LoRA relative final lm loss degradation <= 1% versus Pure LoRA",
        "baseline_pure_lora": baseline,
        "memrift_weight_only": candidate,
        "pure_lora_final_lm_loss": baseline_loss,
        "memrift_final_lm_loss": candidate_loss,
        "relative_loss_degradation": relative_degradation,
        "relative_loss_degradation_percent": relative_degradation * 100.0,
        "threshold_relative_loss_degradation": threshold,
        "threshold_relative_loss_degradation_percent": threshold * 100.0,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    result = compare_loss(
        args.model_key,
        args.model_name,
        _read_metrics(args.baseline),
        _read_metrics(args.candidate),
        args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
