#!/usr/bin/env python3
"""Evaluate GSM8K CoT and HellaSwag through a loaded Megatron model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scripts.metrics.benchmark_tasks import (
    BenchmarkExample,
    load_task_examples,
    score_choices,
    score_generation,
)


def evaluate_examples(
    task: str,
    examples: list[BenchmarkExample],
    backend: Any,
    gsm8k_max_new_tokens: int = 64,
) -> dict[str, Any]:
    if not examples:
        raise ValueError(f"{task} has no examples")
    if task == "gsm8k_cot":
        outputs = backend.generate([example.prompt for example in examples], max_new_tokens=gsm8k_max_new_tokens)
        if len(outputs) != len(examples):
            raise RuntimeError(f"backend returned {len(outputs)} generations for {len(examples)} examples")
        scores = [score_generation(example, output) for example, output in zip(examples, outputs)]
        scoring = "8-shot greedy generation exact match"
    elif task == "hellaswag":
        requests = [(example.prompt, choice) for example in examples for choice in example.choices]
        outputs = backend.loglikelihood(requests)
        if len(outputs) != len(requests):
            raise RuntimeError(f"backend returned {len(outputs)} likelihoods for {len(requests)} requests")
        scores = []
        for index, example in enumerate(examples):
            group = outputs[index * 4 : (index + 1) * 4]
            scores.append(score_choices(example, [item[0] for item in group], [item[1] for item in group]))
        scoring = "four-choice length-normalized conditional loglikelihood (acc_norm)"
    else:
        raise ValueError(f"unsupported task: {task}")

    details = [
        {"sample_id": score.sample_id, "prediction": score.prediction, "target": score.target, "correct": score.correct}
        for score in scores
    ]
    correct = sum(score.correct for score in scores)
    return {
        "task": task,
        "num_samples": len(scores),
        "num_correct": correct,
        "accuracy": correct / len(scores),
        "scoring": scoring,
        "details": details,
    }


def write_benchmark_result(output: Path, variant: str, tasks: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not tasks:
        raise ValueError("result requires at least one benchmark task")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"variant": variant, "tasks": tasks, "metadata": metadata}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _optional_limit(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def _enabled_tasks() -> list[tuple[str, int | None]]:
    enabled = []
    for task, env_name in (("gsm8k_cot", "GSM8K_LIMIT"), ("hellaswag", "HELLASWAG_LIMIT")):
        limit = _optional_limit(env_name)
        if limit == 0:
            continue
        enabled.append((task, limit))
    if not enabled:
        raise ValueError("at least one benchmark task must be enabled")
    return enabled


def _shard_examples(examples: list[BenchmarkExample]) -> tuple[list[BenchmarkExample], int, int]:
    shard_count = int(os.environ.get("BENCHMARK_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("BENCHMARK_SHARD_INDEX", "0"))
    if shard_count <= 0:
        raise ValueError("BENCHMARK_SHARD_COUNT must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("BENCHMARK_SHARD_INDEX must be in [0, BENCHMARK_SHARD_COUNT)")
    sharded = examples[shard_index::shard_count]
    if not sharded:
        raise ValueError(f"benchmark shard {shard_index}/{shard_count} has no examples")
    return sharded, shard_index, shard_count


def main() -> int:
    from scripts.metrics.megatron_benchmark_backend import MegatronBenchmarkBackend

    output = Path(os.environ["BENCHMARK_OUTPUT"])
    variant = os.environ["BENCHMARK_VARIANT"]
    backend = MegatronBenchmarkBackend()
    tasks = {}
    enabled_tasks = _enabled_tasks()
    for task, limit in enabled_tasks:
        examples = load_task_examples(task, limit)
        examples, shard_index, shard_count = _shard_examples(examples)
        print(
            f"[benchmark] task={task} shard={shard_index + 1}/{shard_count} "
            f"samples={len(examples)}",
            flush=True,
        )
        tasks[task] = evaluate_examples(
            task,
            examples,
            backend,
            gsm8k_max_new_tokens=int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "64")),
        )
    write_benchmark_result(
        output,
        variant,
        tasks,
        {
            "checkpoint": os.environ.get("BENCHMARK_CHECKPOINT"),
            "mode": os.environ.get("BENCHMARK_MODE"),
            "gsm8k_limit": _optional_limit("GSM8K_LIMIT"),
            "hellaswag_limit": _optional_limit("HELLASWAG_LIMIT"),
            "gsm8k_max_new_tokens": int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "64")),
            "shard_index": int(os.environ.get("BENCHMARK_SHARD_INDEX", "0")),
            "shard_count": int(os.environ.get("BENCHMARK_SHARD_COUNT", "1")),
            "task_definition": "lm_eval_tasks/gsm8k_cot_local.yaml and hellaswag_local.yaml compatible",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
