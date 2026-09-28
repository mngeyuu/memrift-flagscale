#!/usr/bin/env python3
"""Render a compact, video-friendly acceptance summary for MemRift metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_ORDER = (
    "compression_ratio",
    "load_time_reduction",
    "accuracy_loss",
    "train_context_gain",
)

MODEL_CATEGORIES = {
    "qwen3_8b": "业界主流模型",
    "llama8b": "业界主流模型",
    "aquila": "智源自研模型",
}

SCENARIOS = {
    "compression_ratio": "模型压缩比例及精度损耗测试",
    "accuracy_loss": "模型压缩比例及精度损耗测试",
    "train_context_gain": "训练与推理上下文长度测试",
    "load_time_reduction": "推理模型载入时间测试",
}

MEASUREMENT_SCOPES = {
    "compression_ratio": (
        "BF16 模型参数参考大小与 MemRift 压缩目录文件总大小；"
        "以存储节省率作为压缩收益。"
    ),
    "accuracy_loss": (
        "Pure LoRA 与 MemRift 权重压缩+LoRA 使用相同模型、数据、batch size 和训练配置；"
        "关闭激活压缩，比较训练日志中的最终 lm loss。"
    ),
    "train_context_gain": (
        "固定 GPU、模型、batch size 和训练配置，对比最大可运行训练长度。"
        "注意：当前脚本未测试推理侧最大上下文。"
    ),
    "load_time_reduction": (
        "相同文件集合与读取实现下的权重磁盘读取时间；不含模型实例化和首轮前向。"
        "正式验收应保持存储和 OS 缓存条件一致。"
    ),
}


@dataclass(frozen=True)
class DisplayRow:
    item: str
    measured: str
    criterion: str
    status: str
    calculation: str = ""


def _percent(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.2f}%"


def _percent_precise(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.4f}%"


def _gib(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value) / (1024 ** 3):.2f} GiB"


def _gib_precise(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value) / (1024 ** 3):.4f} GiB"


def _seconds(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.2f}s"


def _seconds_precise(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.4f}s"


def _decimal(value: Any, digits: int = 6) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _status(data: dict[str, Any]) -> str:
    value = data.get("pass")
    if not isinstance(value, bool):
        return "ERROR"
    return "PASS" if value else "FAIL"


def _decision(status: str) -> str:
    return "符合指标 (PASS)" if status == "PASS" else "不符合指标 (FAIL)"


def _row_for_compression(data: dict[str, Any]) -> list[DisplayRow]:
    status = _status(data)
    reference = _gib(data.get("bf16_reference_bytes"))
    compressed = _gib(data.get("compressed_bytes"))
    saving = _percent(data.get("saving_percent"))
    measured = f"BF16 {reference} -> MemRift {compressed}; 节省 {saving}"
    target = data.get("target_saving_fraction")
    target_percent = _percent(target * 100 if isinstance(target, (int, float)) else None)
    criterion = f"存储节省率 >= {target_percent}"
    reference_precise = _gib_precise(data.get("bf16_reference_bytes"))
    compressed_precise = _gib_precise(data.get("compressed_bytes"))
    saving_precise = _percent_precise(data.get("saving_percent"))
    calculation = (
        f"({reference_precise} - {compressed_precise}) / {reference_precise} × 100% = "
        f"{saving_precise}; {saving_precise} >= {target_percent} => {_decision(status)}"
    )
    return [DisplayRow("模型压缩比例", measured, criterion, status, calculation)]


def _row_for_load_time(data: dict[str, Any]) -> list[DisplayRow]:
    status = _status(data)
    baseline = data.get("baseline_model_weights")
    candidate = data.get("memrift_compressed_weights")
    baseline_time = baseline.get("read_seconds") if isinstance(baseline, dict) else None
    candidate_time = candidate.get("read_seconds") if isinstance(candidate, dict) else None
    baseline_seconds = _seconds(baseline_time)
    candidate_seconds = _seconds(candidate_time)
    reduction = _percent(data.get("reduction_percent"))
    measured = f"T0={baseline_seconds} -> T1={candidate_seconds}; 降低 {reduction}"
    target = data.get("target_reduction_fraction")
    target_percent = _percent(target * 100 if isinstance(target, (int, float)) else None)
    criterion = "T1 <= 0.70 × T0"
    baseline_precise = _seconds_precise(baseline_time)
    candidate_precise = _seconds_precise(candidate_time)
    reduction_precise = _percent_precise(data.get("reduction_percent"))
    time_ratio = (
        candidate_time / baseline_time
        if isinstance(baseline_time, (int, float))
        and isinstance(candidate_time, (int, float))
        and baseline_time > 0
        else None
    )
    time_ratio_text = f"{time_ratio:.4f}" if time_ratio is not None else "n/a"
    calculation = (
        f"({baseline_precise} - {candidate_precise}) / {baseline_precise} × 100% = "
        f"{reduction_precise}; T1/T0 = {time_ratio_text} <= 0.7000 "
        f"=> {_decision(status)}"
    )
    return [DisplayRow("推理模型载入时间", measured, criterion, status, calculation)]


def _accuracy_task_row(task: str, data: dict[str, Any]) -> DisplayRow:
    names = {"gsm8k_cot": "精度损耗 / GSM8K CoT", "hellaswag": "精度损耗 / HellaSwag"}
    baseline = data.get("pure_lora_accuracy")
    candidate = data.get("memrift_accuracy")
    status = _status(data)
    baseline_percent = _percent(baseline * 100 if isinstance(baseline, (int, float)) else None)
    candidate_percent = _percent(candidate * 100 if isinstance(candidate, (int, float)) else None)
    drop = _percent(data.get("relative_accuracy_drop_percent"))
    measured = f"LoRA {baseline_percent} -> MemRift {candidate_percent}; 损耗 {drop}"
    threshold = data.get("threshold_relative_accuracy_drop_percent")
    threshold_percent = _percent(threshold)
    baseline_precise = _percent_precise(
        baseline * 100 if isinstance(baseline, (int, float)) else None
    )
    candidate_precise = _percent_precise(
        candidate * 100 if isinstance(candidate, (int, float)) else None
    )
    drop_precise = _percent_precise(data.get("relative_accuracy_drop_percent"))
    return DisplayRow(
        names.get(task, f"Accuracy / {task}"),
        measured,
        f"相对精度损耗 <= {threshold_percent}",
        status,
        (
            f"({baseline_precise} - {candidate_precise}) / {baseline_precise} × 100% = "
            f"{drop_precise}; {drop_precise} <= {threshold_percent} => {_decision(status)}"
        ),
    )


def _row_for_accuracy(data: dict[str, Any]) -> list[DisplayRow]:
    tasks = data.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        status = _status(data)
        baseline_data = data.get("baseline_pure_lora")
        candidate_data = data.get("memrift_weight_only")
        if not isinstance(candidate_data, dict):
            candidate_data = data.get("memrift_weight_act_async")
        baseline_loss = data.get("pure_lora_final_lm_loss")
        candidate_loss = data.get("memrift_final_lm_loss")
        if baseline_loss is None and isinstance(baseline_data, dict):
            baseline_loss = baseline_data.get("final_lm_loss")
        if candidate_loss is None and isinstance(candidate_data, dict):
            candidate_loss = candidate_data.get("final_lm_loss")
        baseline_text = _decimal(baseline_loss)
        candidate_text = _decimal(candidate_loss)
        degradation = _percent(data.get("relative_loss_degradation_percent"))
        degradation_precise = _percent_precise(data.get("relative_loss_degradation_percent"))
        threshold = data.get("threshold_relative_loss_degradation_percent", 1.0)
        threshold_text = _percent(threshold)
        measured = (
            f"L0={baseline_text} -> L1={candidate_text}; ΔLoss {degradation}"
        )
        calculation = (
            f"({candidate_text} - {baseline_text}) / {baseline_text} × 100% = "
            f"{degradation_precise}; {degradation_precise} <= {threshold_text} "
            f"=> {_decision(status)}"
        )
        return [
            DisplayRow(
                "Loss 损失",
                measured,
                f"ΔLoss <= {threshold_text}",
                status,
                calculation,
            )
        ]
    order = ("gsm8k_cot", "hellaswag")
    rows = []
    for task in order:
        task_data = tasks.get(task)
        if isinstance(task_data, dict):
            rows.append(_accuracy_task_row(task, task_data))
    for task, task_data in tasks.items():
        if task not in order and isinstance(task_data, dict):
            rows.append(_accuracy_task_row(task, task_data))
    if not rows:
        return [DisplayRow("模型精度损耗", "任务结果无效", "每项 <= 1.00%", "ERROR")]
    return rows


def _row_for_context(data: dict[str, Any]) -> list[DisplayRow]:
    status = _status(data)
    baseline = data.get("baseline_max_context_tokens")
    candidate = data.get("memrift_max_context_tokens")
    baseline_tokens = str(baseline) if isinstance(baseline, int) else "n/a"
    candidate_tokens = str(candidate) if isinstance(candidate, int) else "n/a"
    gain = _percent(data.get("context_gain_percent"))
    gain_precise = _percent_precise(data.get("context_gain_percent"))
    context_ratio = (
        candidate / baseline
        if isinstance(baseline, int) and isinstance(candidate, int) and baseline > 0
        else None
    )
    context_ratio_text = f"{context_ratio:.4f}" if context_ratio is not None else "n/a"
    measured = f"L0={baseline_tokens} -> L1={candidate_tokens} tokens; 增长 {gain}"
    calculation = (
        f"({candidate_tokens} - {baseline_tokens}) / {baseline_tokens} × 100% = {gain_precise}; "
        f"L1/L0 = {context_ratio_text} >= 1.2000 => {_decision(status)}"
    )
    return [DisplayRow("训练最大上下文长度", measured, "L1 >= 1.20 × L0", status, calculation)]


ROW_BUILDERS = {
    "compression_ratio": _row_for_compression,
    "load_time_reduction": _row_for_load_time,
    "accuracy_loss": _row_for_accuracy,
    "train_context_gain": _row_for_context,
}


def build_report(
    result_paths: list[Path],
    model_key: str,
    model_name: str,
    freshness_marker: Path | None = None,
    metrics: tuple[str, ...] = METRIC_ORDER,
) -> dict[str, Any]:
    paths_by_metric = {path.parent.name: path for path in result_paths}
    marker_mtime = (
        freshness_marker.stat().st_mtime_ns
        if freshness_marker is not None and freshness_marker.is_file()
        else None
    )
    results: dict[str, Any] = {}
    issues: dict[str, str] = {}
    rows: list[DisplayRow] = []

    for metric in metrics:
        path = paths_by_metric.get(metric)
        if path is None:
            issues[metric] = "result path was not supplied"
        elif not path.is_file():
            issues[metric] = f"missing result: {path}"
        elif marker_mtime is not None and path.stat().st_mtime_ns < marker_mtime:
            issues[metric] = f"result was not refreshed by this run: {path}"
        else:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON value is not an object")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issues[metric] = f"invalid result {path}: {exc}"
            else:
                results[metric] = data
                rows.extend(ROW_BUILDERS[metric](data))
                continue

        rows.append(DisplayRow(metric.replace("_", " ").title(), issues[metric], "required", "MISSING"))

    metric_pass = {
        metric: isinstance(results.get(metric, {}).get("pass"), bool)
        and results[metric]["pass"]
        for metric in metrics
    }
    complete = len(results) == len(metrics) and not issues
    overall_pass = complete and all(metric_pass.values())
    return {
        "model_key": model_key,
        "model_name": model_name,
        "requested_metrics": list(metrics),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete": complete,
        "overall_pass": overall_pass,
        "metric_pass": metric_pass,
        "issues": issues,
        "metrics": results,
        "rows": rows,
    }


def _display_width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in value)


def _clip(value: str, width: int) -> str:
    if _display_width(value) <= width:
        return value
    clipped = ""
    for char in value:
        if _display_width(clipped + char) > width - 3:
            break
        clipped += char
    return clipped + "..."


def _pad(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def _acceptance_metadata(report: dict[str, Any]) -> tuple[str, str]:
    requested_metrics = report["requested_metrics"]
    if len(requested_metrics) == 1:
        metric = requested_metrics[0]
        return SCENARIOS[metric], MEASUREMENT_SCOPES[metric]
    return (
        "全部性能指标验收（三项测试场景、四项量化指标）",
        "各项采用对应脚本的统一基线、优化方案和判定阈值；详见下表。",
    )


def render_report(report: dict[str, Any], color: bool = False) -> str:
    widths = (26, 57, 22, 9)
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(values: tuple[str, str, str, str]) -> str:
        cells = [f" {_pad(_clip(value, width), width)} " for value, width in zip(values, widths)]
        return "|" + "|".join(cells) + "|"

    status_colors = {
        "PASS": "\033[32;1m",
        "FAIL": "\033[31;1m",
        "ERROR": "\033[31;1m",
        "MISSING": "\033[33;1m",
    }
    reset = "\033[0m"
    table = [border, line(("性能指标", "实测结果", "验收要求", "判定")), border]
    for row in report["rows"]:
        rendered = line((row.item, row.measured, row.criterion, row.status))
        if color:
            rendered = rendered.replace(
                f" {_pad(row.status, widths[3])} ",
                f" {status_colors.get(row.status, '')}{_pad(row.status, widths[3])}{reset} ",
            )
        table.append(rendered)
    table.append(border)

    calculations = [row for row in report["rows"] if row.calculation]
    if calculations:
        table.append("收益比例计算与验收判定：")
        for row in calculations:
            table.append(f"  - {row.item}: {row.calculation}")

    requested_metrics = report["requested_metrics"]
    scenario, measurement_scope = _acceptance_metadata(report)
    passed = sum(bool(value) for value in report["metric_pass"].values())
    overall = "PASS" if report["overall_pass"] else "FAIL"
    if len(requested_metrics) == 1:
        overall_text = f"本项最终判定：{_decision(overall)}"
    else:
        overall_text = (
            f"总体最终判定：{_decision(overall)} "
            f"（{passed}/{len(requested_metrics)} 项量化指标通过）"
        )
    if color:
        overall_text = f"{status_colors[overall]}{overall_text}{reset}"

    return "\n".join(
        [
            "",
            "=" * len(border),
            (
                "大模型压缩系统 - 单项性能指标验收结果"
                if len(requested_metrics) == 1
                else "大模型压缩系统 - 性能指标验收汇总"
            ),
            "软件名称：大模型压缩系统",
            f"测试模型：{report['model_name']} ({report['model_key']})",
            f"模型类别：{MODEL_CATEGORIES.get(report['model_key'], '未指定')}",
            f"测试场景：{scenario}",
            f"测试口径：{measurement_scope}",
            f"生成时间（UTC）：{report['generated_at_utc']}",
            *table,
            overall_text,
            "=" * len(border),
        ]
    )


def _json_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "rows"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--freshness-marker", type=Path)
    parser.add_argument("--only", choices=METRIC_ORDER)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    report = build_report(
        args.results,
        args.model_key,
        args.model_name,
        freshness_marker=args.freshness_marker,
        metrics=(args.only,) if args.only else METRIC_ORDER,
    )
    output = _json_report(report)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary_out.with_suffix(args.summary_out.suffix + ".tmp")
        temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(args.summary_out)

    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    print(render_report(report, color=use_color))
    evidence = args.summary_out if args.summary_out is not None else args.results[0]
    print(f"验收证据 JSON：{evidence}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
