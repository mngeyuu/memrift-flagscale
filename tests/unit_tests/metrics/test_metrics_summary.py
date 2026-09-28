import json
import os
from pathlib import Path

from scripts.metrics.metrics_summary import build_report, render_report


ROOT = Path(__file__).resolve().parents[3]


def write_result(root: Path, metric: str, data: dict) -> Path:
    path = root / metric / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def passing_results(root: Path) -> list[Path]:
    return [
        write_result(
            root,
            "compression_ratio",
            {
                "saving_percent": 43.75,
                "bf16_reference_bytes": 16 * 1024**3,
                "compressed_bytes": 9 * 1024**3,
                "target_saving_fraction": 0.30,
                "pass": True,
            },
        ),
        write_result(
            root,
            "load_time_reduction",
            {
                "metric": "load_time_reduction",
                "reduction_percent": 40.0,
                "target_reduction_fraction": 0.30,
                "baseline_model_weights": {"read_seconds": 10.0},
                "memrift_compressed_weights": {"read_seconds": 6.0},
                "pass": True,
            },
        ),
        write_result(
            root,
            "accuracy_loss",
            {
                "metric": "accuracy_loss",
                "baseline_pure_lora": {"final_lm_loss": 2.0},
                "memrift_weight_act_async": {"final_lm_loss": 2.01},
                "pure_lora_final_lm_loss": 2.0,
                "memrift_final_lm_loss": 2.01,
                "relative_loss_degradation_percent": 0.5,
                "threshold_relative_loss_degradation_percent": 1.0,
                "pass": True,
            },
        ),
        write_result(
            root,
            "train_context_gain",
            {
                "metric": "train_context_gain",
                "baseline_max_context_tokens": 10000,
                "memrift_max_context_tokens": 12500,
                "context_gain_percent": 25.0,
                "pass": True,
            },
        ),
    ]


def test_complete_passing_report_renders_video_friendly_table(tmp_path):
    report = build_report(passing_results(tmp_path), "demo", "Demo Model")

    assert report["complete"] is True
    assert report["overall_pass"] is True
    text = render_report(report)
    assert "大模型压缩系统 - 性能指标验收汇总" in text
    assert "软件名称：大模型压缩系统" in text
    assert "测试模型：Demo Model (demo)" in text
    assert "Loss 损失" in text
    assert "L0=2.000000 -> L1=2.010000" in text
    assert "总体最终判定：符合指标 (PASS) （4/4 项量化指标通过）" in text


def test_single_metric_report_has_standalone_result(tmp_path):
    compression = passing_results(tmp_path)[0]
    report = build_report(
        [compression],
        "demo",
        "Demo Model",
        metrics=("compression_ratio",),
    )

    assert report["overall_pass"] is True
    text = render_report(report)
    assert "大模型压缩系统 - 单项性能指标验收结果" in text
    assert "测试场景：模型压缩比例及精度损耗测试" in text
    assert "收益比例计算与验收判定" in text
    assert "(16.0000 GiB - 9.0000 GiB) / 16.0000 GiB × 100% = 43.7500%" in text
    assert "43.7500% >= 30.00% => 符合指标 (PASS)" in text
    assert "本项最终判定：符合指标 (PASS)" in text
    assert "总体最终判定" not in text


def test_failed_criterion_is_an_overall_failure(tmp_path):
    paths = passing_results(tmp_path)
    context_path = tmp_path / "train_context_gain" / "result.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["pass"] = False
    context_path.write_text(json.dumps(context), encoding="utf-8")

    report = build_report(paths, "demo", "Demo Model")

    assert report["complete"] is True
    assert report["overall_pass"] is False
    assert "总体最终判定：不符合指标 (FAIL) （3/4 项量化指标通过）" in render_report(report)


def test_missing_or_stale_results_cannot_reuse_old_acceptance_evidence(tmp_path):
    paths = passing_results(tmp_path)
    marker = tmp_path / "run.marker"
    marker.touch()
    old_mtime = marker.stat().st_mtime_ns - 1_000_000_000
    for path in paths:
        os.utime(path, ns=(old_mtime, old_mtime))
    missing = paths.pop()
    missing.unlink()

    report = build_report(paths, "demo", "Demo Model", freshness_marker=marker)

    assert report["complete"] is False
    assert report["overall_pass"] is False
    assert set(report["issues"]) == {
        "compression_ratio",
        "load_time_reduction",
        "accuracy_loss",
        "train_context_gain",
    }
    assert "MISSING" in render_report(report)


def test_every_individual_metric_script_installs_final_result_display():
    for model in ("qwen3_8b", "llama8b", "aquila"):
        for metric in (
            "compression_ratio",
            "load_time_reduction",
            "accuracy_loss",
            "train_context_gain",
        ):
            text = (ROOT / f"scripts/metrics/{model}/{metric}.sh").read_text(encoding="utf-8")
            assert 'start_metric_result_display "$RESULT"' in text
            assert "finish_metric_result_display" in text
            assert f" {metric} " in text
