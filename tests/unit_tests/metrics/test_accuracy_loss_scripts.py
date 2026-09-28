from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMMON = (ROOT / "scripts/metrics/common.sh").read_text(encoding="utf-8")


def script(model):
    return (ROOT / f"scripts/metrics/{model}/accuracy_loss.sh").read_text(encoding="utf-8")


def test_accuracy_scripts_compare_final_lm_loss():
    for model in ("qwen3_8b", "llama8b", "aquila"):
        text = script(model)
        assert text.count("run_yaml_train") == 2
        assert text.count("parse_train_log_json") == 2
        assert "common_accuracy_loss.py" in text
        assert "run_benchmark_accuracy" not in text
        assert "benchmark_accuracy_result.py" not in text


def test_accuracy_loss_runs_do_not_save_checkpoints():
    for model in ("qwen3_8b", "llama8b", "aquila"):
        text = script(model)
        assert text.count("DISABLE_TRAIN_CHECKPOINT=true") == 2
        assert text.count("MEMRIFT_DISABLE_FINAL_CHECKPOINT=1") == 2
        assert "checkpoints" not in text
