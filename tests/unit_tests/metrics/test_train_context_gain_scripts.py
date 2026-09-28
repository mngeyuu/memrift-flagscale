from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMMON = (ROOT / "scripts/metrics/common.sh").read_text(encoding="utf-8")


def script(model):
    return (ROOT / f"scripts/metrics/{model}/train_context_gain.sh").read_text(encoding="utf-8")


def test_context_search_does_not_save_checkpoints():
    for model in ("aquila", "llama8b"):
        text = script(model)
        assert "find_max_context_len lora" in text
        assert "find_max_context_len memrift_async" in text
        assert "checkpoint.save" not in text
        assert "checkpoints" not in text


def test_memrift_search_starts_at_120_percent_of_lora_max():
    for model in ("aquila", "llama8b"):
        text = script(model)
        assert "MEMRIFT_SEARCH_START_FILE" in text
        assert "LoRA context search did not find any runnable sequence length" in text
        assert "math.ceil(lora_max * 1.2)" in text
        assert '"$MEMRIFT_SEARCH_START"' in text
        assert '"$CONTEXT_SEARCH_START"' in text
        assert text.index('"$CONTEXT_SEARCH_START"') < text.rindex('"$MEMRIFT_SEARCH_START"')


def test_context_search_writes_root_markdown_summary():
    for model in ("aquila", "llama8b"):
        text = script(model)
        assert 'METRICS_MD="${METRICS_MD:-/share/project/mengyc/MemRift_metrics.md}"' in text
        assert '"$METRICS_MD"' in text
        assert "# MemRift Metrics" in text
        assert "train_context_gain" in text


def test_context_search_uses_requested_two_stage_defaults():
    for model in ("aquila", "llama8b"):
        text = script(model)
        assert 'CONTEXT_TRAIN_ITERS="${CONTEXT_TRAIN_ITERS:-2}"' in text
        assert 'CONTEXT_SEARCH_START="${CONTEXT_SEARCH_START:-8192}"' in text
        assert 'CONTEXT_SEARCH_STEP="${CONTEXT_SEARCH_STEP:-1024}"' in text
        assert 'CONTEXT_SEARCH_FINE_STEP="${CONTEXT_SEARCH_FINE_STEP:-128}"' in text
        assert 'CONTEXT_SEARCH_CAP="${CONTEXT_SEARCH_CAP:-131072}"' in text


def test_context_search_success_requires_clean_exit_and_fine_boundary():
    assert "MEMRIFT_DISABLE_FINAL_CHECKPOINT=1" in COMMON
    training = (ROOT / "flagscale/train/megatron/training/training.py").read_text(encoding="utf-8")
    assert 'os.environ.get("MEMRIFT_DISABLE_FINAL_CHECKPOINT") != "1"' in training
    assert "+train.system.checkpoint.save=null" in COMMON
    assert "+train.system.checkpoint.load=null" in COMMON
    assert "train.system.checkpoint.save_interval=1000000000" in COMMON
    assert "wall_time_file" in COMMON
    assert "launcher_return_code" in COMMON
    assert "data.get(\"launcher_return_code\") == 0" in COMMON
    assert "local fine_step_len=\"${CONTEXT_SEARCH_FINE_STEP:-128}\"" in COMMON
    assert "coarse linear probing" in COMMON
    assert "seq=\"$start_len\"" in COMMON
    assert "seq=\"$((seq + step_len))\"" in COMMON
    assert "fine probing" in COMMON
    assert "fine_index" in COMMON
    assert '"phase": phase' in COMMON
