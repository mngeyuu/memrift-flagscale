import json

import pytest

from scripts.metrics.benchmark_accuracy import _enabled_tasks, evaluate_examples, write_benchmark_result
from scripts.metrics.benchmark_tasks import BenchmarkExample
from scripts.metrics.megatron_benchmark_backend import _load_memrift_adapters, _uses_adapter_only_load, adapter_state_dict


class FakeBackend:
    def generate(self, prompts, max_new_tokens):
        return ["The answer is 2."] * len(prompts)

    def loglikelihood(self, requests):
        values = [(-4.0, 4), (-3.0, 3), (-1.0, 2), (-5.0, 5)]
        return values * (len(requests) // 4)


def test_gsm8k_runner_generates_and_scores():
    examples = [BenchmarkExample("gsm8k_cot:0", "Q: 1+1?\nA:", "2")]
    result = evaluate_examples("gsm8k_cot", examples, FakeBackend())
    assert result["num_samples"] == 1
    assert result["num_correct"] == 1
    assert result["accuracy"] == 1.0
    assert result["details"][0]["sample_id"] == "gsm8k_cot:0"


def test_hellaswag_runner_emits_four_requests_per_example():
    examples = [BenchmarkExample("hellaswag:0", "ctx", 2, ("a", "b", "c", "d"))]
    result = evaluate_examples("hellaswag", examples, FakeBackend())
    assert result["num_samples"] == 1
    assert result["num_correct"] == 1
    assert result["accuracy"] == 1.0


def test_backend_output_count_mismatch_fails():
    class BadBackend(FakeBackend):
        def generate(self, prompts, max_new_tokens):
            return []

    with pytest.raises(RuntimeError, match="returned"):
        evaluate_examples("gsm8k_cot", [BenchmarkExample("x", "p", "1")], BadBackend())


def test_result_is_written_only_with_both_tasks(tmp_path):
    output = tmp_path / "result.json"
    tasks = {
        "gsm8k_cot": {"task": "gsm8k_cot", "num_samples": 1, "num_correct": 1, "accuracy": 1.0},
        "hellaswag": {"task": "hellaswag", "num_samples": 1, "num_correct": 1, "accuracy": 1.0},
    }
    write_benchmark_result(output, "pure_lora", tasks, {"checkpoint": "/x"})
    assert json.loads(output.read_text())["variant"] == "pure_lora"

    write_benchmark_result(output, "pure_lora", {"gsm8k_cot": tasks["gsm8k_cot"]}, {})
    assert json.loads(output.read_text())["tasks"]["gsm8k_cot"]["accuracy"] == 1.0


def test_adapter_state_dict_drops_memrift_weight_placeholders():
    state = {
        "decoder.layer.to_wrap.weight": "placeholder",
        "decoder.layer.adapter.linear_in.weight": "adapter",
    }
    assert adapter_state_dict(state) == {"decoder.layer.adapter.linear_in.weight": "adapter"}


def test_adapter_state_dict_rejects_empty_state():
    with pytest.raises(ValueError, match="adapter"):
        adapter_state_dict({"decoder.layer.weight": "weight"})


def test_enabled_tasks_skip_zero_limits(monkeypatch):
    monkeypatch.setenv("GSM8K_LIMIT", "0")
    monkeypatch.setenv("HELLASWAG_LIMIT", "64")
    assert _enabled_tasks() == [("hellaswag", 64)]


def test_load_memrift_adapters_accepts_none_incompatible(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    iteration_dir = ckpt_dir / "iter_0000001" / "mp_rank_00"
    iteration_dir.mkdir(parents=True)
    (ckpt_dir / "latest_checkpointed_iteration.txt").write_text("1")

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            return {"model": {"decoder.layer.adapter.linear_in.weight": "adapter"}}

    class FakeModel:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = (state_dict, strict)
            return None

    model = FakeModel()
    _load_memrift_adapters(model, str(ckpt_dir), FakeTorch())
    assert model.loaded == ({"decoder.layer.adapter.linear_in.weight": "adapter"}, False)


def test_adapter_only_load_mode_includes_plain_lora_benchmark():
    assert _uses_adapter_only_load("memrift_async") is True
    assert _uses_adapter_only_load("lora_adapter_only") is True
    assert _uses_adapter_only_load("lora") is False
