import json

import pytest

from scripts.metrics.benchmark_tasks import (
    BenchmarkExample,
    load_task_examples,
    score_choices,
    score_generation,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_gsm8k_examples_use_cot_fewshot_and_limit(tmp_path):
    path = tmp_path / "gsm8k.jsonl"
    write_jsonl(path, [{"question": "What is 1 + 1?", "answer": "work\n#### 2"}])
    examples = load_task_examples("gsm8k_cot", 1, gsm8k_path=path)
    assert len(examples) == 1
    assert examples[0].sample_id == "gsm8k_cot:0"
    assert examples[0].prompt.count("Q:") == 9
    assert examples[0].prompt.endswith("Q: What is 1 + 1?\nA:")
    assert examples[0].target == "2"


def test_gsm8k_scoring_extracts_last_number():
    example = BenchmarkExample("gsm8k_cot:0", "prompt", "1,234")
    result = score_generation(example, "Reasoning. The answer is $1,234.")
    assert result.correct is True
    assert result.prediction == "1234"


def test_hellaswag_uses_length_normalized_loglikelihood(tmp_path):
    path = tmp_path / "hellaswag.jsonl"
    write_jsonl(
        path,
        [{"activity_label": "Cooking", "ctx_a": "A person cooks.", "ctx_b": "they", "endings": ["a", "bb", "ccc", "dddd"], "label": 1}],
    )
    example = load_task_examples("hellaswag", 1, hellaswag_path=path)[0]
    assert example.prompt == "Cooking: A person cooks. They"
    assert example.choices == ("a", "bb", "ccc", "dddd")
    result = score_choices(example, [-1.0, -2.0, -1.2, -4.0], [1, 4, 3, 4])
    assert result.prediction == 2
    assert result.correct is False


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_fails(tmp_path, limit):
    with pytest.raises(ValueError, match="limit"):
        load_task_examples("gsm8k_cot", limit, gsm8k_path=tmp_path / "missing")


def test_wrong_choice_count_fails():
    example = BenchmarkExample("hellaswag:0", "prompt", 0, ("a", "b", "c", "d"))
    with pytest.raises(ValueError, match="four"):
        score_choices(example, [-1.0], [1])
