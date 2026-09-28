import pytest

from scripts.metrics.benchmark_accuracy_result import aggregate_results, compare_task


def task_result(task, accuracy, samples=100):
    return {
        "task": task,
        "accuracy": accuracy,
        "num_samples": samples,
        "num_correct": round(accuracy * samples),
    }


def test_relative_drop_uses_baseline_accuracy():
    result = compare_task(
        "gsm8k_cot",
        task_result("gsm8k_cot", 0.45),
        task_result("gsm8k_cot", 0.4455),
    )
    assert result["absolute_drop_percentage_points"] == pytest.approx(0.45)
    assert result["relative_accuracy_drop_percent"] == pytest.approx(1.0)
    assert result["pass"] is True


def test_candidate_improvement_passes_with_negative_drop():
    result = compare_task(
        "hellaswag",
        task_result("hellaswag", 0.50),
        task_result("hellaswag", 0.51),
    )
    assert result["relative_accuracy_drop_percent"] == pytest.approx(-2.0)
    assert result["pass"] is True


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        (task_result("gsm8k_cot", 0.0), task_result("gsm8k_cot", 0.0), "positive"),
        (task_result("gsm8k_cot", 0.5, 0), task_result("gsm8k_cot", 0.5), "num_samples"),
        (task_result("gsm8k_cot", 0.5), task_result("hellaswag", 0.5), "task"),
    ],
)
def test_invalid_task_results_fail(baseline, candidate, message):
    with pytest.raises(ValueError, match=message):
        compare_task("gsm8k_cot", baseline, candidate)


def test_both_tasks_must_pass():
    baseline = {
        "tasks": {
            "gsm8k_cot": task_result("gsm8k_cot", 0.5),
            "hellaswag": task_result("hellaswag", 0.5),
        }
    }
    candidate = {
        "tasks": {
            "gsm8k_cot": task_result("gsm8k_cot", 0.5),
            "hellaswag": task_result("hellaswag", 0.49),
        }
    }
    result = aggregate_results("m", "Model", baseline, candidate)
    assert result["tasks"]["gsm8k_cot"]["pass"] is True
    assert result["tasks"]["hellaswag"]["pass"] is False
    assert result["pass"] is False


def test_single_task_comparison_is_allowed():
    baseline = {
        "tasks": {
            "gsm8k_cot": task_result("gsm8k_cot", 0.5),
            "hellaswag": task_result("hellaswag", 0.5),
        }
    }
    candidate = {
        "tasks": {
            "gsm8k_cot": task_result("gsm8k_cot", 0.1),
            "hellaswag": task_result("hellaswag", 0.497),
        }
    }
    result = aggregate_results("m", "Model", baseline, candidate, tasks=("hellaswag",))
    assert result["selected_tasks"] == ["hellaswag"]
    assert set(result["tasks"]) == {"hellaswag"}
    assert result["pass"] is True


def test_missing_task_fails():
    baseline = {"tasks": {"gsm8k_cot": task_result("gsm8k_cot", 0.5)}}
    candidate = {"tasks": {"gsm8k_cot": task_result("gsm8k_cot", 0.5)}}
    with pytest.raises(ValueError, match="contain"):
        aggregate_results("m", "Model", baseline, candidate)
