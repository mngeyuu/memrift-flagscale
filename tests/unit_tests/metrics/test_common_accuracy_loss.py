import pytest

from scripts.metrics.common_accuracy_loss import compare_loss


def metrics(loss, *, log_exists=True, failed=False):
    return {
        "log_exists": log_exists,
        "failed_pattern_found": failed,
        "final_lm_loss": loss,
    }


def test_relative_lm_loss_degradation_at_threshold_passes():
    result = compare_loss("m", "Model", metrics(2.0), metrics(2.02))

    assert result["relative_loss_degradation_percent"] == pytest.approx(1.0)
    assert result["pass"] is True


def test_higher_than_one_percent_loss_degradation_fails():
    result = compare_loss("m", "Model", metrics(2.0), metrics(2.021))

    assert result["relative_loss_degradation_percent"] == pytest.approx(1.05)
    assert result["pass"] is False


def test_lower_memrift_loss_passes():
    result = compare_loss("m", "Model", metrics(2.0), metrics(1.9))

    assert result["relative_loss_degradation_percent"] == pytest.approx(-5.0)
    assert result["pass"] is True


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        (metrics(None), metrics(2.0), "numeric"),
        (metrics(0.0), metrics(2.0), "positive"),
        (metrics(2.0, log_exists=False), metrics(2.0), "missing"),
        (metrics(2.0), metrics(2.0, failed=True), "error pattern"),
    ],
)
def test_invalid_training_metrics_are_rejected(baseline, candidate, message):
    with pytest.raises(ValueError, match=message):
        compare_loss("m", "Model", baseline, candidate)
