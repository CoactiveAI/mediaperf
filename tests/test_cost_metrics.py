"""Tests for cost metrics calculation."""

import pytest

from benchmarks.metrics.cost_metrics import CostMetricCalculator
from tests.fixtures.metrics_data import create_inference_results_with_usage, write_to_jsonl


@pytest.mark.fast
def test_hand_calculated_cost(tmp_path):
    """
    Test cost calculation with hand-calculated expectations.

    Test data:
        vid_1: 1,000,000 input tokens, 100,000 output tokens
        vid_2: 500,000 input tokens, 50,000 output tokens

    Pricing: $2.50 per 1M input, $10.00 per 1M output

    Expected calculation:
        vid_1: (1M / 1M * $2.50) + (100K / 1M * $10.00) = $2.50 + $1.00 = $3.50
        vid_2: (500K / 1M * $2.50) + (50K / 1M * $10.00) = $1.25 + $0.50 = $1.75
        Total: $3.50 + $1.75 = $5.25
    """
    tokens_per_video = {
        "vid_1": {"input_tokens": 1_000_000, "output_tokens": 100_000},
        "vid_2": {"input_tokens": 500_000, "output_tokens": 50_000},
    }

    results = create_inference_results_with_usage(["vid_1", "vid_2"], tokens_per_video=tokens_per_video)

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = CostMetricCalculator(input_price_per_1m=2.50, output_price_per_1m=10.00)
    metrics = calculator.calculate(predictions_path=str(results_file))

    assert abs(metrics["total_cost"] - 5.25) < 0.01
    assert metrics["total_tokens"] == 1_650_000
    assert metrics["total_prompt_tokens"] == 1_500_000
    assert metrics["total_completion_tokens"] == 150_000


@pytest.mark.fast
def test_per_video_cost_aggregation(tmp_path):
    """Test that costs are aggregated correctly per video."""
    tokens_per_video = {
        "vid_1": {"input_tokens": 1000, "output_tokens": 500},
        "vid_2": {"input_tokens": 2000, "output_tokens": 1000},
        "vid_3": {"input_tokens": 1500, "output_tokens": 750},
    }

    results = create_inference_results_with_usage(["vid_1", "vid_2", "vid_3"], tokens_per_video=tokens_per_video)

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = CostMetricCalculator(input_price_per_1m=2.00, output_price_per_1m=10.00)
    metrics = calculator.calculate(predictions_path=str(results_file))

    assert metrics["total_inferences"] == 3
    assert metrics["total_prompt_tokens"] == 4500
    assert metrics["total_completion_tokens"] == 2250


@pytest.mark.fast
def test_per_iteration_cost_grouping(tmp_path):
    """Test that costs are grouped by iteration for cost benchmark tasks."""
    tokens_per_video = {
        "vid_1_iter1": {"input_tokens": 1000, "output_tokens": 100},
        "vid_2_iter1": {"input_tokens": 1000, "output_tokens": 100},
        "vid_1_iter2": {"input_tokens": 2000, "output_tokens": 200},
        "vid_2_iter2": {"input_tokens": 2000, "output_tokens": 200},
    }
    iteration_per_video = {
        "vid_1_iter1": 1,
        "vid_2_iter1": 1,
        "vid_1_iter2": 2,
        "vid_2_iter2": 2,
    }

    results = create_inference_results_with_usage(
        list(tokens_per_video.keys()), tokens_per_video=tokens_per_video, iteration_per_video=iteration_per_video
    )

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = CostMetricCalculator(input_price_per_1m=1.00, output_price_per_1m=1.00)
    metrics = calculator.calculate(predictions_path=str(results_file))

    assert "per_iteration_stats" in metrics
    assert "1" in metrics["per_iteration_stats"]
    assert "2" in metrics["per_iteration_stats"]


@pytest.mark.fast
def test_missing_token_fields(tmp_path):
    """Test that missing token fields default to 0."""
    results = [
        {"video_id": "vid_1", "model": "test"},
        {"video_id": "vid_2", "model": "test", "prompt_tokens": 1000},
    ]

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = CostMetricCalculator(input_price_per_1m=2.00, output_price_per_1m=10.00)
    metrics = calculator.calculate(predictions_path=str(results_file))

    assert metrics["total_prompt_tokens"] == 1000
    assert metrics["total_completion_tokens"] == 0
