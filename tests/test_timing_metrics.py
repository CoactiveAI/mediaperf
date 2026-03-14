"""Tests for timing metrics calculation."""

import pytest

from benchmarks.metrics.timing_metrics import TimingMetricCalculator
from tests.fixtures.metrics_data import create_inference_results_with_usage, write_to_jsonl


@pytest.mark.fast
def test_hand_calculated_timing_statistics(tmp_path):
    """
    Test timing statistics with both real and fallback percentile cases.

    Test data (10 values): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    Calculated:
        mean = (1+2+...+10) / 10 = 5.5
        median = (5 + 6) / 2 = 5.5
        min = 1.0, max = 10.0
        p90 = 9.9 (real calculation, has ≥10 points)
        p95 = 10.0 (fallback to max, needs ≥20 points)
        p99 = 10.0 (fallback to max, needs ≥100 points)
    """
    timing_per_video = {f"vid_{i}": {"api_call_time_seconds": float(i)} for i in range(1, 11)}

    results = create_inference_results_with_usage(list(timing_per_video.keys()), timing_per_video=timing_per_video)

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = TimingMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(results_file))

    api_timing = metrics["api_call_timing"]
    assert api_timing["avg_seconds"] == 5.5
    assert api_timing["median_seconds"] == 5.5
    assert api_timing["min_seconds"] == 1.0
    assert api_timing["max_seconds"] == 10.0
    assert api_timing["p90_seconds"] == 9.9
    assert api_timing["p95_seconds"] == 10.0
    assert api_timing["p99_seconds"] == 10.0


@pytest.mark.fast
def test_timing_mean_median_min_max(tmp_path):
    """Test basic timing statistics."""
    timing_per_video = {
        "vid_1": {"api_call_time_seconds": 2.5, "preprocessing_time_seconds": 0.5},
        "vid_2": {"api_call_time_seconds": 3.0, "preprocessing_time_seconds": 0.6},
        "vid_3": {"api_call_time_seconds": 1.5, "preprocessing_time_seconds": 0.4},
    }

    results = create_inference_results_with_usage(list(timing_per_video.keys()), timing_per_video=timing_per_video)

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = TimingMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(results_file))

    api_timing = metrics["api_call_timing"]
    assert "avg_seconds" in api_timing
    assert "median_seconds" in api_timing
    assert "min_seconds" in api_timing
    assert "max_seconds" in api_timing
    assert api_timing["min_seconds"] == 1.5
    assert api_timing["max_seconds"] == 3.0


@pytest.mark.fast
def test_per_iteration_timing_grouping(tmp_path):
    """Test that timing is grouped by iteration."""
    timing_per_video = {
        "vid_1": {"api_call_time_seconds": 2.0},
        "vid_2": {"api_call_time_seconds": 2.5},
        "vid_3": {"api_call_time_seconds": 3.0},
        "vid_4": {"api_call_time_seconds": 3.5},
    }
    iteration_per_video = {
        "vid_1": 1,
        "vid_2": 1,
        "vid_3": 2,
        "vid_4": 2,
    }

    results = create_inference_results_with_usage(
        list(timing_per_video.keys()), timing_per_video=timing_per_video, iteration_per_video=iteration_per_video
    )

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = TimingMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(results_file))

    assert "per_iteration_timing" in metrics
    assert "iteration_1" in metrics["per_iteration_timing"]
    assert "iteration_2" in metrics["per_iteration_timing"]


@pytest.mark.fast
def test_single_data_point_timing(tmp_path):
    """Test timing metrics with single inference (stdev should be 0)."""
    timing_per_video = {"vid_1": {"api_call_time_seconds": 2.5}}

    results = create_inference_results_with_usage(["vid_1"], timing_per_video=timing_per_video)

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = TimingMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(results_file))

    api_timing = metrics["api_call_timing"]
    assert api_timing["avg_seconds"] == 2.5
    assert api_timing["median_seconds"] == 2.5
    assert api_timing["min_seconds"] == 2.5
    assert api_timing["max_seconds"] == 2.5


@pytest.mark.fast
def test_total_task_time_included(tmp_path):
    """Test that total task time is included when provided."""
    timing_per_video = {"vid_1": {"api_call_time_seconds": 1.0}}

    results = create_inference_results_with_usage(["vid_1"], timing_per_video=timing_per_video)

    results_file = tmp_path / "results.jsonl"
    write_to_jsonl(results, results_file)

    calculator = TimingMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(results_file), total_task_time=100.0)

    assert metrics["total_task_time_seconds"] == 100.0
