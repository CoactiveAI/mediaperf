"""Tests for LLM judge metrics calculation."""

import pytest

from benchmarks.metrics.llm_judge_metrics import LLMJudgeMetricCalculator
from tests.fixtures.evaluation_data import create_evaluation_results, write_evaluations_to_jsonl


@pytest.mark.fast
def test_judge_metrics_sum_aggregation(tmp_path):
    """Test sum aggregation: grand_total = sum of overall_scores."""
    evaluations = create_evaluation_results(num_evaluations=3, aggregation_method="sum")

    eval_file = tmp_path / "evaluations.jsonl"
    write_evaluations_to_jsonl(evaluations, eval_file)

    calculator = LLMJudgeMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(eval_file))

    assert metrics["aggregation_method"] == "sum"
    expected_grand_total = sum(e["overall_score"] for e in evaluations)
    assert metrics["grand_total"] == round(expected_grand_total, 2)
    assert metrics["total_evaluations"] == 3


@pytest.mark.fast
def test_judge_metrics_average_aggregation(tmp_path):
    """Test average aggregation: grand_total = average of overall_scores."""
    evaluations = create_evaluation_results(num_evaluations=5, aggregation_method="average")

    eval_file = tmp_path / "evaluations.jsonl"
    write_evaluations_to_jsonl(evaluations, eval_file)

    calculator = LLMJudgeMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(eval_file))

    assert metrics["aggregation_method"] == "average"
    overall_scores = [e["overall_score"] for e in evaluations]
    expected_grand_total = sum(overall_scores) / len(overall_scores)
    assert metrics["grand_total"] == round(expected_grand_total, 2)


@pytest.mark.fast
def test_judge_metrics_criterion_statistics(tmp_path):
    """Test per-criterion statistics: mean, median, stdev, min, max, count."""
    evaluations = create_evaluation_results(num_evaluations=5, aggregation_method="sum")

    eval_file = tmp_path / "evaluations.jsonl"
    write_evaluations_to_jsonl(evaluations, eval_file)

    calculator = LLMJudgeMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(eval_file))

    assert "criterion_metrics" in metrics
    criterion_metrics = metrics["criterion_metrics"]
    assert len(criterion_metrics) == 4
    assert "Factual Accuracy" in criterion_metrics

    factual_metrics = criterion_metrics["Factual Accuracy"]
    assert "mean" in factual_metrics
    assert "median" in factual_metrics
    assert "stdev" in factual_metrics
    assert "min" in factual_metrics
    assert "max" in factual_metrics
    assert "count" in factual_metrics
    assert factual_metrics["count"] == 5


@pytest.mark.fast
def test_judge_metrics_excludes_failures(tmp_path):
    """Test that failed evaluations (with 'error' field) are excluded."""
    evaluations = create_evaluation_results(num_evaluations=3, include_failures=2, aggregation_method="sum")

    eval_file = tmp_path / "evaluations.jsonl"
    write_evaluations_to_jsonl(evaluations, eval_file)

    calculator = LLMJudgeMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(eval_file))

    assert metrics["total_evaluations"] == 3

    successful_evals = [e for e in evaluations if "error" not in e]
    expected_grand_total = sum(e["overall_score"] for e in successful_evals)
    assert metrics["grand_total"] == round(expected_grand_total, 2)


@pytest.mark.fast
def test_judge_metrics_empty_file(tmp_path):
    """Test handling of empty evaluation file."""
    eval_file = tmp_path / "empty.jsonl"
    eval_file.write_text("")

    calculator = LLMJudgeMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(eval_file))

    assert metrics["total_evaluations"] == 0
    assert "error" in metrics


@pytest.mark.fast
def test_judge_metrics_all_failures(tmp_path):
    """Test handling when all evaluations failed."""
    evaluations = create_evaluation_results(num_evaluations=0, include_failures=3, aggregation_method="sum")

    eval_file = tmp_path / "all_failed.jsonl"
    write_evaluations_to_jsonl(evaluations, eval_file)

    calculator = LLMJudgeMetricCalculator()
    metrics = calculator.calculate(predictions_path=str(eval_file))

    assert metrics["total_evaluations"] == 0
    assert "error" in metrics
