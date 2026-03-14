"""Test fixtures for LLM judge evaluation data."""

from pathlib import Path
from typing import Dict, List

import jsonlines


def create_evaluation_results(
    num_evaluations: int = 5,
    criteria: List[str] = None,
    aggregation_method: str = "sum",
    include_failures: int = 0,
) -> List[Dict]:
    if criteria is None:
        criteria = [
            "Factual Accuracy",
            "Structural Alignment",
            "Completeness",
            "Objectivity & Language Style",
        ]

    evaluations = []

    for i in range(num_evaluations):
        criterion_scores = []
        for criterion in criteria:
            score = 3.0 + (i % 3)
            criterion_scores.append(
                {"criterion": criterion, "score": score, "reasoning": f"Test reasoning for {criterion}"}
            )

        scores_list = [cs["score"] for cs in criterion_scores]
        if aggregation_method == "sum":
            overall_score = sum(scores_list)
        else:
            overall_score = sum(scores_list) / len(scores_list)

        evaluation = {
            "video_id": f"vid_{i}",
            "model": "test-model",
            "criterion_scores": criterion_scores,
            "overall_score": overall_score,
            "aggregation_method": aggregation_method,
        }

        evaluations.append(evaluation)

    for i in range(include_failures):
        evaluations.append({"video_id": f"vid_failed_{i}", "error": "Evaluation failed"})

    return evaluations


def write_evaluations_to_jsonl(evaluations: List[Dict], output_path: Path):
    """Write evaluation results to JSONL file."""
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(evaluations)
