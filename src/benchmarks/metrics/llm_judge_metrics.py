import statistics
from typing import Any, Dict

import jsonlines
from loguru import logger

from .base import MetricCalculator


class LLMJudgeMetricCalculator(MetricCalculator):
    """Calculate aggregate metrics from LLM judge evaluations."""

    def calculate(
        self,
        predictions_path: str,
        ground_truth_path: str = None,
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from LLM judge evaluation results.

        Args:
            predictions_path: Path to evaluations JSONL file (output from evaluation pipeline)
            ground_truth_path: Not used for this calculator (evaluations already contain comparisons)

        Returns:
            Dictionary with aggregated metrics per criterion and overall
        """
        logger.info(f"Calculating LLM judge metrics from {predictions_path}")

        # Load evaluation results
        evaluations = []
        with jsonlines.open(predictions_path) as reader:
            for obj in reader:
                # Skip failed evaluations
                if "error" in obj:
                    logger.warning(f"Skipping failed evaluation for video {obj.get('video_id')}")
                    continue
                evaluations.append(obj)

        if not evaluations:
            logger.warning("No valid evaluations found")
            return {"total_evaluations": 0, "error": "No valid evaluations found"}

        logger.info(f"Found {len(evaluations)} valid evaluations")

        # Collect scores per criterion
        criterion_scores_dict = {}
        overall_scores = []

        for evaluation in evaluations:
            # Collect criterion scores
            for criterion_score in evaluation.get("criterion_scores", []):
                criterion_name = criterion_score.get("criterion")
                score = criterion_score.get("score")

                if criterion_name not in criterion_scores_dict:
                    criterion_scores_dict[criterion_name] = []

                criterion_scores_dict[criterion_name].append(score)

            # Collect overall scores
            overall_score = evaluation.get("overall_score")
            if overall_score is not None:
                overall_scores.append(overall_score)

        # Detect aggregation method from first evaluation
        aggregation_method = evaluations[0].get("aggregation_method", "sum") if evaluations else "sum"

        # Calculate statistics for each criterion
        criterion_metrics = {}
        for criterion_name, scores in criterion_scores_dict.items():
            if scores:
                criterion_metrics[criterion_name] = {
                    "mean": round(statistics.mean(scores), 2),
                    "median": round(statistics.median(scores), 2),
                    "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }

        # Calculate overall statistics
        overall_metrics = {}
        if overall_scores:
            overall_metrics = {
                "mean": round(statistics.mean(overall_scores), 2),
                "median": round(statistics.median(overall_scores), 2),
                "stdev": round(statistics.stdev(overall_scores), 2) if len(overall_scores) > 1 else 0.0,
                "min": round(min(overall_scores), 2),
                "max": round(max(overall_scores), 2),
                "count": len(overall_scores),
            }

        # Calculate grand total based on aggregation method
        if overall_scores:
            if aggregation_method == "sum":
                grand_total = round(sum(overall_scores), 2)
            else:  # average
                grand_total = round(statistics.mean(overall_scores), 2)
        else:
            grand_total = 0.0

        # Build final metrics dict
        metrics = {
            "total_evaluations": len(evaluations),
            "aggregation_method": aggregation_method,
            "criterion_metrics": criterion_metrics,
            "overall_metrics": overall_metrics,
            "grand_total": grand_total,
        }

        logger.info(f"Aggregation method: {aggregation_method}")
        logger.info(f"Overall mean score: {overall_metrics.get('mean', 0):.2f}")
        logger.info(f"Grand total: {grand_total:.2f}")

        return metrics
