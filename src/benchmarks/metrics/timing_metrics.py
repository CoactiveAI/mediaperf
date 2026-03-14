import json
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MetricCalculator


class TimingMetricCalculator(MetricCalculator):
    """Metric calculator for timing analysis."""

    def __init__(self, total_task_time: Optional[float] = None):
        """
        Args:
            total_task_time: Total task execution time in seconds (optional)
        """
        self.total_task_time = total_task_time

    def calculate(
        self,
        predictions_path: str,
        ground_truth_path: str = None,
        total_task_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate timing metrics from inference results.

        Args:
            predictions_path: Path to predictions JSONL file with timing data
            ground_truth_path: Not used for timing metrics (optional)
            total_task_time: Total task time in seconds (overrides init value if provided)

        Returns:
            Dictionary containing timing metrics
        """
        # Use provided task time or fall back to init value
        task_time = total_task_time if total_task_time is not None else self.total_task_time

        return _calculate_timing_metrics_impl(
            results_path=predictions_path,
            total_task_time=task_time,
        )


def _calculate_timing_metrics_impl(results_path: str, total_task_time: Optional[float] = None) -> Dict:
    """
    Internal implementation for calculating timing metrics.

    Args:
        results_path: Path to JSONL file with inference results
        total_task_time: Total task execution time in seconds (optional)

    Returns:
        Dictionary with timing metrics
    """
    logger.info(f"Calculating timing metrics from {results_path}")

    # Load all inference results
    inferences = []
    with open(results_path, "r") as f:
        for line in f:
            inferences.append(json.loads(line))

    num_inferences = len(inferences)
    logger.info(f"   Loaded {num_inferences} inferences")

    # Collect timing data
    api_call_times = []
    preprocessing_times = []
    total_inference_times = []
    overhead_times = []

    # Per-iteration data (for cost benchmark tasks)
    per_iteration_data = defaultdict(
        lambda: {
            "api_call_times": [],
            "preprocessing_times": [],
            "total_inference_times": [],
            "overhead_times": [],
        }
    )

    # Per-video data
    per_video_timing = []

    for inf in inferences:
        # Extract timing fields
        api_time = inf.get("api_call_time_seconds")
        preprocessing_time = inf.get("preprocessing_time_seconds")
        total_time = inf.get("total_inference_time_seconds")
        overhead_time = inf.get("overhead_time_seconds")

        # Collect for overall aggregation
        if api_time is not None:
            api_call_times.append(api_time)
        if preprocessing_time is not None:
            preprocessing_times.append(preprocessing_time)
        if total_time is not None:
            total_inference_times.append(total_time)
        if overhead_time is not None:
            overhead_times.append(overhead_time)

        # Store per-video timing
        per_video_timing.append(
            {
                "video_id": inf.get("video_id"),
                "api_call_time_seconds": api_time,
                "preprocessing_time_seconds": preprocessing_time,
                "total_inference_time_seconds": total_time,
                "overhead_time_seconds": overhead_time,
            }
        )

        # Group by iteration if present (for cost benchmark tasks)
        iteration = inf.get("iteration_number")
        if iteration is not None:
            if api_time is not None:
                per_iteration_data[iteration]["api_call_times"].append(api_time)
            if preprocessing_time is not None:
                per_iteration_data[iteration]["preprocessing_times"].append(preprocessing_time)
            if total_time is not None:
                per_iteration_data[iteration]["total_inference_times"].append(total_time)
            if overhead_time is not None:
                per_iteration_data[iteration]["overhead_times"].append(overhead_time)

    # Calculate aggregate statistics
    def calc_stats(times: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of times."""
        if not times:
            return {}
        return {
            "total_seconds": sum(times),
            "avg_seconds": statistics.mean(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "median_seconds": statistics.median(times),
            "p90_seconds": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
            "p95_seconds": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "p99_seconds": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
        }

    # Build metrics dict
    metrics = {
        "num_inferences": num_inferences,
        "api_call_timing": calc_stats(api_call_times),
        "total_inference_timing": calc_stats(total_inference_times),
        "overhead_timing": calc_stats(overhead_times),
    }

    # Add preprocessing timing if applicable
    if preprocessing_times:
        metrics["preprocessing_timing"] = {
            "num_with_preprocessing": len(preprocessing_times),
            **calc_stats(preprocessing_times),
        }

    # Add task-level timing if provided
    if total_task_time is not None:
        metrics["total_task_time_seconds"] = total_task_time

    # Add per-iteration breakdown if available (for cost benchmark)
    if per_iteration_data:
        per_iteration_timing = {}
        for iteration in sorted(per_iteration_data.keys()):
            iter_data = per_iteration_data[iteration]
            per_iteration_timing[f"iteration_{iteration}"] = {
                "num_inferences": len(iter_data["api_call_times"]),
                "api_call_timing": calc_stats(iter_data["api_call_times"]),
                "total_inference_timing": calc_stats(iter_data["total_inference_times"]),
                "overhead_timing": calc_stats(iter_data["overhead_times"]),
            }
            if iter_data["preprocessing_times"]:
                per_iteration_timing[f"iteration_{iteration}"]["preprocessing_timing"] = calc_stats(
                    iter_data["preprocessing_times"]
                )

        metrics["per_iteration_timing"] = per_iteration_timing

    # Add per-video timing
    metrics["per_video_timing"] = per_video_timing

    logger.success("   Timing metrics calculated successfully")
    return metrics
