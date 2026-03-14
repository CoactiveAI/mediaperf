import json
from collections import defaultdict
from typing import Any, Dict

from loguru import logger

from .base import MetricCalculator


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> float:
    """
    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        input_price_per_1m: Input price per 1M tokens (USD)
        output_price_per_1m: Output price per 1M tokens (USD)

    Returns:
        Estimated cost in USD
    """
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_1m
    output_cost = (completion_tokens / 1_000_000) * output_price_per_1m
    return input_cost + output_cost


class CostMetricCalculator(MetricCalculator):
    """Metric calculator for cost analysis tasks."""

    def __init__(self, input_price_per_1m: float, output_price_per_1m: float):
        """
        Args:
            input_price_per_1m: Input token price per 1M tokens (USD)
            output_price_per_1m: Output token price per 1M tokens (USD)
        """
        self.input_price_per_1m = input_price_per_1m
        self.output_price_per_1m = output_price_per_1m

    def calculate(
        self,
        predictions_path: str,
        ground_truth_path: str = None,
    ) -> Dict[str, Any]:
        """
        Calculate cost metrics from inference results.

        Args:
            predictions_path: Path to predictions JSONL file with token usage
            ground_truth_path: Not used for cost metrics (optional)

        Returns:
            Dictionary containing cost metrics
        """
        return _calculate_cost_metrics_impl(
            results_path=predictions_path,
            input_price_per_1m=self.input_price_per_1m,
            output_price_per_1m=self.output_price_per_1m,
        )


def _calculate_cost_metrics_impl(results_path: str, input_price_per_1m: float, output_price_per_1m: float) -> Dict:
    """
    Internal implementation for calculating cost metrics.

    Args:
        results_path: Path to JSONL file with inference results
        input_price_per_1m: Input token price per 1M tokens (USD)
        output_price_per_1m: Output token price per 1M tokens (USD)

    Returns:
        Dictionary with cost metrics
    """
    logger.info(f"Calculating cost metrics from {results_path}")

    results = []
    with open(results_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        logger.warning("No results found in file")
        return {}

    # Get model ID from first result
    model_id = results[0].get("model", "unknown")
    logger.info(f"Model: {model_id}")
    logger.info(f"Pricing: ${input_price_per_1m}/1M input, ${output_price_per_1m}/1M output")

    # Calculate per-inference costs (micro-level)
    per_inference_costs = []
    cost_by_iteration = defaultdict(float)
    cost_by_video = defaultdict(float)

    # Per-iteration detailed stats
    iteration_stats = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "num_inferences": 0}
    )

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for result in results:
        prompt_tokens = result.get("prompt_tokens", 0)
        completion_tokens = result.get("completion_tokens", 0)
        video_id = result.get("video_id")
        iteration = result.get("iteration_number")

        # Calculate cost for this inference
        cost = calculate_cost(prompt_tokens, completion_tokens, input_price_per_1m, output_price_per_1m)

        per_inference_costs.append(
            {
                "video_id": video_id,
                "iteration": iteration,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": round(cost, 6),
            }
        )

        # Accumulate totals
        total_cost += cost
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += prompt_tokens + completion_tokens

        # Group by iteration and video
        if iteration:
            cost_by_iteration[iteration] += cost
            iteration_stats[iteration]["prompt_tokens"] += prompt_tokens
            iteration_stats[iteration]["completion_tokens"] += completion_tokens
            iteration_stats[iteration]["total_tokens"] += prompt_tokens + completion_tokens
            iteration_stats[iteration]["cost"] += cost
            iteration_stats[iteration]["num_inferences"] += 1

        if video_id:
            cost_by_video[video_id] += cost

    # Calculate aggregate metrics (macro-level)
    num_inferences = len(results)
    num_videos = len(set(r.get("video_id") for r in results if r.get("video_id")))
    num_iterations = len(set(r.get("iteration_number") for r in results if r.get("iteration_number")))

    # Build per-iteration detailed metrics
    per_iteration_stats = {}
    for iteration, stats in sorted(iteration_stats.items()):
        num_inf = stats["num_inferences"]
        per_iteration_stats[str(iteration)] = {
            "total_cost": round(stats["cost"], 4),
            "total_inferences": num_inf,
            "total_prompt_tokens": stats["prompt_tokens"],
            "total_completion_tokens": stats["completion_tokens"],
            "total_tokens": stats["total_tokens"],
            "cost_per_inference": round(stats["cost"] / num_inf, 6) if num_inf > 0 else 0.0,
            "avg_prompt_tokens": round(stats["prompt_tokens"] / num_inf, 1) if num_inf > 0 else 0,
            "avg_completion_tokens": round(stats["completion_tokens"] / num_inf, 1) if num_inf > 0 else 0,
        }

    metrics = {
        "model": model_id,
        "pricing": {
            "input_per_1m_tokens": input_price_per_1m,
            "output_per_1m_tokens": output_price_per_1m,
        },
        # Totals
        "total_cost": round(total_cost, 4),
        "total_inferences": num_inferences,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        # Averages
        "cost_per_inference": round(total_cost / num_inferences, 6) if num_inferences > 0 else 0.0,
        "cost_per_video": round(total_cost / num_videos, 4) if num_videos > 0 else 0.0,
        "cost_per_iteration": round(total_cost / num_iterations, 4) if num_iterations > 0 else 0.0,
        "avg_prompt_tokens": round(total_prompt_tokens / num_inferences, 1) if num_inferences > 0 else 0,
        "avg_completion_tokens": round(total_completion_tokens / num_inferences, 1) if num_inferences > 0 else 0,
        # Breakdowns
        "cost_by_iteration": {str(k): round(v, 4) for k, v in sorted(cost_by_iteration.items())},
        "per_iteration_stats": per_iteration_stats,
        "num_videos": num_videos,
        "num_iterations": num_iterations,
        # Micro-level (per-inference)
        "per_inference_costs": per_inference_costs,
    }

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("COST METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"Total inferences: {num_inferences}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Cost per video: ${metrics['cost_per_video']:.4f}")
    logger.info(f"Cost per iteration: ${metrics['cost_per_iteration']:.4f}")
    avg_input = metrics["avg_prompt_tokens"]
    avg_output = metrics["avg_completion_tokens"]
    logger.info(f"Avg tokens per inference: {avg_input:.0f} input + {avg_output:.0f} output")
    logger.info("=" * 60)

    return metrics
