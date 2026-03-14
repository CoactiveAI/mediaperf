import time
from pathlib import Path

import jsonlines
from loguru import logger

from ..builders import (
    build_result_saver,
    build_result_writer,
)
from .summary_evaluation_pipeline import run_summary_evaluation_pipeline


def load_summaries_from_jsonl(file_path: str) -> list:
    """Load summaries from JSONL file."""
    summaries = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            summaries.append(obj)
    return summaries


def load_ground_truth_summaries(file_path: str) -> dict:
    """Load ground truth summaries into dict mapping video_id to summary."""
    ground_truth = {}
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            video_id = obj.get("video_id")
            summary = obj.get("summary", "")
            if video_id:
                ground_truth[video_id] = summary
    return ground_truth


def run_summary_evaluation(config: dict):
    """Run the summary evaluation task using LLM-as-a-judge."""

    # We'll need to build the judge later, but for now import here to avoid circular import
    from ..builders import build_llm_judge

    # Extract paths from config
    OUTPUTS_DIR = Path(config["paths"]["outputs_dir"])

    # Create output directory if it doesn't exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get judge model name and timestamp for consistent file naming
    JUDGE_MODEL_NAME = config["pipeline"]["judge"]["config"].get(
        "model_name", config["pipeline"]["judge"]["config"].get("model_id", "Unknown")
    )
    timestamp = config.get("_timestamp", time.strftime("%y%m%d_%H%M%S"))

    logger.info("=" * 60)
    logger.info(f"SUMMARY EVALUATION - {JUDGE_MODEL_NAME}")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("[1/5] Loading summaries and ground truth...")
    summaries_path = config["paths"]["summaries_file"]
    ground_truth_path = config["paths"]["ground_truth_file"]

    summaries_data = load_summaries_from_jsonl(summaries_path)
    ground_truth_data = load_ground_truth_summaries(ground_truth_path)

    logger.info(f"   Loaded {len(summaries_data)} summaries from {summaries_path}")
    logger.info(f"   Loaded {len(ground_truth_data)} ground truth summaries from {ground_truth_path}")

    # Filter to only summaries that have ground truth (intersection)
    ground_truth_video_ids = set(ground_truth_data.keys())
    summaries_data_filtered = [s for s in summaries_data if s.get("video_id") in ground_truth_video_ids]
    skipped_no_gt = len(summaries_data) - len(summaries_data_filtered)

    if skipped_no_gt > 0:
        logger.info(
            f"   Filtered to {len(summaries_data_filtered)} summaries with ground truth "
            f"(skipped {skipped_no_gt} without ground truth)"
        )

    summaries_data = summaries_data_filtered

    # Step 2: Load evaluation criteria and aggregation method
    logger.info("[2/5] Loading evaluation criteria...")
    criteria = config["evaluation"]["criteria"]
    aggregation_method = config["evaluation"].get("aggregation_method", "sum")
    logger.info(f"   Evaluation criteria: {', '.join([c['name'] for c in criteria])}")
    logger.info(f"   Aggregation method: {aggregation_method}")

    # Step 3: Build judge model
    logger.info("[3/5] Building judge model...")
    judge = build_llm_judge(config["pipeline"]["judge"])
    logger.info(f"   Judge: {config['pipeline']['judge']['type']}")
    logger.info(f"   Model: {JUDGE_MODEL_NAME}")

    # Build result writer with timestamped filename
    evaluations_path = OUTPUTS_DIR / f"evaluation_{JUDGE_MODEL_NAME}_results_{timestamp}.jsonl"

    # Update config with timestamped path
    config["pipeline"]["result_writer"]["config"]["output_path"] = str(evaluations_path)
    result_writer = build_result_writer(config["pipeline"]["result_writer"])

    logger.info(f"   Result writer: {config['pipeline']['result_writer']['type']}")

    # Get usage tracking flag from config
    return_usage = config.get("tracking", {}).get("return_usage", False)
    progress_interval = config.get("tracking", {}).get("progress_interval", 100)
    logger.info(f"   Usage tracking: {'enabled' if return_usage else 'disabled'}")
    logger.info(f"   Progress logging interval: {progress_interval} summaries")

    # Step 4: Run evaluation
    logger.info(f"[4/5] Running evaluation on {len(summaries_data)} summaries...")
    task_start = time.time()
    run_summary_evaluation_pipeline(
        judge=judge,
        summaries_data=summaries_data,
        ground_truth_data=ground_truth_data,
        criteria=criteria,
        result_writer=result_writer,
        aggregation_method=aggregation_method,
        return_usage=return_usage,
        progress_interval=progress_interval,
    )
    task_time = time.time() - task_start
    logger.info(f"   Evaluations saved to: {evaluations_path}")
    logger.info(f"   Total task time: {task_time:.2f} seconds")

    # Step 5: Calculate aggregate metrics
    logger.info("[5/5] Calculating aggregate metrics...")
    from ..builders import build_metric_calculator

    # Build metric calculator
    metric_calculator = build_metric_calculator({"type": "llm_judge", "config": {}})

    # Calculate aggregate metrics from evaluations
    metrics = metric_calculator.calculate(
        predictions_path=str(evaluations_path),
        ground_truth_path=None,  # Not needed for judge metrics
    )

    # Save aggregate metrics
    result_saver = build_result_saver(config.get("result_saver", {"type": "json"}))
    metrics_path = result_saver.save(
        metrics, f"evaluation_{JUDGE_MODEL_NAME}_metrics", OUTPUTS_DIR, timestamp=timestamp
    )
    logger.info(f"   Aggregate metrics saved to: {metrics_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total evaluations: {metrics.get('total_evaluations', 0)}")
    logger.info(f"Aggregation method: {metrics.get('aggregation_method', 'sum')}")
    logger.info("")
    logger.info("Per-Criterion Metrics:")
    for criterion, criterion_stats in metrics.get("criterion_metrics", {}).items():
        logger.info(f"  {criterion}:")
        logger.info(f"    Mean: {criterion_stats.get('mean', 0):.2f}")
        logger.info(f"    Median: {criterion_stats.get('median', 0):.2f}")
        logger.info(f"    Std Dev: {criterion_stats.get('stdev', 0):.2f}")
        logger.info(f"    Min: {criterion_stats.get('min', 0)}")
        logger.info(f"    Max: {criterion_stats.get('max', 0)}")
    logger.info("")
    overall_metrics = metrics.get("overall_metrics", {})
    if overall_metrics:
        logger.info("Overall Score Metrics:")
        logger.info(f"  Mean: {overall_metrics.get('mean', 0):.2f}")
        logger.info(f"  Median: {overall_metrics.get('median', 0):.2f}")
        logger.info(f"  Std Dev: {overall_metrics.get('stdev', 0):.2f}")
        logger.info(f"  Min: {overall_metrics.get('min', 0):.2f}")
        logger.info(f"  Max: {overall_metrics.get('max', 0):.2f}")
    logger.info("")
    logger.info(f"Grand Total: {metrics.get('grand_total', 0):.2f}")

    # Calculate and save cost metrics if tracking is enabled and pricing configured
    if return_usage:
        pricing = config.get("pricing", {})
        input_price = pricing.get("input_per_1m_tokens")
        output_price = pricing.get("output_per_1m_tokens")

        if input_price is not None and output_price is not None:
            logger.info("")
            logger.info("Calculating cost metrics...")
            cost_calculator = build_metric_calculator(
                {
                    "type": "cost",
                    "config": {
                        "input_price_per_1m": input_price,
                        "output_price_per_1m": output_price,
                    },
                }
            )
            cost_metrics = cost_calculator.calculate(
                predictions_path=str(evaluations_path),
                ground_truth_path=None,
            )

            # Save cost metrics
            cost_metrics_path = result_saver.save(
                cost_metrics, f"evaluation_{JUDGE_MODEL_NAME}_cost", OUTPUTS_DIR, timestamp=timestamp
            )
            logger.info(f"   Cost metrics saved to: {cost_metrics_path}")

            # Print cost summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("COST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total cost: ${cost_metrics.get('total_cost', 0):.4f}")
            logger.info(f"Cost per evaluation: ${cost_metrics.get('cost_per_inference', 0):.6f}")
            logger.info(f"Total tokens: {cost_metrics.get('total_tokens', 0):,}")
            avg_prompt = cost_metrics.get("avg_prompt_tokens", 0)
            avg_completion = cost_metrics.get("avg_completion_tokens", 0)
            total_tokens = avg_prompt + avg_completion
            logger.info(
                f"Average tokens per evaluation: {avg_prompt:.1f} input + "
                f"{avg_completion:.1f} output = {total_tokens:.1f} total"
            )

        # Calculate timing metrics
        logger.info("")
        logger.info("Calculating timing metrics...")
        timing_calculator = build_metric_calculator({"type": "timing", "config": {}})
        timing_metrics = timing_calculator.calculate(
            predictions_path=str(evaluations_path),
            total_task_time=task_time,
        )

        # Save timing metrics
        timing_metrics_path = result_saver.save(
            timing_metrics, f"evaluation_{JUDGE_MODEL_NAME}_timing", OUTPUTS_DIR, timestamp=timestamp
        )
        logger.info(f"   Timing metrics saved to: {timing_metrics_path}")

        # Print timing summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total task time: {timing_metrics.get('total_task_time_seconds', 0):.2f}s")
        logger.info(f"Number of evaluations: {timing_metrics.get('num_inferences', 0)}")

        api_timing = timing_metrics.get("api_call_timing", {})
        if api_timing:
            logger.info("")
            logger.info("API Call Timing:")
            logger.info(f"  Average: {api_timing.get('avg_seconds', 0):.3f}s")
            logger.info(f"  Min: {api_timing.get('min_seconds', 0):.3f}s")
            logger.info(f"  Max: {api_timing.get('max_seconds', 0):.3f}s")
            logger.info(f"  Median: {api_timing.get('median_seconds', 0):.3f}s")
            logger.info(f"  P95: {api_timing.get('p95_seconds', 0):.3f}s")

        total_inference_timing = timing_metrics.get("total_inference_timing", {})
        if total_inference_timing:
            logger.info("")
            logger.info("Total Evaluation Timing:")
            logger.info(f"  Average: {total_inference_timing.get('avg_seconds', 0):.3f}s")
            logger.info(f"  Min: {total_inference_timing.get('min_seconds', 0):.3f}s")
            logger.info(f"  Max: {total_inference_timing.get('max_seconds', 0):.3f}s")
            logger.info(f"  Median: {total_inference_timing.get('median_seconds', 0):.3f}s")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY EVALUATION TASK COMPLETE")
    logger.info("=" * 60)
