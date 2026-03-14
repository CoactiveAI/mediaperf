import json
import time
from pathlib import Path

from loguru import logger

from ..builders import (
    build_metric_calculator,
    build_preprocessor,
    build_result_saver,
    build_result_writer,
)
from ..datasets.youtube_ads import extract_video_id
from ..utils.video_selection import sample_videos
from .cost_benchmark_pipeline import run_cost_benchmark_pipeline


def load_label_descriptions(label_descriptions_path: str) -> dict:
    """Load label descriptions from JSON file."""
    with open(label_descriptions_path, "r") as f:
        return json.load(f)


def run_cost_benchmark(config: dict):
    """Run the workload benchmark task with per-iteration model and prompt configurations."""

    # Extract paths from config
    INPUTS_DIR = Path(config["paths"]["inputs_dir"])
    OUTPUTS_DIR = Path(config["paths"]["outputs_dir"])

    # Create output directory if it doesn't exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Extract inference config
    THRESHOLD = config["inference"]["threshold"]

    # Load label descriptions
    LABEL_DESCRIPTIONS_PATH = INPUTS_DIR / "youtube_ads_label_descriptions.json"
    YOUTUBE_ADS_LABEL_DESCRIPTIONS = load_label_descriptions(str(LABEL_DESCRIPTIONS_PATH))

    # Extract cost benchmark config
    cost_config = config["cost_benchmark"]
    iterations_config = cost_config["iterations"]
    num_iterations = len(iterations_config)

    # Get model name for logging and file naming
    MODEL_NAME = config["pipeline"]["tagger"]["config"].get(
        "model_name", config["pipeline"]["tagger"]["config"].get("model_id", "Unknown")
    )

    logger.info("=" * 60)
    logger.info(f"COST BENCHMARK - {MODEL_NAME}")
    logger.info(f"Iterations: {num_iterations}")
    logger.info("=" * 60)

    # Step 1: Sample videos based on config
    logger.info("[1/4] Sampling videos...")
    video_list, video_sources = sample_videos(config, INPUTS_DIR)
    logger.info(f"   Selected {len(video_list)} videos")

    # Step 2: Prepare tag definitions
    logger.info("[2/4] Loading tag definitions...")
    ALLOWED_TAGS = list(YOUTUBE_ADS_LABEL_DESCRIPTIONS.keys())
    TAG_DEFS = YOUTUBE_ADS_LABEL_DESCRIPTIONS
    logger.info(f"   Loaded {len(ALLOWED_TAGS)} tags")

    # Filter to test tags if specified
    test_tags = config.get("inference", {}).get("test_tags")
    if test_tags:
        ALLOWED_TAGS = [tag for tag in test_tags if tag in ALLOWED_TAGS]
        TAG_DEFS = {k: v for k, v in TAG_DEFS.items() if k in ALLOWED_TAGS}
        logger.info(f"   TEST MODE: Limited to {len(ALLOWED_TAGS)} tag(s): {ALLOWED_TAGS}")

    # Step 3: Build pipeline components
    logger.info("[3/4] Building pipeline components...")

    # Prepare per-iteration configs by merging base tagger config with iteration overrides
    base_tagger_config = config["pipeline"]["tagger"]
    merged_iteration_configs = []

    for iter_spec in iterations_config:
        # Copy base tagger config
        merged_tagger_config = {"type": base_tagger_config["type"], "config": {**base_tagger_config["config"]}}

        # Apply iteration-specific overrides
        if "model_id" in iter_spec:
            merged_tagger_config["config"]["model_id"] = iter_spec["model_id"]
        if "labels" in iter_spec:
            merged_tagger_config["config"]["labels"] = iter_spec["labels"]

        # Build iteration config
        iteration_config = {
            "iteration": iter_spec["iteration"],
            "prompt_path": iter_spec["prompt_path"],
            "system_prompt_path": iter_spec.get("system_prompt_path"),  # Optional
            "tagger_config": merged_tagger_config,
            "skip_tag_injection": iter_spec.get("skip_tag_injection", False),
        }
        merged_iteration_configs.append(iteration_config)

    logger.info(f"   Prepared {len(merged_iteration_configs)} iteration configs")

    # Warn if using invoke API (input tokens unavailable)
    if base_tagger_config["config"].get("api_type") == "invoke":
        logger.warning("   Using invoke_model API: input token counts will be unavailable (only output tokens tracked)")

    # Get shared timestamp from config (created in main.py)
    timestamp = config.get("_timestamp", time.strftime("%y%m%d_%H%M%S"))

    # Build preprocessor (optional)
    preprocessor_config = config["pipeline"].get("preprocessor")
    if preprocessor_config and preprocessor_config.get("type"):
        # Extract cache_run_id if provided
        cache_run_id = preprocessor_config.get("config", {}).pop("run_id_for_cache", None)

        # Always generate current run_id
        run_id = f"{MODEL_NAME}_{timestamp}"
        logger.info(f"   Current run_id: {run_id}")

        preprocessor = build_preprocessor(
            preprocessor_config, run_id=run_id, cache_run_id=cache_run_id, full_config=config
        )
        preprocessor_type = preprocessor_config["type"]
        logger.info(f"   Preprocessor: {preprocessor_type}")

        if preprocessor_type == "video_base64":
            logger.info(f"   Downloading and encoding {len(video_sources)} videos...")

        video_data = preprocessor.preprocess(video_sources)
    else:
        # No preprocessing - use video sources directly with None for preprocessing_time
        video_data = [(extract_video_id(vid), src, src, None) for vid, src in zip(video_list, video_sources)]
        logger.info("   Preprocessor: none (using sources directly)")

    # Build result writer with timestamped filename

    # Create predictions filename: workload_{model_name}_predictions_{timestamp}.jsonl
    predictions_path = OUTPUTS_DIR / f"workload_{MODEL_NAME}_predictions_{timestamp}.jsonl"

    # Update config with timestamped path
    config["pipeline"]["result_writer"]["config"]["output_path"] = str(predictions_path)
    result_writer = build_result_writer(config["pipeline"]["result_writer"])

    logger.info(f"   Result writer: {config['pipeline']['result_writer']['type']}")

    # Get usage tracking flag from config (default: True for cost benchmarks)
    return_usage = config.get("tracking", {}).get("return_usage", True)
    progress_interval = config.get("tracking", {}).get("progress_interval", 100)
    ignore_threshold = config.get("inference", {}).get("ignore_threshold", False)
    logger.info(f"   Usage tracking: {'enabled' if return_usage else 'disabled'}")
    logger.info(f"   Progress logging interval: {progress_interval} videos")
    logger.info(f"   Threshold filtering: {'disabled' if ignore_threshold else f'enabled (threshold={THRESHOLD})'}")

    # Step 4: Run cost benchmark
    logger.info("[4/4] Running cost benchmark...")
    logger.info(f"   Videos: {len(video_sources)}")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Total inferences: {len(video_sources) * num_iterations}")
    logger.info("")

    task_start = time.time()
    run_cost_benchmark_pipeline(
        iteration_configs=merged_iteration_configs,
        video_data=video_data,
        result_writer=result_writer,
        threshold=THRESHOLD,
        allowed_tags=ALLOWED_TAGS,
        tag_defs=TAG_DEFS,
        return_usage=return_usage,
        progress_interval=progress_interval,
        ignore_threshold=ignore_threshold,
    )
    task_time = time.time() - task_start

    logger.info("")
    logger.info("=" * 60)
    logger.info("COST BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results: {predictions_path}")
    logger.info(f"Total inferences: {len(video_sources) * num_iterations}")
    logger.info(f"Total task time: {task_time:.2f} seconds")

    # Calculate cost metrics
    logger.info("")
    logger.info("Calculating cost metrics...")

    pricing = config.get("pricing", {})
    input_price = pricing.get("input_per_1m_tokens", 0.0)
    output_price = pricing.get("output_per_1m_tokens", 0.0)

    # Build cost metric calculator
    cost_metric_calculator = build_metric_calculator(
        {
            "type": "cost",
            "config": {
                "input_price_per_1m": input_price,
                "output_price_per_1m": output_price,
            },
        }
    )

    # Calculate cost metrics
    cost_metrics = cost_metric_calculator.calculate(predictions_path=str(predictions_path), ground_truth_path=None)

    # Save cost metrics
    logger.info("   Saving cost metrics...")
    result_saver = build_result_saver(config.get("result_saver", {"type": "json"}))
    cost_metrics_path = result_saver.save(cost_metrics, f"workload_{MODEL_NAME}_cost", OUTPUTS_DIR, timestamp=timestamp)
    logger.info(f"   Cost metrics saved to: {cost_metrics_path}")

    # Print cost summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("COST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total cost: ${cost_metrics.get('total_cost', 0):.4f}")
    logger.info(f"Cost per inference: ${cost_metrics.get('cost_per_inference', 0):.6f}")
    logger.info(f"Total tokens: {cost_metrics.get('total_tokens', 0):,}")
    avg_prompt = cost_metrics.get("avg_prompt_tokens", 0)
    avg_completion = cost_metrics.get("avg_completion_tokens", 0)
    total_tokens = avg_prompt + avg_completion
    logger.info(
        f"Average tokens per inference: {avg_prompt:.1f} input + {avg_completion:.1f} output = {total_tokens:.1f} total"
    )

    # Calculate and save timing metrics if tracking is enabled
    if return_usage:
        logger.info("")
        logger.info("Calculating timing metrics...")
        timing_calculator = build_metric_calculator({"type": "timing", "config": {}})
        timing_metrics = timing_calculator.calculate(
            predictions_path=str(predictions_path),
            total_task_time=task_time,
        )

        # Save timing metrics
        logger.info("   Saving timing metrics...")
        timing_metrics_path = result_saver.save(
            timing_metrics, f"workload_{MODEL_NAME}_timing", OUTPUTS_DIR, timestamp=timestamp
        )
        logger.info(f"   Timing metrics saved to: {timing_metrics_path}")

        # Print timing summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total task time: {timing_metrics.get('total_task_time_seconds', 0):.2f}s")
        logger.info(f"Number of inferences: {timing_metrics.get('num_inferences', 0)}")

        api_timing = timing_metrics.get("api_call_timing", {})
        if api_timing:
            logger.info("")
            logger.info("API Call Timing:")
            logger.info(f"  Average: {api_timing.get('avg_seconds', 0):.3f}s")
            logger.info(f"  Min: {api_timing.get('min_seconds', 0):.3f}s")
            logger.info(f"  Max: {api_timing.get('max_seconds', 0):.3f}s")
            logger.info(f"  Median: {api_timing.get('median_seconds', 0):.3f}s")
            logger.info(f"  P95: {api_timing.get('p95_seconds', 0):.3f}s")

        total_timing = timing_metrics.get("total_inference_timing", {})
        if total_timing:
            logger.info("")
            logger.info("Total Inference Timing:")
            logger.info(f"  Average: {total_timing.get('avg_seconds', 0):.3f}s")
            logger.info(f"  Total: {total_timing.get('total_seconds', 0):.2f}s")

        # Per-iteration summary
        per_iteration = timing_metrics.get("per_iteration_timing", {})
        if per_iteration:
            logger.info("")
            logger.info("Per-Iteration Summary:")
            for iter_key in sorted(per_iteration.keys()):
                iter_data = per_iteration[iter_key]
                iter_num = iter_key.replace("iteration_", "")
                api_avg = iter_data.get("api_call_timing", {}).get("avg_seconds", 0)
                logger.info(f"  Iteration {iter_num}: {api_avg:.3f}s avg API time")

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Cost metrics: {cost_metrics_path}")
    if return_usage:
        logger.info(f"Timing metrics: {timing_metrics_path}")
