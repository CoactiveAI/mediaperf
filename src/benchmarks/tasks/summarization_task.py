import time
from pathlib import Path

from loguru import logger

from ..builders import (
    build_metric_calculator,
    build_preprocessor,
    build_result_saver,
    build_result_writer,
    build_summarizer,
)
from ..datasets.youtube_ads import extract_video_id
from ..utils.video_selection import sample_videos
from .video_summarization_pipeline import run_video_summarization_pipeline


def run_summarization(config: dict):
    """Run the video summarization task."""

    # Extract paths from config
    INPUTS_DIR = Path(config["paths"]["inputs_dir"])
    OUTPUTS_DIR = Path(config["paths"]["outputs_dir"])

    # Create output directory if it doesn't exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get model name and timestamp for consistent file naming
    MODEL_NAME = config["pipeline"]["summarizer"]["config"].get(
        "model_name",
        config["pipeline"]["summarizer"]["config"].get("model_id", "Unknown"),
    )
    timestamp = config.get("_timestamp", time.strftime("%y%m%d_%H%M%S"))

    logger.info("=" * 60)
    logger.info(f"VIDEO SUMMARIZATION - {MODEL_NAME}")
    logger.info("=" * 60)

    # Step 1: Sample videos based on config
    logger.info("[1/3] Sampling videos...")
    video_list, video_sources = sample_videos(config, INPUTS_DIR)

    # Step 2: Build pipeline components
    logger.info("[2/3] Building pipeline components...")

    # Build summarizer
    summarizer = build_summarizer(config["pipeline"]["summarizer"])
    logger.info(f"   Summarizer: {config['pipeline']['summarizer']['type']}")

    # Warn if using invoke API (input tokens unavailable)
    if config["pipeline"]["summarizer"]["config"].get("api_type") == "invoke":
        logger.warning("   Using invoke_model API: input token counts will be unavailable (only output tokens tracked)")

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
    predictions_path = OUTPUTS_DIR / f"summarization_{MODEL_NAME}_predictions_{timestamp}.jsonl"

    # Update config with timestamped path
    config["pipeline"]["result_writer"]["config"]["output_path"] = str(predictions_path)
    result_writer = build_result_writer(config["pipeline"]["result_writer"])

    logger.info(f"   Result writer: {config['pipeline']['result_writer']['type']}")

    # Get usage tracking flag from config
    return_usage = config.get("tracking", {}).get("return_usage", False)
    progress_interval = config.get("tracking", {}).get("progress_interval", 100)
    logger.info(f"   Usage tracking: {'enabled' if return_usage else 'disabled'}")
    logger.info(f"   Progress logging interval: {progress_interval} videos")

    # Step 3: Run inference
    logger.info(f"[3/3] Running summarization on {len(video_sources)} videos...")
    task_start = time.time()
    run_video_summarization_pipeline(
        summarizer=summarizer,
        video_data=video_data,
        result_writer=result_writer,
        return_usage=return_usage,
        progress_interval=progress_interval,
    )
    task_time = time.time() - task_start
    logger.info(f"   Predictions saved to: {predictions_path}")
    logger.info(f"   Total task time: {task_time:.2f} seconds")

    # Calculate cost and timing metrics if tracking is enabled
    if return_usage:
        # Calculate cost metrics if pricing info is available
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
                predictions_path=str(predictions_path),
                ground_truth_path=None,
            )

            # Save cost metrics
            result_saver = build_result_saver(config.get("result_saver", {"type": "json"}))
            cost_metrics_path = result_saver.save(
                cost_metrics,
                f"summarization_{MODEL_NAME}_cost",
                OUTPUTS_DIR,
                timestamp=timestamp,
            )
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
                f"Average tokens per inference: {avg_prompt:.1f} input + "
                f"{avg_completion:.1f} output = {total_tokens:.1f} total"
            )

        # Calculate timing metrics
        logger.info("")
        logger.info("Calculating timing metrics...")
        timing_calculator = build_metric_calculator({"type": "timing", "config": {}})
        timing_metrics = timing_calculator.calculate(
            predictions_path=str(predictions_path),
            total_task_time=task_time,
        )

        # Save timing metrics
        result_saver = build_result_saver(config.get("result_saver", {"type": "json"}))
        timing_metrics_path = result_saver.save(
            timing_metrics,
            f"summarization_{MODEL_NAME}_timing",
            OUTPUTS_DIR,
            timestamp=timestamp,
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

        total_inference_timing = timing_metrics.get("total_inference_timing", {})
        if total_inference_timing:
            logger.info("")
            logger.info("Total Inference Timing:")
            logger.info(f"  Average: {total_inference_timing.get('avg_seconds', 0):.3f}s")
            logger.info(f"  Min: {total_inference_timing.get('min_seconds', 0):.3f}s")
            logger.info(f"  Max: {total_inference_timing.get('max_seconds', 0):.3f}s")
            logger.info(f"  Median: {total_inference_timing.get('median_seconds', 0):.3f}s")

        preprocessing_timing = timing_metrics.get("preprocessing_timing", {})
        if preprocessing_timing:
            logger.info("")
            logger.info("Preprocessing Timing:")
            logger.info(f"  Average: {preprocessing_timing.get('avg_seconds', 0):.3f}s")
            logger.info(f"  Min: {preprocessing_timing.get('min_seconds', 0):.3f}s")
            logger.info(f"  Max: {preprocessing_timing.get('max_seconds', 0):.3f}s")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARIZATION TASK COMPLETE")
    logger.info("=" * 60)
