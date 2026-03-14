import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..builders import build_tagger
from ..utils.prompts import load_prompt_for_iteration
from ..utils.timing import extract_token_counts, merge_timing_data
from ..writers.base import ResultWriter


def run_cost_benchmark_pipeline(
    iteration_configs: List[Dict[str, Any]],
    video_data: List[Tuple[str, str, Any, Optional[float]]],
    result_writer: ResultWriter,
    threshold: float,
    allowed_tags: List[str],
    tag_defs: Dict[str, str],
    return_usage: bool = True,
    progress_interval: int = 100,
    ignore_threshold: bool = False,
) -> None:
    """
    Args:
        iteration_configs: List of complete per-iteration configurations (already merged)
        video_data: List of tuples (video_id, source_path, preprocessed_data, preprocessing_time)
        result_writer: ResultWriter instance for saving results
        threshold: Confidence threshold for filtering tags
        allowed_tags: List of allowed tags
        tag_defs: Tag definitions dictionary
        return_usage: Whether to track timing/token usage (default: True)
        progress_interval: Log progress every N videos (default: 100)
        ignore_threshold: If True, keep all predicted tags regardless of confidence score
    """
    num_iterations = len(iteration_configs)
    total_videos = len(video_data)
    total_inferences = total_videos * num_iterations
    logger.info(f"Running {num_iterations} iterations on {total_videos} videos")
    logger.info(f"Total inferences: {total_inferences}")

    # Track statistics across all iterations
    stats = {
        "total": total_inferences,
        "success": 0,
        "failed": 0,
    }

    inference_count = 0

    with result_writer:
        for iter_config in iteration_configs:
            iteration = iter_config["iteration"]
            logger.info("=" * 80)
            logger.info(f"ITERATION {iteration}/{num_iterations}")
            logger.info("=" * 80)

            # Log iteration configuration details
            tagger_config = iter_config["tagger_config"]
            prompt_path = iter_config["prompt_path"]
            system_prompt_path = iter_config.get("system_prompt_path")

            logger.info("Iteration Configuration:")
            logger.info(f"  User prompt path: {prompt_path}")
            if system_prompt_path:
                logger.info(f"  System prompt path: {system_prompt_path}")

            # Log model_id if present
            if "model_id" in tagger_config["config"]:
                logger.info(f"  Model ID: {tagger_config['config']['model_id']}")

            # Log labels if present
            if "labels" in tagger_config["config"]:
                labels = tagger_config["config"]["labels"]
                logger.info(f"  Labels: {labels}")

            # Build tagger for this iteration
            logger.info(f"  Tagger type: {tagger_config['type']}")
            tagger = build_tagger(tagger_config)
            model_name = tagger.get_model_name()
            logger.info(f"  Model name: {model_name}")

            # Load prompts for this iteration
            iteration_user_prompt = load_prompt_for_iteration(prompt_path)
            logger.info(f"  User prompt length: {len(iteration_user_prompt)} chars")

            iteration_system_prompt = None
            if system_prompt_path:
                iteration_system_prompt = load_prompt_for_iteration(system_prompt_path)
                logger.info(f"  System prompt length: {len(iteration_system_prompt)} chars")

            # Check if we should skip tag injection (for self-contained prompts)
            skip_tag_injection = iter_config.get("skip_tag_injection", False)
            iteration_tag_defs = {} if skip_tag_injection else tag_defs
            if skip_tag_injection:
                logger.info("  Tag injection: DISABLED (using self-contained prompt)")
            else:
                logger.info(f"  Tag injection: ENABLED ({len(tag_defs)} tag definitions)")

            # Debug: Log the actual prompts that will be sent to the API (once per iteration)
            # Build the final user prompt to see what will actually be sent
            final_user_prompt = tagger._build_user_prompt(allowed_tags, iteration_tag_defs, iteration_user_prompt)

            # Determine final system prompt (use custom or default template)
            final_system_prompt = iteration_system_prompt if iteration_system_prompt else tagger.system_prompt_template

            logger.debug("")
            logger.debug("=" * 80)
            logger.debug(f"ITERATION {iteration} - FINAL SYSTEM PROMPT (sent to API):")
            logger.debug("=" * 80)
            logger.debug(final_system_prompt)
            logger.debug("=" * 80)
            logger.debug("")
            logger.debug("=" * 80)
            logger.debug(f"ITERATION {iteration} - FINAL USER PROMPT (sent to API):")
            logger.debug("=" * 80)
            logger.debug(final_user_prompt)
            logger.debug("=" * 80)
            logger.debug("")

            for (
                video_id,
                source_path,
                preprocessed_data,
                preprocessing_time,
            ) in video_data:
                inference_count += 1

                try:
                    # Conditionally measure timing based on flag
                    if return_usage:
                        inference_start = time.time()
                        video_output = tagger.tag_video(
                            preprocessed_data,
                            allowed_tags,
                            iteration_tag_defs,
                            custom_system_prompt=iteration_system_prompt,
                            custom_user_prompt=iteration_user_prompt,
                            return_usage=True,
                        )
                        total_inference_time = time.time() - inference_start
                    else:
                        video_output = tagger.tag_video(
                            preprocessed_data,
                            allowed_tags,
                            iteration_tag_defs,
                            custom_system_prompt=iteration_system_prompt,
                            custom_user_prompt=iteration_user_prompt,
                            return_usage=False,
                        )

                    # Extract tags and scores
                    video_tags = []
                    video_scores = {}
                    for tag_obj in video_output.get("tags", []):
                        tag_name = tag_obj["tag"]
                        confidence = tag_obj["confidence"]
                        video_scores[tag_name] = confidence
                        if ignore_threshold or confidence >= threshold:
                            video_tags.append(tag_name)

                    # Build result with iteration info
                    result = {
                        "video_id": video_id,
                        "source_path": source_path,
                        "model": model_name,
                        "threshold": threshold,
                        "iteration_number": iteration,
                        "prompt_length_chars": len(iteration_user_prompt),
                        "video_level_tags": video_tags,
                        "video_level_scores": video_scores,
                    }

                    # Merge timing and token data if tracking enabled
                    if return_usage:
                        result.update(merge_timing_data(video_output, preprocessing_time, total_inference_time))
                        result.update(extract_token_counts(video_output))

                    result_writer.write(result)
                    stats["success"] += 1

                except Exception as e:
                    logger.error(f"Failed video {video_id} on iteration {iteration}: {e}")
                    error_result = {
                        "video_id": video_id,
                        "source_path": source_path,
                        "model": model_name,
                        "threshold": threshold,
                        "iteration_number": iteration,
                        "error": str(e),
                        "video_level_tags": [],
                        "video_level_scores": {},
                    }
                    result_writer.write(error_result)
                    stats["failed"] += 1

                # Log progress at intervals
                if (inference_count % progress_interval == 0) or (inference_count == stats["total"]):
                    logger.info(
                        f"Progress: [{inference_count}/{stats['total']}] - "
                        f"Success: {stats['success']}, Failed: {stats['failed']}"
                    )

    logger.success(
        f"Cost benchmark completed: {inference_count} total inferences - "
        f"Success: {stats['success']}, Failed: {stats['failed']}"
    )
