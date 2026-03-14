import time

from loguru import logger

from ..exceptions import FatalError
from ..utils.timing import extract_token_counts, merge_timing_data


def run_summary_evaluation_pipeline(
    judge,
    summaries_data: list,
    ground_truth_data: dict,
    criteria: list,
    result_writer,
    aggregation_method: str = "sum",
    return_usage: bool = False,
    progress_interval: int = 100,
):
    """
    Run LLM judge evaluation on summaries.

    Args:
        judge: LLMJudge instance
        summaries_data: List of dicts with video_id and summary from predictions
        ground_truth_data: Dict mapping video_id to ground truth summary
        criteria: List of evaluation criteria dicts with 'name' and 'description'
        result_writer: ResultWriter instance
        aggregation_method: How to aggregate criterion scores ("sum" or "average")
        return_usage: Whether to track token usage and timing
        progress_interval: Log progress every N videos
    """
    total_summaries = len(summaries_data)
    logger.info(f"Evaluating {total_summaries} summaries...")

    # Track statistics
    stats = {
        "total": len(summaries_data),
        "success": 0,
        "failed": 0,
    }

    with result_writer:
        for i, summary_record in enumerate(summaries_data, start=1):
            video_id = summary_record.get("video_id")
            generated_summary = summary_record.get("summary", "")

            # Get ground truth summary for this video
            ground_truth_summary = ground_truth_data.get(video_id, "")

            if not ground_truth_summary:
                logger.warning(f"No ground truth found for video {video_id}, skipping evaluation")
                result_writer.write(
                    {
                        "video_id": video_id,
                        "model": judge.get_model_name(),
                        "ground_truth": "",
                        "generated_summary": generated_summary,
                        "criterion_scores": [],
                        "overall_score": 0.0,
                        "overall_reasoning": "",
                        "error": "No ground truth summary found",
                    }
                )
                stats["failed"] += 1
                continue

            if not generated_summary:
                logger.warning(f"Empty generated summary for video {video_id}, skipping evaluation")
                result_writer.write(
                    {
                        "video_id": video_id,
                        "model": judge.get_model_name(),
                        "ground_truth": ground_truth_summary,
                        "generated_summary": "",
                        "criterion_scores": [],
                        "overall_score": 0.0,
                        "overall_reasoning": "",
                        "error": "Empty generated summary",
                    }
                )
                stats["failed"] += 1
                continue

            try:
                evaluation_start = time.time()
                result = judge.evaluate_summary(
                    ground_truth_summary=ground_truth_summary,
                    generated_summary=generated_summary,
                    criteria=criteria,
                    return_usage=return_usage,
                )
                evaluation_time = time.time() - evaluation_start

                # Calculate overall score from criterion scores based on aggregation method
                criterion_scores = result.get("criterion_scores", [])
                if criterion_scores:
                    scores = [cs["score"] for cs in criterion_scores]
                    if aggregation_method == "sum":
                        overall_score = sum(scores)
                    elif aggregation_method == "average":
                        overall_score = sum(scores) / len(scores)
                    else:
                        raise ValueError(
                            f"Unknown aggregation_method: {aggregation_method}. Must be 'sum' or 'average'"
                        )
                else:
                    overall_score = 0.0

                # Build output record
                output = {
                    "video_id": video_id,
                    "model": judge.get_model_name(),
                    "ground_truth": ground_truth_summary,
                    "generated_summary": generated_summary,
                    "criterion_scores": criterion_scores,
                    "overall_score": overall_score,
                    "overall_reasoning": result.get("overall_reasoning", ""),
                    "aggregation_method": aggregation_method,  # Store for metrics calculator
                }

                # Add timing and token data if tracking enabled
                if return_usage:
                    output.update(merge_timing_data(result, None, evaluation_time))
                    output.update(extract_token_counts(result))

                result_writer.write(output)
                stats["success"] += 1

            except FatalError as e:
                # Fatal error (credentials, configuration) - halt pipeline
                logger.error(f"Fatal error: {e}")
                raise

            except Exception as e:
                # Per-video failure - log and continue
                logger.error(f"Failed to evaluate summary for video {video_id}: {e}")
                result_writer.write(
                    {
                        "video_id": video_id,
                        "model": judge.get_model_name(),
                        "ground_truth": ground_truth_summary,
                        "generated_summary": generated_summary,
                        "criterion_scores": [],
                        "overall_score": 0.0,
                        "overall_reasoning": "",
                        "error": str(e),
                    }
                )
                stats["failed"] += 1

            # Log progress at intervals or on completion (after processing)
            if (i % progress_interval == 0) or (i == stats["total"]):
                logger.info(
                    f"Progress: [{i}/{stats['total']}] - Success: {stats['success']}, Failed: {stats['failed']}"
                )

    logger.success(
        f"Evaluation completed: {total_summaries} summaries - Success: {stats['success']}, Failed: {stats['failed']}"
    )
