import time

from loguru import logger

from ..exceptions import FatalError
from ..utils.timing import extract_token_counts, merge_timing_data


def run_video_summarization_pipeline(
    summarizer,
    video_data: list,
    result_writer,
    return_usage: bool = False,
    progress_interval: int = 100,
):
    """
    Run summarization inference on videos.

    Args:
        summarizer: VideoSummarizer instance
        video_data: List of (video_id, video_source, processed_source, preprocessing_time) tuples
        result_writer: ResultWriter instance
        return_usage: Whether to track token usage and timing
        progress_interval: Log progress every N videos
    """
    total_videos = len(video_data)
    logger.info(f"Processing {total_videos} videos...")

    # Track statistics
    stats = {
        "total": len(video_data),
        "success": 0,
        "failed": 0,
    }

    with result_writer:
        for i, (video_id, original_source, processed_source, preprocessing_time) in enumerate(video_data, start=1):
            try:
                inference_start = time.time()
                result = summarizer.summarize_video(
                    processed_source,
                    return_usage=return_usage,
                )
                inference_time = time.time() - inference_start

                # Check if summarizer returned an error (e.g., parsing failure, Vertex MAX_TOKENS)
                if "error" in result:
                    logger.error(f"Failed to summarize video {video_id}: {result['error']}")
                    error_result = {
                        "video_id": video_id,
                        "source_path": original_source,
                        "model": summarizer.get_model_name(),
                        "summary": "",
                        "error": result["error"],
                    }
                    result_writer.write(error_result)
                    stats["failed"] += 1
                    continue

                # Extract summary
                summary = result.get("summary", "")

                # Build output record
                output = {
                    "video_id": video_id,
                    "source_path": original_source,
                    "model": summarizer.get_model_name(),
                    "summary": summary,
                }

                # Add timing and token data if tracking enabled
                if return_usage:
                    output.update(merge_timing_data(result, preprocessing_time, inference_time))
                    output.update(extract_token_counts(result))

                result_writer.write(output)
                stats["success"] += 1

            except FatalError as e:
                # Fatal error (credentials, configuration) - halt pipeline
                logger.error(f"Fatal error: {e}")
                raise

            except Exception as e:
                # Per-video failure - log and continue
                logger.error(f"Failed to summarize video {video_id}: {e}")
                result_writer.write(
                    {
                        "video_id": video_id,
                        "source_path": original_source,
                        "model": summarizer.get_model_name(),
                        "summary": "",
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
        f"Summarization completed: {total_videos} videos - Success: {stats['success']}, Failed: {stats['failed']}"
    )
