import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..exceptions import FatalError
from ..models.base import VideoTagger
from ..utils.timing import extract_token_counts, merge_timing_data
from ..writers.base import ResultWriter


def run_video_tagging_pipeline(
    tagger: VideoTagger,
    video_data: List[Tuple[str, str, Any, Optional[float]]],
    result_writer: ResultWriter,
    threshold: float,
    allowed_tags: List[str],
    tag_defs: Dict[str, str],
    return_usage: bool = False,
    progress_interval: int = 100,
    ignore_threshold: bool = False,
) -> None:
    """
    Run video tagging pipeline with a VideoTagger implementation.

    Args:
        tagger: VideoTagger instance
        video_data: List of tuples (video_id, source_path, preprocessed_data, preprocessing_time)
            - video_id: Video identifier
            - source_path: Original video path/URI for logging
            - preprocessed_data: Preprocessed input for tagger (S3 URI or base64 frames)
            - preprocessing_time: Time taken for preprocessing (None if no preprocessing)
        result_writer: ResultWriter instance for saving results
        threshold: Confidence threshold for filtering tags
        allowed_tags: List of allowed tags
        tag_defs: Tag definitions dictionary
        return_usage: Whether to track and return timing/token usage
        progress_interval: Log progress every N videos (default: 100)
        ignore_threshold: If True, keep all predicted tags regardless of confidence score
    """
    # Track statistics
    stats = {
        "total": len(video_data),
        "success": 0,
        "failed": 0,
    }

    with result_writer:
        for i, (video_id, source_path, preprocessed_data, preprocessing_time) in enumerate(video_data, 1):
            try:
                # Conditionally measure timing based on flag
                if return_usage:
                    inference_start = time.time()
                    video_output = tagger.tag_video(preprocessed_data, allowed_tags, tag_defs, return_usage=True)
                    total_inference_time = time.time() - inference_start
                else:
                    video_output = tagger.tag_video(preprocessed_data, allowed_tags, tag_defs, return_usage=False)

                # Check if tagger returned an error (e.g., parsing failure, Vertex MAX_TOKENS)
                if "error" in video_output:
                    logger.error(f"Failed to process {video_id}: {video_output['error']}")
                    error_result = {
                        "video_id": video_id,
                        "source_path": source_path,
                        "model": tagger.get_model_name(),
                        "threshold": threshold,
                        "error": video_output["error"],
                        "video_level_tags": [],
                        "video_level_scores": {},
                    }
                    result_writer.write(error_result)
                    stats["failed"] += 1
                    continue

                video_tags = []
                video_scores = {}
                for tag_obj in video_output.get("tags", []):
                    tag_name = tag_obj["tag"]
                    confidence = tag_obj["confidence"]
                    video_scores[tag_name] = confidence
                    if ignore_threshold or confidence >= threshold:
                        video_tags.append(tag_name)

                result = {
                    "video_id": video_id,
                    "source_path": source_path,
                    "model": tagger.get_model_name(),
                    "threshold": threshold,
                    "video_level_tags": video_tags,
                    "video_level_scores": video_scores,
                }

                # Merge timing and token data if tracking enabled
                if return_usage:
                    result.update(merge_timing_data(video_output, preprocessing_time, total_inference_time))
                    result.update(extract_token_counts(video_output))

                result_writer.write(result)
                stats["success"] += 1

            except FatalError as e:
                # Fatal error (credentials, configuration) - halt pipeline
                logger.error(f"Fatal error: {e}")
                raise

            except Exception as e:
                # Per-video failure - log and continue
                logger.error(f"Failed to process {video_id}: {e}")
                error_result = {
                    "video_id": video_id,
                    "source_path": source_path,
                    "model": tagger.get_model_name(),
                    "threshold": threshold,
                    "error": str(e),
                    "video_level_tags": [],
                    "video_level_scores": {},
                }
                result_writer.write(error_result)
                stats["failed"] += 1

            # Log progress at intervals
            if (i % progress_interval == 0) or (i == stats["total"]):
                logger.info(
                    f"Progress: [{i}/{stats['total']}] - Success: {stats['success']}, Failed: {stats['failed']}"
                )
