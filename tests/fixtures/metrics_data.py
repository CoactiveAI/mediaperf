"""Test fixtures for cost and timing metrics data."""

from pathlib import Path
from typing import Dict, List

import jsonlines


def create_inference_results_with_usage(
    video_ids: List[str],
    tokens_per_video: Dict[str, Dict[str, int]] = None,
    timing_per_video: Dict[str, Dict[str, float]] = None,
    iteration_per_video: Dict[str, int] = None,
) -> List[Dict]:
    """
    Create synthetic inference results with token usage and timing data.

    Args:
        video_ids: List of video IDs
        tokens_per_video: Dict mapping video_id to {"input_tokens": int, "output_tokens": int}
        timing_per_video: Dict mapping video_id to {"api_call_time_seconds": float, ...}
        iteration_per_video: Dict mapping video_id to iteration number (for cost benchmark)

    Returns:
        List of inference result dicts
    """
    results = []

    for video_id in video_ids:
        result = {"video_id": video_id, "model": "test-model"}

        if tokens_per_video and video_id in tokens_per_video:
            tokens = tokens_per_video[video_id]
            result["prompt_tokens"] = tokens.get("input_tokens", 0)
            result["completion_tokens"] = tokens.get("output_tokens", 0)

        if timing_per_video and video_id in timing_per_video:
            timing = timing_per_video[video_id]
            result.update(timing)

        if iteration_per_video and video_id in iteration_per_video:
            result["iteration_number"] = iteration_per_video[video_id]

        results.append(result)

    return results


def write_to_jsonl(data: List[Dict], output_path: Path):
    """Write data to JSONL file."""
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(data)
