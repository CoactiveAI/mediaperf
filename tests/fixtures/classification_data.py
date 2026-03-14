"""Test fixtures for multi-label classification data."""

from pathlib import Path
from typing import Dict, List

import jsonlines


def create_predictions(
    video_ids: List[str],
    tags_per_video: Dict[str, List[str]] = None,
    include_failures: List[str] = None,
) -> List[Dict]:
    """
    Create synthetic prediction data.

    Args:
        video_ids: List of video IDs to create predictions for
        tags_per_video: Dict mapping video_id to list of predicted tags
        include_failures: List of video IDs that should have "error" field

    Returns:
        List of prediction dicts (JSONL format)
    """
    include_failures = include_failures or []
    predictions = []

    for video_id in video_ids:
        if video_id in include_failures:
            predictions.append({"video_id": video_id, "error": "API call failed"})
        else:
            tags = tags_per_video.get(video_id, []) if tags_per_video else []
            predictions.append({"video_id": video_id, "video_level_tags": tags})

    return predictions


def create_ground_truth(
    video_ids: List[str],
    tags_per_video: Dict[str, List[str]],
) -> List[Dict]:
    """
    Create synthetic ground truth data.

    Args:
        video_ids: List of video IDs to create ground truth for
        tags_per_video: Dict mapping video_id to list of ground truth tags

    Returns:
        List of ground truth dicts (JSONL format)
    """
    ground_truth = []

    for video_id in video_ids:
        tags = tags_per_video.get(video_id, [])
        ground_truth.append({"video_id": video_id, "tags": tags})

    return ground_truth


def write_to_jsonl(data: List[Dict], output_path: Path):
    """Write data to JSONL file."""
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(data)
