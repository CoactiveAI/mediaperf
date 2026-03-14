"""I/O utilities for saving results."""

import json
from typing import Dict, List, Tuple

from ..aggregation.max_confidence import aggregate_video


def write_aggregates_jsonl(
    video_to_aggregate: Dict[str, Tuple[List[str], Dict[str, float]]],
    out_path: str,
    threshold: float = 0.6,
) -> None:
    """
    Write aggregated video-level tags to a JSONL file.

    Args:
        video_to_aggregate: mapping video_id -> (video_level_tags, video_level_scores)
        out_path: path to write the JSONL file
        threshold: threshold used to produce the aggregates (stored for provenance)
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for vid, (tags, scores) in video_to_aggregate.items():
            line = {
                "video_id": vid,
                "threshold": threshold,
                "video_level_tags": tags,
                "video_level_scores": scores,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def aggregate_and_save_from_frames(
    per_video_frames: Dict[str, List],
    out_path: str,
    threshold: float = 0.5,
) -> None:
    """
    Convenience wrapper:
    Given a dict of {video_id: [frame_pred, ...]}, aggregate and save JSONL.

    Args:
        per_video_frames: mapping of video_id -> list of per-frame prediction dicts
        out_path: JSONL output path
        threshold: confidence threshold for selecting video-level tags
    """

    aggregates = {}
    for vid, frames in per_video_frames.items():
        aggregates[vid] = aggregate_video(frames, threshold)
    write_aggregates_jsonl(aggregates, out_path, threshold)
