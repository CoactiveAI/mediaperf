"""Max confidence aggregation strategy."""

from typing import Dict, List, Tuple


def aggregate_video(frame_predictions: List[Dict], threshold: float) -> Tuple[List[str], Dict[str, float]]:
    """
    Aggregate a video's frame-level predictions to video-level tags.

    Rule: if any frame has tag confidence >= threshold, the video gets that tag.
    We also return per-tag max confidence ("video_level_scores") for transparency.

    Args:
        frame_predictions: iterable of frame outputs, each like {"tags":[{"tag":..., "confidence":...}, ...]}
        threshold: float in [0,1]. Tags with max(frame_conf) >= threshold are selected.

    Returns:
        (video_level_tags, video_level_scores)
        - video_level_tags: sorted list of tags selected by the threshold rule
        - video_level_scores: dict[tag] = max_confidence_over_frames
    """
    max_conf: Dict[str, float] = {}
    for frame in frame_predictions:
        for item in frame.get("tags", []):
            tag = item.get("tag")
            conf = float(item.get("confidence", 0.0))
            if tag is None:
                continue
            if conf > max_conf.get(tag, 0.0):
                max_conf[tag] = conf

    selected = sorted([t for t, c in max_conf.items() if c >= threshold])
    return selected, max_conf
