"""YouTube Advertisements dataset utilities."""

import collections
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, Tuple

import jsonlines


@dataclass
class LabelHolder:
    """Label holder."""

    description: str | None
    """Description of the label."""
    type: str | None
    """Label type."""


@dataclass
class YoutubeAdvertisementsLabels:
    """YouTube Advertisements labels."""

    effective: int
    funny: float
    exciting: float
    topic_abbreviation: str
    topic_description: str
    sentiment_abbreviation: str
    sentiment_description: str
    language: str | None


_video_language_label_mapping = {
    "1": "english",
    "0": "not english",
    "-1": "language does not matter to understand the ad",
}


def get_video_labels(label_data, video_id: str) -> YoutubeAdvertisementsLabels:
    """Get labels for a given video."""
    return YoutubeAdvertisementsLabels(
        effective=label_data["video_Effective"].get(video_id, None),
        funny=label_data["video_Funny"].get(video_id, None),
        exciting=label_data["video_Exciting"].get(video_id, None),
        language=_video_language_label_mapping.get(
            label_data["video_Language"].get(video_id, None),
            None,
        ),
        sentiment_description=label_data["Sentiments"][str(label_data["video_Sentiments"].get(video_id, None))][
            "description"
        ],
        sentiment_abbreviation=label_data["Sentiments"][str(label_data["video_Sentiments"].get(video_id, None))][
            "abbreviation"
        ],
        topic_abbreviation=label_data["Topics"][str(label_data["video_Topics"].get(video_id, None))]["abbreviation"],
        topic_description=label_data["Topics"][str(label_data["video_Topics"].get(video_id, None))]["description"],
    )


def read_advertisements_labels(
    label_path: str,
    video_list: list[str],
    label_descriptions: Dict[str, str],
    gt_threshold: float = 0.5,
) -> Tuple[Dict[str, Set], Dict[str, LabelHolder]]:
    """
    Read YouTube Advertisements labels from file.

    Args:
        label_path: Path to labels JSON file
        video_list: List of video filenames
        label_descriptions: Dictionary of label descriptions
        gt_threshold: Threshold for binary labels

    Returns:
        Tuple of (label_to_videos, label_descriptions_dict)
    """
    if not 0.0 <= gt_threshold <= 1.0:
        raise ValueError(f"gt_threshold={gt_threshold}, must be between 0 and 1")

    with open(label_path, encoding="utf-8") as label_f:
        label_data = json.load(label_f)

    label_to_videos = defaultdict(set)
    label_descriptions_dict = {}

    for video_id in video_list:
        match = re.match(r"vid_(.*)\.mp4", video_id)
        if match:
            video_id = match.group(1)
            clean_video_id = f"vid_{video_id}"
            all_labels = get_video_labels(label_data, video_id)
            if all_labels.funny > gt_threshold:
                label_to_videos["funny"].add(clean_video_id)
            if all_labels.exciting > gt_threshold:
                label_to_videos["exciting"].add(clean_video_id)
            if float(all_labels.effective) > 5.0 * gt_threshold:
                label_to_videos["effective"].add(clean_video_id)
            label_to_videos[all_labels.topic_abbreviation].add(clean_video_id)
            label_descriptions_dict[all_labels.topic_abbreviation] = LabelHolder(
                description=label_descriptions[all_labels.topic_abbreviation],
                type="topic",
            )
            label_to_videos[all_labels.sentiment_abbreviation].add(clean_video_id)
            label_descriptions_dict[all_labels.sentiment_abbreviation] = LabelHolder(
                description=label_descriptions[all_labels.sentiment_abbreviation],
                type="sentiment",
            )
    return label_to_videos, label_descriptions_dict


def read_video_annotations(
    annotations_path: str,
    tag_descriptions_path: str,
    video_list: list[str],
) -> Tuple[Dict[str, Set], Dict[str, LabelHolder]]:
    """
    Read video annotations from canonical JSON format.

    Args:
        annotations_path: Path to video_annotations.json
        tag_descriptions_path: Path to tag_descriptions.json
        video_list: List of video filenames to filter

    Returns:
        Tuple of (label_to_videos, label_descriptions_dict)
    """
    with open(annotations_path, encoding="utf-8") as f:
        video_to_tags = json.load(f)

    with open(tag_descriptions_path, encoding="utf-8") as f:
        tag_descriptions_raw = json.load(f)

    video_id_map = {}
    for video_filename in video_list:
        clean_id = extract_video_id(video_filename)
        video_id_map[clean_id] = video_filename

    label_to_videos = defaultdict(set)

    for clean_id in video_id_map.keys():
        if clean_id in video_to_tags:
            tags = video_to_tags[clean_id]
            for tag in tags:
                label_to_videos[tag].add(clean_id)

    # Build label descriptions for ALL available tags (not just tags in selected videos)
    # This ensures the model sees all possible tags during inference
    label_descriptions_dict = {}
    for tag, description in tag_descriptions_raw.items():
        label_descriptions_dict[tag] = LabelHolder(
            description=description,
            type="general",
        )

    return dict(label_to_videos), label_descriptions_dict


def inverted_to_jsonl(inv_json: Dict[str, Set], out_jsonl: str) -> None:
    """
    Convert inverted index (label -> videos) to JSONL (video -> labels).

    Args:
        inv_json: Dictionary mapping labels to sets of video IDs
        out_jsonl: Output JSONL file path
    """
    vids = collections.defaultdict(set)
    for tag, vids_list in inv_json.items():
        for vid in vids_list:
            vids[vid].add(tag)
    with jsonlines.open(out_jsonl, "w") as w:
        for vid, tags in vids.items():
            # Keep "vid_" prefix to match tagging module format
            w.write({"video_id": vid, "tags": sorted(tags)})


def extract_video_id(video_path: str) -> str:
    """
    Extract the video id from filenames, keeping the 'vid_' prefix.

    Examples:
      'YoutubeAdvertisements_vid_-1yXdOufzKE.mp4'      -> 'vid_-1yXdOufzKE'
      'YoutubeAdvertisements_vid_-44_igsZtgU.mp4'      -> 'vid_-44_igsZtgU'
      'vid_08ySTcxh_cU.mp4'                            -> 'vid_08ySTcxh_cU'
      'vid_-V2r_9Um2Tw.mp4'                            -> 'vid_-V2r_9Um2Tw'
      's3://bucket/prefix/vid_123.mp4'                 -> 'vid_123'
    """
    import os

    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    # Handle "YoutubeAdvertisements_vid_<ID>" pattern
    if stem.startswith("YoutubeAdvertisements_vid_"):
        return stem.replace("YoutubeAdvertisements_", "")

    # Already has vid_ prefix or other format - return as-is (without extension)
    return stem
