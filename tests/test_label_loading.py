"""Tests for label loading from YouTube Ads dataset."""

import jsonlines
import pytest

from benchmarks.datasets.youtube_ads import extract_video_id, inverted_to_jsonl, read_advertisements_labels
from tests.fixtures.label_data import create_label_descriptions, create_youtube_ads_label_json, write_label_json


@pytest.mark.fast
def test_read_labels_with_threshold(tmp_path):
    """Test label loading with threshold filtering."""
    label_data = create_youtube_ads_label_json()
    label_descriptions = create_label_descriptions()

    label_file = tmp_path / "labels.json"
    write_label_json(label_data, label_file)

    video_list = ["vid_ABC123.mp4", "vid_DEF456.mp4", "vid_GHI789.mp4"]

    label_to_videos, label_desc_dict = read_advertisements_labels(
        label_path=str(label_file), video_list=video_list, label_descriptions=label_descriptions, gt_threshold=0.5
    )

    assert "funny" in label_to_videos
    assert "vid_ABC123" in label_to_videos["funny"]
    assert "vid_DEF456" not in label_to_videos["funny"]

    assert "exciting" in label_to_videos
    assert "vid_DEF456" in label_to_videos["exciting"]

    assert "food" in label_to_videos
    assert "tech" in label_to_videos
    assert "cheerful" in label_to_videos


@pytest.mark.fast
@pytest.mark.parametrize("invalid_threshold", [1.5, -0.1, 2.0, -1.0])
def test_threshold_validation(invalid_threshold):
    """Test that invalid threshold raises ValueError."""
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        read_advertisements_labels(
            label_path="dummy", video_list=[], label_descriptions={}, gt_threshold=invalid_threshold
        )


@pytest.mark.fast
def test_extract_video_id_youtube_format():
    """Test video ID extraction from YouTube Advertisements format."""
    assert extract_video_id("YoutubeAdvertisements_vid_-1yXdOufzKE.mp4") == "vid_-1yXdOufzKE"
    assert extract_video_id("YoutubeAdvertisements_vid_-44_igsZtgU.mp4") == "vid_-44_igsZtgU"
    assert extract_video_id("YoutubeAdvertisements_vid__underscoreStart.mp4") == "vid__underscoreStart"


@pytest.mark.fast
def test_extract_video_id_simple_format():
    """Test video ID extraction from simple vid_*.mp4 format."""
    assert extract_video_id("vid_08ySTcxh_cU.mp4") == "vid_08ySTcxh_cU"
    assert extract_video_id("vid_-V2r_9Um2Tw.mp4") == "vid_-V2r_9Um2Tw"
    assert extract_video_id("vid_123.mp4") == "vid_123"


@pytest.mark.fast
def test_extract_video_id_s3_uri():
    """Test video ID extraction from S3 URI."""
    assert extract_video_id("s3://bucket/prefix/vid_123.mp4") == "vid_123"
    assert extract_video_id("s3://my-bucket/videos/vid_-abc.mp4") == "vid_-abc"


@pytest.mark.fast
def test_inverted_to_jsonl(tmp_path):
    """Test conversion from label→videos to video→labels JSONL format."""
    label_to_videos = {
        "chocolate": {"vid_1", "vid_2"},
        "cheerful": {"vid_1"},
        "cars": {"vid_2", "vid_3"},
    }

    output_file = tmp_path / "output.jsonl"
    inverted_to_jsonl(label_to_videos, str(output_file))

    with jsonlines.open(output_file) as reader:
        results = list(reader)

    results_dict = {r["video_id"]: set(r["tags"]) for r in results}

    assert results_dict["vid_1"] == {"chocolate", "cheerful"}
    assert results_dict["vid_2"] == {"cars", "chocolate"}
    assert results_dict["vid_3"] == {"cars"}


@pytest.mark.fast
def test_inverted_to_jsonl_tags_sorted(tmp_path):
    """Test that tags in JSONL output are sorted."""
    label_to_videos = {
        "zebra": {"vid_1"},
        "apple": {"vid_1"},
        "banana": {"vid_1"},
    }

    output_file = tmp_path / "output.jsonl"
    inverted_to_jsonl(label_to_videos, str(output_file))

    with jsonlines.open(output_file) as reader:
        results = list(reader)

    assert len(results) == 1
    assert results[0]["tags"] == ["apple", "banana", "zebra"]
