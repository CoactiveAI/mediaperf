"""Tests for model response parsing logic in base.py."""

import pytest

from benchmarks.models.base import VideoTagger


class MockTagger(VideoTagger):
    """Minimal implementation for testing base class."""

    def tag_video(self, video_source, allowed_tags, tag_definitions, **kwargs):
        return {"tags": []}

    def get_model_name(self):
        return "mock-tagger"


@pytest.mark.fast
def test_parse_valid_json_response():
    """Test parsing a valid JSON response with tags."""
    tagger = MockTagger()

    response_text = """
    {
        "tags": [
            {"tag": "chocolate", "confidence": 0.9},
            {"tag": "cheerful", "confidence": 0.85}
        ]
    }
    """

    result = tagger._parse_response(response_text, allowed_tags=["chocolate", "cheerful", "cars"])

    assert "tags" in result
    assert len(result["tags"]) == 2
    assert result["tags"][0]["tag"] == "chocolate"
    assert result["tags"][0]["confidence"] == 0.9
    assert result["tags"][1]["tag"] == "cheerful"
    assert result["tags"][1]["confidence"] == 0.85


@pytest.mark.fast
def test_parse_invalid_json():
    """Test handling of malformed JSON."""
    tagger = MockTagger()

    bad_json = "not valid json at all"

    result = tagger._parse_response(bad_json, allowed_tags=["chocolate"])

    assert result["tags"] == []
    assert "error" in result
    assert "Validation failed" in result["error"]


@pytest.mark.fast
def test_parse_response_with_invalid_confidence():
    """Test handling when confidence is out of bounds (< 0 or > 1)."""
    tagger = MockTagger()

    response_text = """
    {
        "tags": [
            {"tag": "chocolate", "confidence": 1.5},
            {"tag": "cheerful", "confidence": -0.1}
        ]
    }
    """

    result = tagger._parse_response(response_text, allowed_tags=["chocolate", "cheerful"])

    assert result["tags"] == []
    assert "error" in result
    assert "Validation failed" in result["error"]


@pytest.mark.fast
def test_parse_response_filters_disallowed_tags():
    """Test that tags not in allowed_list are filtered out."""
    tagger = MockTagger()

    response_text = """
    {
        "tags": [
            {"tag": "chocolate", "confidence": 0.9},
            {"tag": "forbidden_tag", "confidence": 0.8},
            {"tag": "another_bad_tag", "confidence": 0.7}
        ]
    }
    """

    allowed = ["chocolate"]
    result = tagger._parse_response(response_text, allowed_tags=allowed)

    assert len(result["tags"]) == 1
    assert result["tags"][0]["tag"] == "chocolate"


@pytest.mark.fast
def test_parse_response_normalizes_tags():
    """Test tag normalization: uppercase → lowercase, spaces → underscores."""
    tagger = MockTagger()

    response_text = """
    {
        "tags": [
            {"tag": "Happy Mood", "confidence": 0.9}
        ]
    }
    """

    allowed = ["happy_mood"]
    result = tagger._parse_response(response_text, allowed_tags=allowed)

    assert len(result["tags"]) == 1
    assert result["tags"][0]["tag"] == "happy_mood"


@pytest.mark.fast
def test_parse_response_strips_markdown_code_blocks():
    """Test that markdown code blocks are stripped (Gemini compatibility)."""
    tagger = MockTagger()

    response_text = """```json
    {
        "tags": [
            {"tag": "chocolate", "confidence": 0.9}
        ]
    }
    ```"""

    result = tagger._parse_response(response_text, allowed_tags=["chocolate"])

    assert len(result["tags"]) == 1
    assert result["tags"][0]["tag"] == "chocolate"


@pytest.mark.fast
def test_parse_response_empty_tags():
    """Test that empty tags list is handled correctly."""
    tagger = MockTagger()

    response_text = '{"tags": []}'

    result = tagger._parse_response(response_text, allowed_tags=["chocolate"])

    assert result == {"tags": []}


@pytest.mark.fast
def test_parse_response_case_insensitive_filtering():
    """Test that tag filtering is case-insensitive after normalization."""
    tagger = MockTagger()

    response_text = '{"tags": [{"tag": "Chocolate", "confidence": 0.9}]}'

    allowed = ["chocolate"]
    result = tagger._parse_response(response_text, allowed_tags=allowed)

    assert len(result["tags"]) == 1
    assert result["tags"][0]["tag"] == "chocolate"
