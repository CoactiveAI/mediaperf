"""Test fixtures for YouTube Ads label data."""

import json
from pathlib import Path
from typing import Dict


def create_youtube_ads_label_json() -> Dict:
    """
    Create synthetic label JSON matching YouTube Ads format.

    Simplified version with essential fields for testing.
    """
    return {
        "video_Effective": {
            "ABC123": 7,
            "DEF456": 3,
            "GHI789": 8,
        },
        "video_Funny": {
            "ABC123": 0.8,
            "DEF456": 0.3,
            "GHI789": 0.6,
        },
        "video_Exciting": {
            "ABC123": 0.6,
            "DEF456": 0.9,
            "GHI789": 0.4,
        },
        "video_Language": {
            "ABC123": "1",
            "DEF456": "0",
            "GHI789": "1",
        },
        "video_Sentiments": {
            "ABC123": 1,
            "DEF456": 2,
            "GHI789": 1,
        },
        "video_Topics": {
            "ABC123": 1,
            "DEF456": 2,
            "GHI789": 3,
        },
        "Sentiments": {
            "1": {"abbreviation": "cheerful", "description": "Happy and upbeat"},
            "2": {"abbreviation": "sad", "description": "Melancholic"},
        },
        "Topics": {
            "1": {"abbreviation": "food", "description": "Food and beverages"},
            "2": {"abbreviation": "tech", "description": "Technology products"},
            "3": {"abbreviation": "cars", "description": "Automotive"},
        },
    }


def create_label_descriptions() -> Dict[str, str]:
    """Create label descriptions dict."""
    return {
        "food": "Food and beverage products",
        "tech": "Technology and gadgets",
        "cars": "Automotive products",
        "cheerful": "Upbeat and positive tone",
        "sad": "Sad or melancholic tone",
        "funny": "Humorous content",
        "exciting": "Action-packed content",
        "effective": "Highly effective advertisement",
    }


def write_label_json(label_data: Dict, output_path: Path):
    """Write label data to JSON file."""
    with open(output_path, "w") as f:
        json.dump(label_data, f)
