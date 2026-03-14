"""
Builder functions for constructing pipeline components from configuration.
"""

import os
from typing import Any, Dict, Optional

from .metrics.base import MetricCalculator
from .models.base import VideoTagger
from .models.base_judge import LLMJudge
from .models.base_summarizer import VideoSummarizer
from .preprocessing.base import VideoPreprocessor
from .registry import (
    FRAME_STORAGE_REGISTRY,
    LLM_JUDGE_REGISTRY,
    METRICS_REGISTRY,
    PREPROCESSOR_REGISTRY,
    RESULT_SAVER_REGISTRY,
    RESULT_WRITER_REGISTRY,
    SUMMARIZER_REGISTRY,
    TAGGER_REGISTRY,
)
from .savers.base import ResultSaver
from .storage.base import FrameStorage
from .writers.base import ResultWriter


def build_preprocessor(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    cache_run_id: Optional[str] = None,
    full_config: Optional[Dict[str, Any]] = None,
) -> VideoPreprocessor:
    """
    Args:
        config: Preprocessor configuration with 'type' and 'config' keys
        run_id: Run identifier for storage (current run)
        cache_run_id: Optional run identifier to check for cached frames
        full_config: Full configuration dictionary (for accessing video_source, paths, etc.)

    Returns:
        Instantiated VideoPreprocessor
    """
    preprocessor_type = config["type"]
    preprocessor_config = config.get("config", {}).copy()

    if preprocessor_type not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown preprocessor type: {preprocessor_type}. Available: {list(PREPROCESSOR_REGISTRY.keys())}"
        )

    # Handle storage backend if specified
    if "storage" in preprocessor_config:
        storage_config = preprocessor_config.pop("storage")
        storage_backend = build_frame_storage(storage_config)
        preprocessor_config["storage_backend"] = storage_backend

    # Pass run_id if provided
    if run_id is not None:
        preprocessor_config["run_id"] = run_id

    # Pass cache_run_id if provided
    if cache_run_id is not None:
        preprocessor_config["cache_run_id"] = cache_run_id

    # Pass full config if provided (for frame_sampling to access video_source)
    if full_config is not None:
        preprocessor_config["config"] = full_config

    preprocessor_class = PREPROCESSOR_REGISTRY[preprocessor_type]
    return preprocessor_class(**preprocessor_config)


def build_tagger(config: Dict[str, Any]) -> VideoTagger:
    """
    Build a tagger from configuration.

    Args:
        config: Tagger configuration with 'type' and 'config' keys

    Returns:
        Instantiated VideoTagger

    Example config:
        {
            "type": "openai_vision",
            "config": {
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "reasoning_effort": "minimal"
            }
        }
    """
    tagger_type = config["type"]
    tagger_config = config.get("config", {}).copy()  # Copy to avoid modifying original

    if tagger_type not in TAGGER_REGISTRY:
        raise ValueError(f"Unknown tagger type: {tagger_type}. Available: {list(TAGGER_REGISTRY.keys())}")

    # Handle API key environment variable lookup
    if "api_key_env" in tagger_config:
        api_key_env_name = tagger_config.pop("api_key_env")
        tagger_config["api_key"] = os.getenv(api_key_env_name)

    tagger_class = TAGGER_REGISTRY[tagger_type]
    return tagger_class(**tagger_config)


def build_summarizer(config: Dict[str, Any]) -> VideoSummarizer:
    """
    Build a summarizer from configuration.

    Args:
        config: Summarizer configuration with 'type' and 'config' keys

    Returns:
        Instantiated VideoSummarizer

    Example config:
        {
            "type": "bedrock_summarizer",
            "config": {
                "model_id": "us.amazon.nova-lite-v1:0",
                "model_name": "nova-lite",
                "return_usage": true
            }
        }
    """
    summarizer_type = config["type"]
    summarizer_config = config.get("config", {}).copy()

    if summarizer_type not in SUMMARIZER_REGISTRY:
        raise ValueError(f"Unknown summarizer type: {summarizer_type}. Available: {list(SUMMARIZER_REGISTRY.keys())}")

    # Handle API key environment variable lookup
    if "api_key_env" in summarizer_config:
        api_key_env_name = summarizer_config.pop("api_key_env")
        summarizer_config["api_key"] = os.getenv(api_key_env_name)

    summarizer_class = SUMMARIZER_REGISTRY[summarizer_type]
    return summarizer_class(**summarizer_config)


def build_llm_judge(config: Dict[str, Any]) -> LLMJudge:
    """
    Build an LLM judge from configuration.

    Args:
        config: Judge configuration with 'type' and 'config' keys

    Returns:
        Instantiated LLMJudge

    Example config:
        {
            "type": "openai_judge",
            "config": {
                "model_id": "gpt-4o",
                "model_name": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "reasoning_effort": "medium"
            }
        }
    """
    judge_type = config["type"]
    judge_config = config.get("config", {}).copy()

    if judge_type not in LLM_JUDGE_REGISTRY:
        raise ValueError(f"Unknown judge type: {judge_type}. Available: {list(LLM_JUDGE_REGISTRY.keys())}")

    # Handle API key environment variable lookup
    if "api_key_env" in judge_config:
        api_key_env_name = judge_config.pop("api_key_env")
        judge_config["api_key"] = os.getenv(api_key_env_name)

    judge_class = LLM_JUDGE_REGISTRY[judge_type]
    return judge_class(**judge_config)


def build_result_writer(config: Dict[str, Any]) -> ResultWriter:
    """
    Build a result writer from configuration.

    Args:
        config: Result writer configuration with 'type' and 'config' keys

    Returns:
        Instantiated ResultWriter

    Example config:
        {
            "type": "jsonl",
            "config": {
                "output_path": "data/outputs/predictions.jsonl"
            }
        }
    """
    writer_type = config["type"]
    writer_config = config.get("config", {})

    if writer_type not in RESULT_WRITER_REGISTRY:
        raise ValueError(f"Unknown result writer type: {writer_type}. Available: {list(RESULT_WRITER_REGISTRY.keys())}")

    writer_class = RESULT_WRITER_REGISTRY[writer_type]
    return writer_class(**writer_config)


def build_metric_calculator(config: Dict[str, Any]) -> MetricCalculator:
    """
    Build a metric calculator from configuration.

    Args:
        config: Metric calculator configuration with 'type' and 'config' keys

    Returns:
        Instantiated MetricCalculator

    Example config:
        {
            "type": "multilabel_classification",
            "config": {
                "pred_key": "video_level_tags",
                "gt_key": "tags"
            }
        }
    """
    metric_type = config["type"]
    metric_config = config.get("config", {})

    if metric_type not in METRICS_REGISTRY:
        raise ValueError(f"Unknown metric calculator type: {metric_type}. Available: {list(METRICS_REGISTRY.keys())}")

    metric_class = METRICS_REGISTRY[metric_type]
    return metric_class(**metric_config)


def build_result_saver(config: Dict[str, Any]) -> ResultSaver:
    """
    Build a result saver from configuration.

    Args:
        config: Result saver configuration with 'type' and 'config' keys

    Returns:
        Instantiated ResultSaver

    Example config:
        {
            "type": "json",
            "config": {
                "include_timestamp": true
            }
        }
    """
    saver_type = config["type"]
    saver_config = config.get("config", {})

    if saver_type not in RESULT_SAVER_REGISTRY:
        raise ValueError(f"Unknown result saver type: {saver_type}. Available: {list(RESULT_SAVER_REGISTRY.keys())}")

    saver_class = RESULT_SAVER_REGISTRY[saver_type]
    return saver_class(**saver_config)


def build_frame_storage(config: Dict[str, Any]) -> FrameStorage:
    """
    Args:
        config: Frame storage configuration with 'type' and 'config' keys

    Returns:
        Instantiated FrameStorage
    """
    storage_type = config["type"]
    storage_config = config.get("config", {})

    if storage_type not in FRAME_STORAGE_REGISTRY:
        raise ValueError(
            f"Unknown frame storage type: {storage_type}. Available: {list(FRAME_STORAGE_REGISTRY.keys())}"
        )

    storage_class = FRAME_STORAGE_REGISTRY[storage_type]
    return storage_class(**storage_config)
