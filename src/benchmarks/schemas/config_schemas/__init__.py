"""
Configuration validation schemas.

Validates configuration at load time to catch errors before expensive operations.
"""

import os
from typing import Any, Dict, Union

from pydantic import ValidationError

from ...enums import TaskType
from .task_configs import (
    CostBenchmarkTaskConfig,
    MediaConvertTaskConfig,
    PreprocessingOnlyConfig,
    StandardTaggingConfig,
    SummarizationTaskConfig,
    SummaryEvaluationTaskConfig,
)

TaskConfig = Union[
    StandardTaggingConfig,
    CostBenchmarkTaskConfig,
    SummarizationTaskConfig,
    SummaryEvaluationTaskConfig,
    PreprocessingOnlyConfig,
    MediaConvertTaskConfig,
]


def _is_env_var_missing(env_var_name: str) -> bool:
    """
    Check if an environment variable is missing or invalid.

    Args:
        env_var_name: Name of environment variable to check

    Returns:
        True if value is None, empty, or whitespace-only
    """
    value = os.getenv(env_var_name)
    return not value or not value.strip()


def _validate_environment_variables(config: TaskConfig) -> None:
    """
    Validate that required environment variables exist based on model types.

    Args:
        config: Validated task configuration

    Raises:
        ValueError: If required environment variables are missing
    """
    missing_vars = []

    if hasattr(config, "pipeline"):
        if hasattr(config.pipeline, "tagger"):
            tagger_type = config.pipeline.tagger.type
            if "openai" in tagger_type and _is_env_var_missing("OPENAI_API_KEY"):
                missing_vars.append("OPENAI_API_KEY (required for OpenAI models)")
            elif "bedrock" in tagger_type:
                if _is_env_var_missing("AWS_ACCESS_KEY_ID"):
                    missing_vars.append("AWS_ACCESS_KEY_ID (required for Bedrock models)")
                if _is_env_var_missing("AWS_SECRET_ACCESS_KEY"):
                    missing_vars.append("AWS_SECRET_ACCESS_KEY (required for Bedrock models)")
            elif "vertex" in tagger_type and _is_env_var_missing("GOOGLE_APPLICATION_CREDENTIALS"):
                missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS (required for Vertex AI models)")

        if hasattr(config.pipeline, "summarizer"):
            summarizer_type = config.pipeline.summarizer.type
            if "openai" in summarizer_type and _is_env_var_missing("OPENAI_API_KEY"):
                missing_vars.append("OPENAI_API_KEY (required for OpenAI models)")
            elif "bedrock" in summarizer_type:
                if _is_env_var_missing("AWS_ACCESS_KEY_ID"):
                    missing_vars.append("AWS_ACCESS_KEY_ID (required for Bedrock models)")
                if _is_env_var_missing("AWS_SECRET_ACCESS_KEY"):
                    missing_vars.append("AWS_SECRET_ACCESS_KEY (required for Bedrock models)")
            elif "vertex" in summarizer_type and _is_env_var_missing("GOOGLE_APPLICATION_CREDENTIALS"):
                missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS (required for Vertex AI models)")

        if hasattr(config.pipeline, "judge"):
            judge_type = config.pipeline.judge.type
            if "openai" in judge_type and _is_env_var_missing("OPENAI_API_KEY"):
                missing_vars.append("OPENAI_API_KEY (required for OpenAI judge)")

    if hasattr(config, "video_source"):
        if config.video_source.type.value == "s3":
            if _is_env_var_missing("AWS_ACCESS_KEY_ID"):
                missing_vars.append("AWS_ACCESS_KEY_ID (required for S3 video source)")
            if _is_env_var_missing("AWS_SECRET_ACCESS_KEY"):
                missing_vars.append("AWS_SECRET_ACCESS_KEY (required for S3 video source)")
        elif config.video_source.type.value == "gcs":
            if _is_env_var_missing("GOOGLE_APPLICATION_CREDENTIALS"):
                missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS (required for GCS video source)")

    missing_vars = list(dict.fromkeys(missing_vars))

    if missing_vars:
        raise ValueError(
            "Missing required environment variables:\n"
            + "\n".join(f"  - {var}" for var in missing_vars)
            + "\n\nPlease set these variables in your .env file or environment."
        )


def validate_config(config_dict: Dict[str, Any]) -> TaskConfig:
    """
    Validate configuration dictionary and return appropriate task config model.

    Args:
        config_dict: Raw configuration dictionary from YAML

    Returns:
        Validated task configuration model

    Raises:
        ValidationError: If configuration structure is invalid
        ValueError: If required environment variables are missing
    """
    task_type_str = config_dict.get("task_type", "standard_tagging")

    if isinstance(task_type_str, str):
        task_type = TaskType(task_type_str)
    else:
        task_type = task_type_str

    try:
        if task_type == TaskType.STANDARD_TAGGING:
            config = StandardTaggingConfig(**config_dict)
        elif task_type == TaskType.COST_BENCHMARK:
            config = CostBenchmarkTaskConfig(**config_dict)
        elif task_type == TaskType.SUMMARIZATION:
            config = SummarizationTaskConfig(**config_dict)
        elif task_type == TaskType.SUMMARY_EVALUATION:
            config = SummaryEvaluationTaskConfig(**config_dict)
        elif task_type == TaskType.PREPROCESSING_ONLY:
            config = PreprocessingOnlyConfig(**config_dict)
        elif task_type == TaskType.MEDIACONVERT:
            config = MediaConvertTaskConfig(**config_dict)
        else:
            raise ValueError(
                f"Unknown task_type '{task_type.value}'. Valid types: {', '.join(t.value for t in TaskType)}"
            )
    except ValidationError:
        raise

    _validate_environment_variables(config)

    return config
