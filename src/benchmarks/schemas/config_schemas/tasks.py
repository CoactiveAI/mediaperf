import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ...enums import AggregationMethod


class InferenceConfig(BaseModel):
    """Inference configuration for tagging tasks."""

    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_tags: Optional[List[str]] = None


class LabelsConfig(BaseModel):
    """Labels configuration for tagging tasks."""

    allowed_tags: Optional[List[str]] = None
    tag_subset: Optional[List[str]] = None


class CostBenchmarkIterationConfig(BaseModel):
    """Single iteration configuration for cost benchmark."""

    iteration: int = Field(..., ge=1, description="Iteration number")
    prompt_path: str = Field(..., description="Path to iteration-specific prompt")
    skip_tag_injection: bool = Field(default=False, description="Skip tag injection for this iteration")

    @field_validator("prompt_path")
    @classmethod
    def validate_prompt_path_exists(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Iteration prompt file does not exist: {v}")
        return v


class CostBenchmarkConfig(BaseModel):
    """Cost benchmark specific configuration."""

    iterations: List[CostBenchmarkIterationConfig] = Field(
        ..., min_length=1, description="List of iteration configurations"
    )


class EvaluationCriterionConfig(BaseModel):
    """Single evaluation criterion configuration."""

    name: str = Field(..., min_length=1, description="Criterion name")
    description: str = Field(..., min_length=1, description="Criterion description")
    min_score: int = Field(default=1, ge=1, description="Minimum score")
    max_score: int = Field(default=5, ge=1, description="Maximum score")


class EvaluationConfig(BaseModel):
    """Evaluation configuration for summary evaluation."""

    aggregation_method: AggregationMethod = Field(..., description="Score aggregation method")
    criteria: List[EvaluationCriterionConfig] = Field(..., min_length=1, description="Evaluation criteria")


class MediaConvertConfig(BaseModel):
    """MediaConvert-specific configuration for video conversion."""

    output_prefix: str = Field(..., min_length=1, description="S3 prefix for converted videos")
    role_arn: str = Field(..., min_length=1, description="IAM role ARN for MediaConvert")
    quality: str = Field(..., description="Encoding quality preset")
    max_bitrate: int = Field(..., gt=0, description="Maximum bitrate in bps")
    max_height: Optional[int] = Field(None, gt=0, description="Maximum video height for downscaling")
    rate_limit: int = Field(default=10, gt=0, description="Jobs per second")
    poll_interval: int = Field(default=30, gt=0, description="Seconds between status checks")
    progress_interval: int = Field(default=50, gt=0, description="Log progress every N videos")
    check_codec: bool = Field(default=False, description="Skip videos already in target codec")
    avg_duration: float = Field(..., gt=0, description="Average video duration for cost estimation")
    tags: Optional[Dict[str, str]] = Field(None, description="AWS cost allocation tags")
