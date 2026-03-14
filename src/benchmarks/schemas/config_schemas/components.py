from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from ...registry import (
    LLM_JUDGE_REGISTRY,
    METRICS_REGISTRY,
    RESULT_SAVER_REGISTRY,
    RESULT_WRITER_REGISTRY,
    SUMMARIZER_REGISTRY,
    TAGGER_REGISTRY,
)


class TaggerConfig(BaseModel):
    """Tagger configuration."""

    type: str = Field(..., description="Tagger type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Tagger-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_tagger_type(cls, v: str) -> str:
        if v not in TAGGER_REGISTRY:
            available = list(TAGGER_REGISTRY.keys())
            raise ValueError(f"Invalid tagger type '{v}'. Available types: {available}")
        return v


class SummarizerConfig(BaseModel):
    """Summarizer configuration."""

    type: str = Field(..., description="Summarizer type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Summarizer-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_summarizer_type(cls, v: str) -> str:
        if v not in SUMMARIZER_REGISTRY:
            available = list(SUMMARIZER_REGISTRY.keys())
            raise ValueError(f"Invalid summarizer type '{v}'. Available types: {available}")
        return v


class JudgeConfig(BaseModel):
    """LLM judge configuration."""

    type: str = Field(..., description="Judge type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Judge-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_judge_type(cls, v: str) -> str:
        if v not in LLM_JUDGE_REGISTRY:
            available = list(LLM_JUDGE_REGISTRY.keys())
            raise ValueError(f"Invalid judge type '{v}'. Available types: {available}")
        return v


class ResultWriterConfig(BaseModel):
    """Result writer configuration."""

    type: str = Field(..., description="Result writer type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Writer-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_writer_type(cls, v: str) -> str:
        if v not in RESULT_WRITER_REGISTRY:
            available = list(RESULT_WRITER_REGISTRY.keys())
            raise ValueError(f"Invalid result writer type '{v}'. Available types: {available}")
        return v


class MetricsConfig(BaseModel):
    """Metrics configuration."""

    type: str = Field(..., description="Metrics calculator type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Metrics-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_metrics_type(cls, v: str) -> str:
        if v not in METRICS_REGISTRY:
            available = list(METRICS_REGISTRY.keys())
            raise ValueError(f"Invalid metrics type '{v}'. Available types: {available}")
        return v


class ResultSaverConfig(BaseModel):
    """Result saver configuration."""

    type: str = Field(..., description="Result saver type")
    config: Dict[str, Any] = Field(default_factory=dict, description="Saver-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_saver_type(cls, v: str) -> str:
        if v not in RESULT_SAVER_REGISTRY:
            available = list(RESULT_SAVER_REGISTRY.keys())
            raise ValueError(f"Invalid result saver type '{v}'. Available types: {available}")
        return v
