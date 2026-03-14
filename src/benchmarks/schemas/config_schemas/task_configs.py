from typing import Optional

from pydantic import BaseModel, Field

from ...enums import TaskType
from .base import LoggingConfig, PricingConfig, TrackingConfig
from .components import (
    JudgeConfig,
    MetricsConfig,
    ResultSaverConfig,
    ResultWriterConfig,
    SummarizerConfig,
    TaggerConfig,
)
from .paths import MediaConvertPathsConfig, PathsConfig, SummaryEvaluationPathsConfig, TaggingPathsConfig
from .preprocessor import PreprocessorConfig
from .tasks import CostBenchmarkConfig, EvaluationConfig, InferenceConfig, LabelsConfig, MediaConvertConfig
from .video import VideoSelectionConfig, VideoSourceConfig


class BaseTaskConfig(BaseModel):
    """Base configuration for all tasks."""

    task_type: TaskType = Field(default=TaskType.STANDARD_TAGGING, description="Task type")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    tracking: TrackingConfig = Field(default_factory=TrackingConfig, description="Tracking configuration")
    pricing: Optional[PricingConfig] = None
    result_saver: ResultSaverConfig = Field(..., description="Result saver configuration")


class PipelineConfig(BaseModel):
    """Base pipeline configuration."""

    preprocessor: Optional[PreprocessorConfig] = None
    result_writer: ResultWriterConfig = Field(..., description="Result writer configuration")
    metrics: Optional[MetricsConfig] = None


class StandardTaggingPipelineConfig(PipelineConfig):
    """Pipeline configuration for standard tagging."""

    tagger: TaggerConfig = Field(..., description="Tagger configuration")


class StandardTaggingConfig(BaseTaskConfig):
    """Configuration for standard tagging task."""

    task_type: TaskType = Field(default=TaskType.STANDARD_TAGGING, frozen=True)
    pipeline: StandardTaggingPipelineConfig = Field(..., description="Pipeline configuration")
    video_source: VideoSourceConfig = Field(..., description="Video source configuration")
    video_selection: VideoSelectionConfig = Field(..., description="Video selection configuration")
    paths: TaggingPathsConfig = Field(..., description="Paths configuration")
    inference: Optional[InferenceConfig] = None
    labels: Optional[LabelsConfig] = None


class CostBenchmarkPipelineConfig(PipelineConfig):
    """Pipeline configuration for cost benchmark."""

    tagger: TaggerConfig = Field(..., description="Tagger configuration")


class CostBenchmarkTaskConfig(BaseTaskConfig):
    """Configuration for cost benchmark task."""

    task_type: TaskType = Field(default=TaskType.COST_BENCHMARK, frozen=True)
    pipeline: CostBenchmarkPipelineConfig = Field(..., description="Pipeline configuration")
    video_source: VideoSourceConfig = Field(..., description="Video source configuration")
    video_selection: VideoSelectionConfig = Field(..., description="Video selection configuration")
    paths: TaggingPathsConfig = Field(..., description="Paths configuration")
    cost_benchmark: CostBenchmarkConfig = Field(..., description="Cost benchmark configuration")
    inference: Optional[InferenceConfig] = None
    labels: Optional[LabelsConfig] = None


class SummarizationPipelineConfig(PipelineConfig):
    """Pipeline configuration for summarization."""

    summarizer: SummarizerConfig = Field(..., description="Summarizer configuration")


class SummarizationTaskConfig(BaseTaskConfig):
    """Configuration for summarization task."""

    task_type: TaskType = Field(default=TaskType.SUMMARIZATION, frozen=True)
    pipeline: SummarizationPipelineConfig = Field(..., description="Pipeline configuration")
    video_source: VideoSourceConfig = Field(..., description="Video source configuration")
    video_selection: VideoSelectionConfig = Field(..., description="Video selection configuration")
    paths: PathsConfig = Field(..., description="Paths configuration")


class SummaryEvaluationPipelineConfig(PipelineConfig):
    """Pipeline configuration for summary evaluation."""

    judge: JudgeConfig = Field(..., description="Judge configuration")


class SummaryEvaluationTaskConfig(BaseTaskConfig):
    """Configuration for summary evaluation task."""

    task_type: TaskType = Field(default=TaskType.SUMMARY_EVALUATION, frozen=True)
    pipeline: SummaryEvaluationPipelineConfig = Field(..., description="Pipeline configuration")
    paths: SummaryEvaluationPathsConfig = Field(..., description="Paths configuration")
    evaluation: EvaluationConfig = Field(..., description="Evaluation configuration")


class PreprocessingOnlyPipelineConfig(BaseModel):
    """Pipeline configuration for preprocessing-only task."""

    preprocessor: PreprocessorConfig = Field(..., description="Preprocessor configuration")


class PreprocessingOnlyPathsConfig(BaseModel):
    """Minimal paths configuration for preprocessing-only task."""

    cache_dir: str = Field(..., description="Cache directory for video downloads")


class PreprocessingOnlyConfig(BaseModel):
    """Configuration for preprocessing-only task (standalone frame extraction)."""

    task_type: TaskType = Field(default=TaskType.PREPROCESSING_ONLY, frozen=True)
    video_source: VideoSourceConfig = Field(..., description="Video source configuration")
    video_selection: VideoSelectionConfig = Field(..., description="Video selection configuration")
    pipeline: PreprocessingOnlyPipelineConfig = Field(..., description="Pipeline configuration")
    paths: PreprocessingOnlyPathsConfig = Field(..., description="Paths configuration")
    logging: Optional[LoggingConfig] = Field(default_factory=LoggingConfig, description="Logging configuration")


class MediaConvertTaskConfig(BaseModel):
    """Configuration for MediaConvert video conversion task."""

    task_type: TaskType = Field(default=TaskType.MEDIACONVERT, frozen=True)
    video_source: VideoSourceConfig = Field(..., description="Video source configuration")
    video_selection: VideoSelectionConfig = Field(..., description="Video selection configuration")
    mediaconvert: MediaConvertConfig = Field(..., description="MediaConvert configuration")
    paths: MediaConvertPathsConfig = Field(..., description="Paths configuration")
    logging: Optional[LoggingConfig] = Field(default_factory=LoggingConfig, description="Logging configuration")
