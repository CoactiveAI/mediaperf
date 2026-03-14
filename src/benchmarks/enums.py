from enum import Enum


class VideoSourceType(Enum):
    """Video source type in configuration."""

    S3 = "s3"
    GCS = "gcs"
    LOCAL = "local"


class VideoSourceMode(Enum):
    """Video source mode (how videos are accessed)."""

    FROM_S3 = "from_s3"
    FROM_GCS = "from_gcs"
    DOWNLOAD_LOCAL = "local"


class VideoSelectionMode(Enum):
    """Video selection mode in configuration."""

    FIXED = "fixed"
    RANDOM = "random"
    ALL = "all"
    DURATION_BASED = "duration_based"


class TaskType(Enum):
    """Task type in configuration."""

    STANDARD_TAGGING = "standard_tagging"
    COST_BENCHMARK = "cost_benchmark"
    INFERENCE_ONLY = "inference_only"
    SUMMARIZATION = "summarization"
    SUMMARY_EVALUATION = "summary_evaluation"
    PREPROCESSING_ONLY = "preprocessing_only"
    MEDIACONVERT = "mediaconvert"


class LogLevel(Enum):
    """Logging level."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AggregationMethod(Enum):
    """Score aggregation method for evaluation."""

    SUM = "sum"
    AVERAGE = "average"


class DurationOrder(Enum):
    """Order for duration-based video selection."""

    LONGEST = "longest"
    SHORTEST = "shortest"


class OutputFormat(Enum):
    """Frame output format for preprocessing."""

    BASE64 = "base64"
    BYTES = "bytes"
