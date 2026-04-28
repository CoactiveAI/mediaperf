"""
Component registry for the video tagging pipeline.

This module provides registries for pluggable pipeline components including
preprocessors, taggers, result writers, metric calculators, and result savers.
"""

from typing import Dict, Type

from .metrics.base import MetricCalculator
from .metrics.cost_metrics import CostMetricCalculator
from .metrics.llm_judge_metrics import LLMJudgeMetricCalculator
from .metrics.multilabel_classification import MultilabelClassificationMetric
from .metrics.timing_metrics import TimingMetricCalculator
from .models.anthropic_summarizer import AnthropicSummarizer
from .models.anthropic_vision import AnthropicVideoTagger
from .models.base import VideoTagger
from .models.base_judge import LLMJudge
from .models.base_summarizer import VideoSummarizer
from .models.bedrock_summarizer import BedrockVideoSummarizer
from .models.bedrock_vision import BedrockVideoTagger
from .models.openai_compatible_summarizer import OpenAICompatibleSummarizer
from .models.openai_compatible_vision import OpenAICompatibleTagger
from .models.openai_judge import OpenAILLMJudge
from .models.openai_summarizer import OpenAISummarizer
from .models.openai_vision import OpenAIVideoTagger
from .models.vertex_summarizer import VertexSummarizer
from .models.vertex_vision import VertexVideoTagger
from .preprocessing.base import VideoPreprocessor

# Import concrete implementations
from .preprocessing.frame_sampling import FrameSamplingPreprocessor
from .preprocessing.video_base64 import VideoBase64Preprocessor
from .savers.base import ResultSaver
from .savers.json_saver import JSONResultSaver
from .storage.base import FrameStorage
from .storage.gcs import GCSFrameStorage
from .storage.local import LocalFrameStorage
from .storage.s3 import S3FrameStorage
from .writers.base import ResultWriter
from .writers.jsonl_writer import JSONLResultWriter

PREPROCESSOR_REGISTRY: Dict[str, Type[VideoPreprocessor]] = {
    "frame_sampling": FrameSamplingPreprocessor,
    "video_base64": VideoBase64Preprocessor,
}

TAGGER_REGISTRY: Dict[str, Type[VideoTagger]] = {
    "anthropic_vision": AnthropicVideoTagger,
    "openai_vision": OpenAIVideoTagger,
    "bedrock_vision": BedrockVideoTagger,
    "vertex_vision": VertexVideoTagger,
    "openai_compatible_vision": OpenAICompatibleTagger,
}

SUMMARIZER_REGISTRY: Dict[str, Type[VideoSummarizer]] = {
    "anthropic_summarizer": AnthropicSummarizer,
    "bedrock_summarizer": BedrockVideoSummarizer,
    "openai_summarizer": OpenAISummarizer,
    "vertex_summarizer": VertexSummarizer,
    "openai_compatible_summarizer": OpenAICompatibleSummarizer,
}

LLM_JUDGE_REGISTRY: Dict[str, Type[LLMJudge]] = {
    "openai_judge": OpenAILLMJudge,
}

RESULT_WRITER_REGISTRY: Dict[str, Type[ResultWriter]] = {
    "jsonl": JSONLResultWriter,
}

METRICS_REGISTRY: Dict[str, Type[MetricCalculator]] = {
    "multilabel_classification": MultilabelClassificationMetric,
    "cost": CostMetricCalculator,
    "timing": TimingMetricCalculator,
    "llm_judge": LLMJudgeMetricCalculator,
}

RESULT_SAVER_REGISTRY: Dict[str, Type[ResultSaver]] = {
    "json": JSONResultSaver,
}

FRAME_STORAGE_REGISTRY: Dict[str, Type[FrameStorage]] = {
    "local": LocalFrameStorage,
    "s3": S3FrameStorage,
    "gcs": GCSFrameStorage,
}
