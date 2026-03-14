from typing import List

from pydantic import BaseModel, Field


class TagPrediction(BaseModel):
    """Single tag with confidence from API response."""

    tag: str = Field(..., description="Tag from the allowed list")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class TagsOutput(BaseModel):
    """API response containing list of tag predictions."""

    tags: List[TagPrediction] = Field(default_factory=list, description="Detected tags with confidence scores")


class BinaryTagsOutput(BaseModel):
    """API response with binary tag presence (no confidence scores)."""

    tags: List[str] = Field(default_factory=list, description="List of present tags (binary - no confidence scores)")


class SummaryOutput(BaseModel):
    """API response containing video summary text."""

    summary: str = Field(..., description="Text summary of the video content")


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion."""

    criterion: str = Field(..., description="Name of the evaluation criterion")
    score: int = Field(..., ge=1, le=5, description="Score from 1-5")
    reasoning: str = Field(..., description="Brief explanation for the score")


class SummaryEvaluationOutput(BaseModel):
    """LLM judge evaluation of a summary."""

    criterion_scores: List[CriterionScore] = Field(..., description="Scores for each evaluation criterion")
    overall_reasoning: str = Field(..., description="Brief overall assessment of the summary quality")
