from typing import Optional

from pydantic import BaseModel, Field

from ...enums import LogLevel


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")


class TrackingConfig(BaseModel):
    """Tracking configuration for metrics and progress."""

    return_usage: bool = Field(default=True, description="Track token usage and timing")
    progress_interval: int = Field(default=10, ge=1, description="Log progress every N videos")


class PricingConfig(BaseModel):
    """Pricing configuration for cost calculation."""

    input_per_1m_tokens: float = Field(..., ge=0, description="Cost per 1M input tokens")
    output_per_1m_tokens: float = Field(..., ge=0, description="Cost per 1M output tokens")
    model_name: Optional[str] = Field(None, description="Model name for pricing")
