from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ...registry import PREPROCESSOR_REGISTRY


class PreprocessorConfig(BaseModel):
    """Preprocessor configuration."""

    type: Optional[str] = Field(..., description="Preprocessor type (or null to disable)")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Preprocessor-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_preprocessor_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in PREPROCESSOR_REGISTRY:
            available = list(PREPROCESSOR_REGISTRY.keys())
            raise ValueError(f"Invalid preprocessor type '{v}'. Available types: {available} (or use null to disable)")
        return v

    @model_validator(mode="after")
    def validate_cache_with_storage(self) -> "PreprocessorConfig":
        """Validate that run_id_for_cache requires storage config (Bug Fix #1)."""
        if self.config and self.config.get("run_id_for_cache"):
            if not self.config.get("storage"):
                raise ValueError(
                    "preprocessor.config.run_id_for_cache is set but 'storage' is not configured. "
                    "When using cache, you must provide a storage backend config."
                )
        return self
