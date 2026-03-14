from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

from ...registry import FRAME_STORAGE_REGISTRY


class StorageS3Config(BaseModel):
    """S3 storage configuration."""

    bucket: str = Field(..., min_length=1)
    prefix: str = Field(default="frames")
    region_name: str = Field(default="us-east-1")


class StorageGCSConfig(BaseModel):
    """GCS storage configuration."""

    bucket: str = Field(..., min_length=1)
    prefix: str = Field(default="frames")


class StorageLocalConfig(BaseModel):
    """Local storage configuration."""

    base_path: str = Field(..., min_length=1)


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    type: str = Field(..., description="Storage backend type")
    config: Dict[str, Any] = Field(..., description="Storage-specific configuration")

    @field_validator("type")
    @classmethod
    def validate_storage_type(cls, v: str) -> str:
        if v not in FRAME_STORAGE_REGISTRY:
            available = list(FRAME_STORAGE_REGISTRY.keys())
            raise ValueError(f"Invalid storage type '{v}'. Available types: {available}")
        return v
