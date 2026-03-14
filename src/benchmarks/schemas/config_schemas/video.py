import os
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ...enums import DurationOrder, VideoSelectionMode, VideoSourceMode, VideoSourceType


class VideoSourceS3Config(BaseModel):
    """S3-specific video source configuration."""

    bucket: str = Field(..., min_length=1, description="S3 bucket name")
    prefix: str = Field(..., description="S3 prefix/folder")
    region: str = Field(default="us-east-1", description="AWS region")


class VideoSourceGCSConfig(BaseModel):
    """GCS-specific video source configuration."""

    bucket: str = Field(..., min_length=1, description="GCS bucket name")
    prefix: str = Field(..., description="GCS prefix/folder")


class VideoSourceLocalConfig(BaseModel):
    """Local filesystem video source configuration."""

    base_path: str = Field(..., min_length=1, description="Base directory for local videos")

    @field_validator("base_path")
    @classmethod
    def validate_base_path_exists(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Local video base_path does not exist: {v}")
        if not os.path.isdir(v):
            raise ValueError(f"Local video base_path is not a directory: {v}")
        return v


class VideoSourceConfig(BaseModel):
    """Video source configuration with type-based validation."""

    type: VideoSourceType = Field(..., description="Video source type")
    mode: Optional[VideoSourceMode] = Field(None, description="Video access mode")
    s3: Optional[VideoSourceS3Config] = None
    gcs: Optional[VideoSourceGCSConfig] = None
    local: Optional[VideoSourceLocalConfig] = None

    @model_validator(mode="after")
    def validate_source_consistency(self) -> "VideoSourceConfig":
        if self.type == VideoSourceType.S3:
            if self.s3 is None:
                raise ValueError("video_source.type='s3' requires 's3' config with bucket, prefix, region")
            if self.mode and self.mode not in [VideoSourceMode.FROM_S3, VideoSourceMode.DOWNLOAD_LOCAL]:
                raise ValueError(f"video_source.type='s3' expects mode 'from_s3' or 'local', got '{self.mode.value}'")
        elif self.type == VideoSourceType.GCS:
            if self.gcs is None:
                raise ValueError("video_source.type='gcs' requires 'gcs' config with bucket, prefix")
            if self.mode and self.mode not in [VideoSourceMode.FROM_GCS, VideoSourceMode.DOWNLOAD_LOCAL]:
                raise ValueError(f"video_source.type='gcs' expects mode 'from_gcs' or 'local', got '{self.mode.value}'")
        elif self.type == VideoSourceType.LOCAL:
            if self.local is None:
                raise ValueError("video_source.type='local' requires 'local' config with base_path")
            if self.mode and self.mode != VideoSourceMode.DOWNLOAD_LOCAL:
                raise ValueError(f"video_source.type='local' expects mode 'local', got '{self.mode.value}'")

        return self


class VideoSelectionConfig(BaseModel):
    """Video selection configuration with mode-based validation."""

    mode: VideoSelectionMode = Field(..., description="Video selection mode")
    fixed_videos: Optional[List[str]] = Field(None, description="List of video IDs for fixed mode")
    random_count: Optional[int] = Field(None, ge=1, description="Number of random videos")
    target_minutes: Optional[int] = Field(None, ge=1, description="Target total minutes for duration_based")
    metadata_file: Optional[str] = Field(None, description="Metadata file for duration_based")
    order: Optional[DurationOrder] = Field(None, description="Order for duration_based selection")

    @model_validator(mode="after")
    def validate_selection_mode(self) -> "VideoSelectionConfig":
        if self.mode == VideoSelectionMode.FIXED:
            if not self.fixed_videos:
                raise ValueError("video_selection.mode='fixed' requires 'fixed_videos' list with at least one video ID")
        elif self.mode == VideoSelectionMode.RANDOM:
            if self.random_count is None:
                raise ValueError("video_selection.mode='random' requires 'random_count' (integer >= 1)")
        elif self.mode == VideoSelectionMode.DURATION_BASED:
            if self.target_minutes is None:
                raise ValueError("video_selection.mode='duration_based' requires 'target_minutes' (integer >= 1)")
            if self.metadata_file is None:
                raise ValueError("video_selection.mode='duration_based' requires 'metadata_file' path")
            if not os.path.exists(self.metadata_file):
                raise ValueError(f"video_selection.metadata_file does not exist: {self.metadata_file}")

        return self
