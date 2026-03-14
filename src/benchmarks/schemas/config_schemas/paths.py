from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    """Base paths configuration."""

    inputs_dir: str = Field(..., description="Input data directory")
    outputs_dir: str = Field(..., description="Output results directory")
    cache_dir: str = Field(..., description="Cache directory")


class TaggingPathsConfig(PathsConfig):
    """Paths configuration for tagging tasks."""

    label_file: str = Field(..., description="Ground truth labels file")
    tag_descriptions: Optional[str] = Field(None, description="Tag descriptions file")

    @model_validator(mode="after")
    def validate_files_exist(self) -> "TaggingPathsConfig":
        if not Path(self.label_file).is_absolute():
            label_path = Path(self.inputs_dir) / self.label_file
        else:
            label_path = Path(self.label_file)

        if not label_path.exists():
            raise ValueError(f"Label file does not exist: {label_path}")

        if self.tag_descriptions:
            if not Path(self.tag_descriptions).is_absolute():
                tag_desc_path = Path(self.inputs_dir) / self.tag_descriptions
            else:
                tag_desc_path = Path(self.tag_descriptions)

            if not tag_desc_path.exists():
                raise ValueError(f"Tag descriptions file does not exist: {tag_desc_path}")

        return self


class SummaryEvaluationPathsConfig(BaseModel):
    """Paths configuration for summary evaluation tasks."""

    outputs_dir: str = Field(..., description="Output results directory")
    summaries_file: str = Field(..., description="Generated summaries file")
    ground_truth_file: str = Field(..., description="Ground truth summaries file")

    @model_validator(mode="after")
    def validate_files_exist(self) -> "SummaryEvaluationPathsConfig":
        summaries_path = Path(self.summaries_file)
        if not summaries_path.exists():
            raise ValueError(f"Summaries file does not exist: {summaries_path}")

        gt_path = Path(self.ground_truth_file)
        if not gt_path.exists():
            raise ValueError(f"Ground truth file does not exist: {gt_path}")

        return self


class MediaConvertPathsConfig(BaseModel):
    """Paths configuration for MediaConvert tasks."""

    outputs_dir: str = Field(..., description="Output directory for conversion logs")
