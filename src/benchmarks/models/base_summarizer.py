import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import ValidationError

from ..schemas import SummaryOutput


class VideoSummarizer(ABC):
    """Abstract base class for video summarization models."""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        inference_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.inference_params = inference_params or {}
        logger.info(
            f"Initialized {self.__class__.__name__} with model: {model_id}, inference_params: {self.inference_params}"
        )

    @abstractmethod
    def summarize_video(
        self,
        video_source: Any,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
        return_usage: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a text summary of a video."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model identifier/name."""
        pass

    def _load_prompts(
        self,
        model_prefix: str = None,
        prompts_dir: Path = None,
        system_prompt_file: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
    ) -> None:
        """
        Load prompt templates from files.

        Args:
            model_prefix: Prefix for default prompt files (e.g., "bedrock", "openai")
            prompts_dir: Directory containing prompt files
            system_prompt_file: Optional custom filename for system prompt (e.g., "custom_system.txt")
            user_prompt_file: Optional custom filename for user prompt (e.g., "universal_user.txt")

        Sets:
            self.system_prompt_template: System prompt text
            self.user_prompt_template: User prompt text
        """
        # Construct paths - use custom filenames if provided, otherwise use model_prefix pattern
        if system_prompt_file:
            system_path = prompts_dir / system_prompt_file
        else:
            system_path = prompts_dir / f"{model_prefix}_system.txt"

        if user_prompt_file:
            user_path = prompts_dir / user_prompt_file
        else:
            user_path = prompts_dir / f"{model_prefix}_user.txt"

        with open(system_path, "r") as f:
            self.system_prompt_template = f.read()

        with open(user_path, "r") as f:
            self.user_prompt_template = f.read()

    def _parse_summary_response(
        self, response_text: str, strip_markdown: bool = True, validate_schema: bool = True
    ) -> Dict[str, Any]:
        """
        Parse model response and extract summary.

        Args:
            response_text: Raw response text from model
            strip_markdown: Whether to strip markdown code blocks
            validate_schema: Whether to validate against SummaryOutput schema

        Returns:
            Dictionary with "summary" key containing the summary text
            Format: {"summary": "The video shows..."}
        """
        try:
            # Strip markdown code blocks if requested
            cleaned_text = response_text.strip()
            if strip_markdown:
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

            if validate_schema:
                # Parse and validate JSON against schema
                output = SummaryOutput.model_validate_json(cleaned_text)
                return {"summary": output.summary}
            else:
                # For plain text responses, just return as-is
                return {"summary": cleaned_text}

        except ValidationError as e:
            logger.error(f"Failed to validate summary response: {e}")
            logger.error(f"Response text: {response_text}")
            return {"summary": "", "error": f"Validation failed: {str(e)}"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            # Fallback: treat as plain text
            return {"summary": response_text.strip()}
