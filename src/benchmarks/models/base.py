import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import ValidationError

from ..schemas import TagsOutput


class VideoTagger(ABC):
    """Abstract base class for video tagging models."""

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
    def tag_video(
        self, video_source: str, allowed_tags: List[str], tag_definitions: Dict[str, str], **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Tag a video with allowed tags and confidence scores."""
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

    def _build_user_prompt(
        self,
        allowed_tags: List[str],
        tag_definitions: Dict[str, str],
        custom_user_prompt: Optional[str] = None,
    ) -> str:
        """
        Build user prompt with formatted tag definitions.

        This method formats tag definitions and injects them into the prompt template.
        Supports two template styles:
        1. Template with {tags_formatted} placeholder (uses .format())
        2. Template without placeholder (appends tags at the end)

        If tag_definitions is empty or None, returns prompt as-is (for self-contained prompts).

        Args:
            allowed_tags: List of allowed tag names
            tag_definitions: Dictionary mapping tag names to descriptions (empty to skip injection)
            custom_user_prompt: Optional custom prompt to use instead of template

        Returns:
            Formatted prompt string with tag definitions (or as-is if tag_definitions empty)
        """
        # If tag_definitions is empty/None, return prompt as-is (self-contained)
        if not tag_definitions:
            final_prompt = custom_user_prompt if custom_user_prompt else self.user_prompt_template
            logger.debug("Skipping tag injection (self-contained prompt)")
            return final_prompt

        # Format tag definitions as list
        tag_list = []
        for tag in allowed_tags:
            definition = tag_definitions.get(tag, "")
            tag_list.append(f"- {tag}: {definition}")

        tags_formatted = "\n".join(tag_list)

        logger.debug(f"Injecting {len(allowed_tags)} tags into prompt")

        # Use custom prompt if provided, otherwise use template
        prompt_template = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        # Smart fallback: check if template has placeholder
        if "{tags_formatted}" in prompt_template:
            final_prompt = prompt_template.format(tags_formatted=tags_formatted)
        else:
            # Fallback: append tags manually for templates without placeholder
            final_prompt = f"{prompt_template}\n\nAvailable tags:\n{tags_formatted}"

        logger.debug(f"Final user prompt:\n{final_prompt}")

        return final_prompt

    def _parse_response(
        self, response_text: str, allowed_tags: List[str], strip_markdown: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse model response and validate tags.

        Parses JSON response, validates against Pydantic schema, and filters
        tags to only include those in the allowed list.

        Args:
            response_text: Raw response text from model
            allowed_tags: List of allowed tag names to filter against
            strip_markdown: Whether to strip markdown code blocks (for Gemini)

        Returns:
            Dictionary with "tags" key containing list of validated tag dicts
            Format: {"tags": [{"tag": "tag_name", "confidence": 0.95}, ...]}
        """
        try:
            # Strip markdown code blocks if requested (Gemini sometimes wraps JSON)
            cleaned_text = response_text.strip()
            if strip_markdown:
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

            # Parse and validate JSON
            output = TagsOutput.model_validate_json(cleaned_text)

            # Filter tags to allowed list (case-insensitive, normalize spaces/underscores)
            # Build mapping: normalized_tag -> original_tag for lookup
            # Normalize by: lowercase + replace spaces/underscores with single format
            def normalize_tag(tag: str) -> str:
                return tag.lower().replace(" ", "_").replace("&", "_")

            allowed_tags_map = {normalize_tag(tag): tag for tag in allowed_tags}

            validated_tags = []
            for tag_pred in output.tags:
                # Normalize predicted tag the same way
                tag_normalized = normalize_tag(tag_pred.tag)

                if tag_normalized not in allowed_tags_map:
                    logger.warning(
                        f"Tag '{tag_pred.tag}' (normalized: '{tag_normalized}') not in allowed list, skipping"
                    )
                    continue

                # Use the original canonical tag name from allowed_tags
                canonical_tag_name = allowed_tags_map[tag_normalized]
                validated_tags.append({"tag": canonical_tag_name, "confidence": tag_pred.confidence})

            return {"tags": validated_tags}

        except ValidationError as e:
            logger.error(f"Failed to validate response: {e}")
            logger.error(f"Response text: {response_text}")
            return {"tags": [], "error": f"Validation failed: {str(e)}"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            return {"tags": [], "error": f"JSON parsing failed: {str(e)}"}
