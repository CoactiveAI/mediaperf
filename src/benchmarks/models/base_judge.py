import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import ValidationError

from ..schemas import SummaryEvaluationOutput


class LLMJudge(ABC):
    """Abstract base class for LLM judge models."""

    @abstractmethod
    def evaluate_summary(
        self,
        ground_truth_summary: str,
        generated_summary: str,
        criteria: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a generated summary against ground truth.

        Args:
            ground_truth_summary: Reference/ground truth summary
            generated_summary: Model-generated summary to evaluate
            criteria: List of evaluation criteria dicts with 'name' and 'description'
            return_usage: Whether to return token usage and timing data

        Returns:
            Dict with evaluation scores, reasoning, and optional usage data
        """
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
            model_prefix: Prefix for default prompt files (e.g., "openai_judge")
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

    def _format_criteria(self, criteria: List[Dict[str, str]]) -> str:
        """
        Format evaluation criteria for prompt injection.

        Args:
            criteria: List of criteria dicts with 'name' and 'description'

        Returns:
            Formatted string with numbered criteria
        """
        formatted = []
        for i, criterion in enumerate(criteria, start=1):
            name = criterion.get("name", "")
            description = criterion.get("description", "")
            formatted.append(f"{i}. **{name}**: {description}")

        return "\n".join(formatted)

    def _parse_evaluation_response(self, response_text: str, strip_markdown: bool = True) -> Dict[str, Any]:
        """
        Parse LLM judge response and extract evaluation scores.

        Args:
            response_text: Raw response text from LLM
            strip_markdown: Whether to strip markdown code blocks

        Returns:
            Dict with evaluation data (criterion_scores, overall_reasoning)
            Note: overall_score aggregation is done in the pipeline based on config
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

            # Parse and validate JSON against schema
            output = SummaryEvaluationOutput.model_validate_json(cleaned_text)

            # Extract criterion scores (no aggregation here - that's for metrics module)
            criterion_scores = [
                {"criterion": cs.criterion, "score": cs.score, "reasoning": cs.reasoning}
                for cs in output.criterion_scores
            ]

            return {"criterion_scores": criterion_scores, "overall_reasoning": output.overall_reasoning}

        except ValidationError as e:
            logger.error(f"Failed to validate evaluation response: {e}")
            logger.error(f"Response text: {response_text}")
            return {
                "criterion_scores": [],
                "overall_score": 0.0,
                "overall_reasoning": "Error: Failed to parse evaluation",
                "error": f"Validation failed: {str(e)}",
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            return {
                "criterion_scores": [],
                "overall_score": 0.0,
                "overall_reasoning": "Error: Failed to parse JSON",
                "error": f"JSON parsing failed: {str(e)}",
            }
