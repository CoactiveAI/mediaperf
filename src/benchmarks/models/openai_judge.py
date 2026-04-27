import json
import time
from typing import Any, Dict, List

from loguru import logger
from openai import OpenAI

from ..constants import (
    DEFAULT_OPENAI_INFERENCE_PARAMS,
    DEFAULT_OPENAI_MODEL,
    EVALUATION_PROMPTS_DIR,
)
from ..exceptions import FatalError
from ..exceptions.openai_errors import is_fatal_openai_error
from ..schemas import SummaryEvaluationOutput
from .base_judge import LLMJudge


class OpenAILLMJudge(LLMJudge):
    """LLM judge using OpenAI models with structured output."""

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_OPENAI_MODEL,
        model_name: str = None,
        inference_params: dict = None,
    ):
        super().__init__(model_id, model_name, inference_params or DEFAULT_OPENAI_INFERENCE_PARAMS)
        self.client = OpenAI(api_key=api_key)
        self._load_prompts(model_prefix="openai_judge", prompts_dir=EVALUATION_PROMPTS_DIR)

    def evaluate_summary(
        self,
        ground_truth_summary: str,
        generated_summary: str,
        criteria: List[Dict[str, str]],
        return_usage: bool = False,
        custom_system_prompt: str = None,
        custom_user_prompt: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a generated summary against ground truth using OpenAI.

        Args:
            ground_truth_summary: Reference/ground truth summary
            generated_summary: Model-generated summary to evaluate
            criteria: List of evaluation criteria dicts with 'name' and 'description'
            return_usage: Whether to return token usage and timing data
            custom_system_prompt: Optional custom system prompt
            custom_user_prompt: Optional custom user prompt template

        Returns:
            Dict with evaluation scores, reasoning, and optional usage data
        """
        # Use custom system prompt if provided, otherwise use template
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template

        # Format criteria for prompt injection
        criteria_formatted = self._format_criteria(criteria)

        # Build user prompt with ground truth and generated summaries
        user_prompt_template = custom_user_prompt if custom_user_prompt else self.user_prompt_template
        user_prompt = user_prompt_template.format(
            criteria=criteria_formatted, ground_truth=ground_truth_summary, generated_summary=generated_summary
        )

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("OPENAI JUDGE - PROMPTS")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt (with {len(criteria)} criteria):\n{user_prompt}")
        logger.debug("=" * 80)

        # Prepare message content (text-only, no images)
        try:
            # Measure API call time
            api_start = time.time()
            resp = self.client.responses.parse(
                model=self.model_id,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
                text_format=SummaryEvaluationOutput,
                **self.inference_params,
            )
            api_time = time.time() - api_start

            # Debug log: show the response from the model
            logger.debug("=" * 80)
            logger.debug("OPENAI JUDGE - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Response length: {len(resp.output_text)} characters")
            logger.debug(f"Response text:\n{resp.output_text}")
            logger.debug("=" * 80)

            # Parse structured output
            result = json.loads(resp.output_text)

            # Add usage and timing if requested
            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                # Add token counts if available from API
                if hasattr(resp, "usage") and resp.usage:
                    usage_dict["input_tokens"] = resp.usage.input_tokens
                    usage_dict["output_tokens"] = resp.usage.output_tokens
                    usage_dict["total_tokens"] = resp.usage.input_tokens + resp.usage.output_tokens

                result["usage"] = usage_dict

            return result

        except Exception as e:
            if is_fatal_openai_error(e):
                raise FatalError(f"OpenAI error: {e}") from e
            # Not fatal - propagate as-is
            logger.error(f"OpenAI judge API call failed: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name if self.model_name else self.model_id
