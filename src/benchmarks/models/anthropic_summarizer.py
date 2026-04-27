import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from loguru import logger

from ..constants import DEFAULT_ANTHROPIC_INFERENCE_PARAMS, DEFAULT_ANTHROPIC_MODEL, SUMMARIZATION_PROMPTS_DIR
from ..exceptions import FatalError
from ..exceptions.anthropic_errors import is_fatal_anthropic_error
from .base_summarizer import VideoSummarizer


class AnthropicSummarizer(VideoSummarizer):
    """Video summarizer using Anthropic Claude models (frame-based)."""

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_ANTHROPIC_MODEL,
        model_name: str = None,
        inference_params: Optional[Dict[str, Any]] = None,
        system_prompt_file: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
    ):
        super().__init__(model_id, model_name, inference_params or DEFAULT_ANTHROPIC_INFERENCE_PARAMS)
        self.client = Anthropic(api_key=api_key)
        self._load_prompts(
            model_prefix="anthropic",
            prompts_dir=SUMMARIZATION_PROMPTS_DIR,
            system_prompt_file=system_prompt_file,
            user_prompt_file=user_prompt_file,
        )

    def summarize_video(
        self,
        video_source: List[str],
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
        return_usage: bool = False,
    ) -> Dict[str, Any]:
        """Summarize video frames. video_source should be a list of base64-encoded frames."""
        # Use custom system prompt if provided, otherwise use template
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        user_text = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("ANTHROPIC SUMMARIZER - PROMPTS")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt (with {len(video_source)} frames):\n{user_text}")
        logger.debug("=" * 80)

        # Build content list with all frames
        content = []

        # Add all frames as image blocks
        for b64_frame in video_source:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_frame,
                    },
                }
            )

        # Add text prompt at the end
        content.append(
            {
                "type": "text",
                "text": user_text,
            }
        )

        try:
            # Measure API call time
            api_start = time.time()
            response = self.client.messages.create(
                model=self.model_id,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                **self.inference_params,
            )
            api_time = time.time() - api_start

            # Extract summary from response
            # Handle both TextBlock and ThinkingBlock (extended thinking models)
            summary_text = None
            for block in response.content:
                if hasattr(block, "text"):
                    summary_text = block.text
                    break

            if summary_text is None:
                raise ValueError("No text content found in response")

            # Debug log: show the response from the model
            logger.debug("=" * 80)
            logger.debug("ANTHROPIC SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Response length: {len(summary_text)} characters")
            logger.debug(f"Response text:\n{summary_text}")
            logger.debug("=" * 80)

            result = {"summary": summary_text}

            # Add usage and timing if requested
            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                # Add token counts from response
                if hasattr(response, "usage") and response.usage:
                    usage_dict["input_tokens"] = response.usage.input_tokens
                    usage_dict["output_tokens"] = response.usage.output_tokens
                    usage_dict["total_tokens"] = response.usage.input_tokens + response.usage.output_tokens

                result["usage"] = usage_dict

            return result

        except Exception as e:
            if is_fatal_anthropic_error(e):
                raise FatalError(f"Anthropic error: {e}") from e
            # Not fatal - propagate as-is
            raise

    def get_model_name(self) -> str:
        return self.model_name if self.model_name else self.model_id
