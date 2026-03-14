import time
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from ..constants import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    SUMMARIZATION_PROMPTS_DIR,
)
from ..exceptions import FatalError
from ..exceptions.openai_errors import is_fatal_openai_error
from .base_summarizer import VideoSummarizer


class OpenAISummarizer(VideoSummarizer):
    """Video summarizer using OpenAI Vision models (frame-based)."""

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_OPENAI_MODEL,
        model_name: str = None,
        reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
        system_prompt_file: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self._load_prompts(
            model_prefix="openai",
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
        **kwargs,
    ) -> Dict[str, Any]:
        """Summarize video frames. video_source should be a list of base64-encoded frames."""
        # Use custom system prompt if provided, otherwise use template
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        user_text = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("OPENAI SUMMARIZER - PROMPTS")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt (with {len(video_source)} frames):\n{user_text}")
        logger.debug("=" * 80)

        user_content = [{"type": "text", "text": user_text}]

        for b64_frame in video_source:
            data_url = f"data:image/jpeg;base64,{b64_frame}"
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})

        try:
            # Measure API call time
            api_start = time.time()
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
            )
            api_time = time.time() - api_start

            # Extract summary from response
            summary_text = resp.choices[0].message.content

            # Debug log: show the response from the model
            logger.debug("=" * 80)
            logger.debug("OPENAI SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Response length: {len(summary_text)} characters")
            logger.debug(f"Response text:\n{summary_text}")
            logger.debug("=" * 80)

            result = {"summary": summary_text}

            # Add usage and timing if requested
            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                # Add token counts if available from API
                if hasattr(resp, "usage") and resp.usage:
                    usage_dict["input_tokens"] = resp.usage.prompt_tokens
                    usage_dict["output_tokens"] = resp.usage.completion_tokens
                    usage_dict["total_tokens"] = resp.usage.total_tokens

                result["usage"] = usage_dict

            return result

        except Exception as e:
            if is_fatal_openai_error(e):
                raise FatalError(f"OpenAI error: {e}") from e
            # Not fatal - propagate as-is
            raise

    def get_model_name(self) -> str:
        return self.model_name if self.model_name else self.model_id
