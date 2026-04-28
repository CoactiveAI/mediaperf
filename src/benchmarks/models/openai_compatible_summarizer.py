import time
from typing import Any, Dict, Optional

from loguru import logger
from openai import OpenAI

from ..constants import DEFAULT_QWEN_INFERENCE_PARAMS, SUMMARIZATION_PROMPTS_DIR
from ..exceptions import FatalError
from ..exceptions.qwen_errors import is_fatal_qwen_error
from .base_summarizer import VideoSummarizer


class QwenSummarizer(VideoSummarizer):
    """Video summarizer using Qwen vision models via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: str,
        model_name: str = None,
        inference_params: Dict = None,
        system_prompt_file: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
    ):
        super().__init__(model_id, model_name, inference_params or DEFAULT_QWEN_INFERENCE_PARAMS)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._load_prompts(
            model_prefix="qwen",
            prompts_dir=SUMMARIZATION_PROMPTS_DIR,
            system_prompt_file=system_prompt_file,
            user_prompt_file=user_prompt_file,
        )

    def summarize_video(
        self,
        video_source: str,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
        return_usage: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Summarize video. video_source should be a base64-encoded video string."""
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        user_text = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("QWEN SUMMARIZER - PROMPTS")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt:\n{user_text}")
        logger.debug("-" * 80)
        logger.debug(f"Video source length: {len(video_source)} characters (base64)")
        logger.debug("=" * 80)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_source}"},
                    },
                ],
            },
        ]

        try:
            api_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **self.inference_params,
            )
            api_time = time.time() - api_start

            response_text = response.choices[0].message.content

            # Debug log: show the response from the model
            logger.debug("=" * 80)
            logger.debug("QWEN SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Response length: {len(response_text)} characters")
            logger.debug(f"Response text:\n{response_text}")
            logger.debug("=" * 80)

            result = {"summary": response_text}

            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                if hasattr(response, "usage") and response.usage:
                    usage_dict["input_tokens"] = response.usage.prompt_tokens
                    usage_dict["output_tokens"] = response.usage.completion_tokens
                    usage_dict["total_tokens"] = response.usage.total_tokens

                result["usage"] = usage_dict

            return result

        except Exception as e:
            if is_fatal_qwen_error(e):
                raise FatalError(f"Qwen error: {e}") from e
            # Not fatal - propagate as-is
            logger.error(f"Qwen API call failed: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name
