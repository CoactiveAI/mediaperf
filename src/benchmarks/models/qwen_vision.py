import time
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from ..constants import DEFAULT_QWEN_INFERENCE_PARAMS, TAGGING_PROMPTS_DIR
from ..exceptions import FatalError
from ..exceptions.qwen_errors import is_fatal_qwen_error
from .base import VideoTagger


class QwenVideoTagger(VideoTagger):
    """Video tagger using Qwen vision models via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_id: str,
        model_name: str = None,
        inference_params: Dict = None,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
    ):
        super().__init__(model_id, model_name, inference_params or DEFAULT_QWEN_INFERENCE_PARAMS)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._load_prompts(
            model_prefix="qwen",
            prompts_dir=TAGGING_PROMPTS_DIR,
            system_prompt_file=custom_system_prompt,
            user_prompt_file=custom_user_prompt,
        )

    def tag_video(
        self,
        video_source: str,
        allowed_tags: List[str],
        tag_definitions: Dict[str, str],
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
        return_usage: bool = False,
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Tag video. video_source should be a base64-encoded video string."""
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        user_text = self._build_user_prompt(allowed_tags, tag_definitions, custom_user_prompt)

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
            result = self._parse_response(response_text, allowed_tags)

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
