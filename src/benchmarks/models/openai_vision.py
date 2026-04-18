import time
from typing import Dict, List

from openai import OpenAI

from ..constants import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    IMAGE_DATA_URL_PREFIX,
    TAGGING_PROMPTS_DIR,
)
from ..exceptions import FatalError
from ..exceptions.openai_errors import is_fatal_openai_error
from ..schemas import TagsOutput
from .base import VideoTagger


class OpenAIVideoTagger(VideoTagger):
    """Video tagger using OpenAI Vision models (frame-based)."""

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_OPENAI_MODEL,
        model_name: str = None,
        reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
        custom_system_prompt: str = None,
        custom_user_prompt: str = None,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self._load_prompts(
            model_prefix="openai",
            prompts_dir=TAGGING_PROMPTS_DIR,
            system_prompt_file=custom_system_prompt,
            user_prompt_file=custom_user_prompt,
        )

    def tag_video(
        self,
        video_source: List[str],
        allowed_tags: List[str],
        tag_definitions: Dict[str, str],
        custom_system_prompt: str = None,
        custom_user_prompt: str = None,
        return_usage: bool = False,
        **kwargs,
    ) -> Dict:
        """Tag video frames. video_source should be a list of base64-encoded frames."""
        # Use custom system prompt if provided, otherwise use template
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template

        # Build user prompt with tag definitions
        user_text = self._build_user_prompt(allowed_tags, tag_definitions, custom_user_prompt)

        user_content = [{"type": "input_text", "text": user_text}]

        for b64_frame in video_source:
            data_url = f"data:image/jpeg;base64,{b64_frame}"
            user_content.append({"type": "input_image", "image_url": data_url})

        try:
            # Measure API call time
            api_start = time.time()
            resp = self.client.responses.parse(
                model=self.model_id,
                reasoning={"effort": self.reasoning_effort},
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                text_format=TagsOutput,
            )
            api_time = time.time() - api_start

            # Parse and filter response to only allowed tags
            result = self._parse_response(resp.output_text, allowed_tags, strip_markdown=False)

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
            raise

    def tag_frame(
        self,
        frame_b64: str,
        allowed_tags: List[str],
        tag_definitions: Dict[str, str],
    ) -> Dict:
        """Tag a single frame. frame_b64 should be a base64-encoded JPEG."""
        defs_str = "\n".join([f"- {t}: {tag_definitions.get(t, '')}" for t in allowed_tags])
        data_url = f"{IMAGE_DATA_URL_PREFIX}{frame_b64}"

        resp = self.client.responses.parse(
            model=self.model_id,
            reasoning={"effort": self.reasoning_effort},
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Allowed tags: {allowed_tags}\n\nTag definitions:\n{defs_str}",
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            text_format=TagsOutput,
        )

        # Parse and filter response to only allowed tags
        return self._parse_response(resp.output_text, allowed_tags, strip_markdown=False)

    def get_model_name(self) -> str:
        return self.model_name
