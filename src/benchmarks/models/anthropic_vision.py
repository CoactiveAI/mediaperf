import time
from typing import Dict, List

from anthropic import Anthropic
from loguru import logger

from ..constants import DEFAULT_ANTHROPIC_MODEL, TAGGING_PROMPTS_DIR
from ..exceptions import FatalError
from ..exceptions.anthropic_errors import is_fatal_anthropic_error
from ..schemas import TagsOutput
from .base import VideoTagger


class AnthropicVideoTagger(VideoTagger):
    """Video tagger using Anthropic Claude models (frame-based)."""

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_ANTHROPIC_MODEL,
        model_name: str = None,
        custom_system_prompt: str = None,
        custom_user_prompt: str = None,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model_id = model_id
        self.model_name = model_name
        self._load_prompts(
            model_prefix="anthropic",
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
    ) -> Dict:
        """Tag video frames. video_source should be a list of base64-encoded frames."""
        # Use custom system prompt if provided, otherwise use template
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template

        # Build user prompt with tag definitions
        user_text = self._build_user_prompt(allowed_tags, tag_definitions, custom_user_prompt)

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
            response = self.client.messages.parse(
                model=self.model_id,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                output_format=TagsOutput,
            )
            api_time = time.time() - api_start

            # Convert Pydantic object to JSON string for _parse_response
            response_json = response.parsed_output.model_dump_json()

            # Log raw response to see what the model is returning
            logger.debug(f"Raw response:")
            logger.debug(f"Response length: {len(response_json)} characters")
            logger.debug(f"Response text:\n{response_json}")

            # Parse and filter response to only allowed tags
            result = self._parse_response(response_json, allowed_tags, strip_markdown=False)

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
        return self.model_name
