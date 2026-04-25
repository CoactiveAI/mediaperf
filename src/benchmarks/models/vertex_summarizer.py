import time
from typing import Any, Dict, Optional

from google import genai
from google.genai import types
from loguru import logger

from ..constants import (
    DEFAULT_VERTEX_INFERENCE_PARAMS,
    DEFAULT_VERTEX_LOCATION,
    DEFAULT_VERTEX_MODEL_ID,
    SUMMARIZATION_PROMPTS_DIR,
)
from ..exceptions import FatalError
from ..exceptions.gcp_errors import is_fatal_gcp_error
from .base_summarizer import VideoSummarizer


class VertexSummarizer(VideoSummarizer):
    """Video summarizer using Google Vertex AI (Gemini models)."""

    def __init__(
        self,
        model_id: str = DEFAULT_VERTEX_MODEL_ID,
        model_name: str = None,
        project_id: Optional[str] = None,
        location: str = DEFAULT_VERTEX_LOCATION,
        inference_params: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
        system_prompt_file: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
    ):
        super().__init__(model_id, model_name, inference_params or DEFAULT_VERTEX_INFERENCE_PARAMS)
        self.project_id = project_id
        self.location = location
        self.labels = labels or {}

        # Create Vertex AI client
        if project_id:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
            # Use default project from environment/credentials
            self.client = genai.Client(vertexai=True)

        self._load_prompts(
            model_prefix="vertex",
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
        if not video_source.startswith("gs://"):
            raise ValueError(f"VertexSummarizer requires GCS URI, got: {video_source}")

        # Use custom prompts if provided, otherwise use templates
        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        user_prompt = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("VERTEX SUMMARIZER - PROMPTS")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt:\n{user_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"Video Source: {video_source}")
        logger.debug("=" * 80)

        try:
            # Measure API call time
            api_start = time.time()
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    user_prompt,
                    types.Part.from_uri(file_uri=video_source, mime_type="video/mp4"),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    labels=self.labels,
                    **self.inference_params,
                ),
            )
            api_time = time.time() - api_start

            # Check finish reason
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == types.FinishReason.MAX_TOKENS:
                    logger.warning(f"Response truncated (MAX_TOKENS) for {video_source}")
                elif finish_reason != types.FinishReason.STOP:
                    logger.warning(f"Unexpected finish reason {finish_reason} for {video_source}")

            response_text = response.text

            # Handle case where response text is None (can happen with truncation)
            if response_text is None:
                logger.error(f"Response text is None for {video_source}, likely due to MAX_TOKENS truncation")
                return {"summary": "", "error": "Response text is None, likely due to MAX_TOKENS truncation"}

            # Log raw response to see what the model is returning
            logger.debug("=" * 80)
            logger.debug("VERTEX SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Response length: {len(response_text)} characters")
            logger.debug(f"Response text:\n{response_text}")
            logger.debug("=" * 80)

            result = {"summary": response_text}

            # Add usage and timing if requested
            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                # Add token counts if available from API
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage_dict["input_tokens"] = response.usage_metadata.prompt_token_count
                    usage_dict["output_tokens"] = response.usage_metadata.candidates_token_count
                    usage_dict["total_tokens"] = response.usage_metadata.total_token_count

                result["usage"] = usage_dict

            return result

        except Exception as e:
            if is_fatal_gcp_error(e):
                raise FatalError(f"GCP error: {e}") from e
            # Not fatal - propagate as-is
            logger.error(f"Vertex AI API call failed for {video_source}: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name if self.model_name else self.model_id
