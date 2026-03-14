import json
import time
from typing import Any, Dict, List, Optional

import boto3
from loguru import logger

from ..constants import (
    BEDROCK_MAX_TOKENS,
    BEDROCK_TEMPERATURE,
    BEDROCK_TOP_K,
    BEDROCK_TOP_P,
    DEFAULT_AWS_REGION,
    DEFAULT_BEDROCK_MODEL_ID,
    SUMMARIZATION_PROMPTS_DIR,
)
from ..exceptions import FatalError
from ..exceptions.aws_errors import ClientError, NoCredentialsError, is_fatal_aws_error
from .base_summarizer import VideoSummarizer


class BedrockVideoSummarizer(VideoSummarizer):
    """Video summarizer using AWS Bedrock (Amazon Nova models and NVIDIA models)."""

    def __init__(
        self,
        model_id: str = DEFAULT_BEDROCK_MODEL_ID,
        model_name: str = None,
        region_name: str = DEFAULT_AWS_REGION,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        inference_params: Optional[Dict[str, Any]] = None,
        api_type: str = "converse",
        account_id: Optional[str] = None,
        input_type: str = "s3_uri",
        system_prompt_file: Optional[str] = None,
        user_prompt_file: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.api_type = api_type
        self.account_id = account_id
        self.input_type = input_type
        self.inference_params = inference_params or {
            "maxTokens": BEDROCK_MAX_TOKENS,
            "temperature": BEDROCK_TEMPERATURE,
            "topP": BEDROCK_TOP_P,
            "topK": BEDROCK_TOP_K,
        }

        # Create Bedrock client
        if aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )
        else:
            self.client = boto3.client("bedrock-runtime", region_name=region_name)

        # Auto-select system prompt based on input_type if not provided
        if not system_prompt_file:
            system_prompt_file = "bedrock_nvidia_system.txt" if input_type == "frames_bytes" else None

        self._load_prompts(
            model_prefix="bedrock",
            prompts_dir=SUMMARIZATION_PROMPTS_DIR,
            system_prompt_file=system_prompt_file,
            user_prompt_file=user_prompt_file,
        )

    def summarize_video(
        self,
        video_source,
        custom_system_prompt: Optional[str] = None,
        custom_user_prompt: Optional[str] = None,
        return_usage: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        # Route based on input_type
        if self.input_type == "frames_bytes":
            if not isinstance(video_source, list):
                raise ValueError(
                    f"BedrockVideoSummarizer with input_type='frames_bytes' "
                    f"requires List[bytes], got: {type(video_source)}"
                )

            if self.api_type == "converse":
                return self._summarize_with_converse_frames(
                    video_source,
                    custom_system_prompt,
                    custom_user_prompt,
                    return_usage,
                )
            elif self.api_type == "invoke":
                return self._summarize_with_invoke_frames(
                    video_source,
                    custom_system_prompt,
                    custom_user_prompt,
                    return_usage,
                )
            else:
                raise ValueError(f"Unknown api_type: {self.api_type}. Must be 'converse' or 'invoke'")

        else:  # input_type == "s3_uri"
            if not isinstance(video_source, str) or not video_source.startswith("s3://"):
                raise ValueError(
                    f"BedrockVideoSummarizer with input_type='s3_uri' requires S3 URI, got: {video_source}"
                )

            if self.api_type == "converse":
                return self._summarize_with_converse(
                    video_source,
                    custom_system_prompt,
                    custom_user_prompt,
                    return_usage,
                )
            elif self.api_type == "invoke":
                return self._summarize_with_invoke(
                    video_source,
                    custom_system_prompt,
                    custom_user_prompt,
                    return_usage,
                )
            else:
                raise ValueError(f"Unknown api_type: {self.api_type}. Must be 'converse' or 'invoke'")

    def _summarize_with_converse(
        self,
        video_source: str,
        custom_system_prompt: Optional[str],
        custom_user_prompt: Optional[str],
        return_usage: bool,
    ) -> Dict[str, Any]:
        user_prompt = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "mp4",
                            "source": {
                                "s3Location": {
                                    "uri": video_source,
                                }
                            },
                        }
                    },
                    {"text": user_prompt},
                ],
            }
        ]

        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        system = [{"text": system_prompt}]

        inference_config = {
            "maxTokens": self.inference_params.get("maxTokens", 1000),
            "temperature": self.inference_params.get("temperature", 0.1),
            "topP": self.inference_params.get("topP", 0.9),
        }

        additional_model_fields = self.inference_params.get(
            "additional_model_fields",
            {"inferenceConfig": {"topK": self.inference_params.get("topK", 20)}},
        )

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("BEDROCK SUMMARIZER - PROMPTS (S3 URI)")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt:\n{user_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"Video Source: {video_source}")
        logger.debug("=" * 80)

        try:
            api_start = time.time()
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields,
            )
            api_time = time.time() - api_start

            logger.debug("=" * 80)
            logger.debug("BEDROCK SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Full Bedrock API response: {response}")
            logger.debug("-" * 80)

            response_text = response["output"]["message"]["content"][0]["text"]

            logger.debug("Raw response text:")
            logger.debug(f"Response length: {len(response_text)} characters")
            logger.debug(f"Response text:\n{response_text}")
            logger.debug("=" * 80)

            # For summarization, we just extract the text without schema validation
            result = self._parse_summary_response(response_text, validate_schema=False)

            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                if "usage" in response:
                    usage_dict["input_tokens"] = response["usage"]["inputTokens"]
                    usage_dict["output_tokens"] = response["usage"]["outputTokens"]
                    usage_dict["total_tokens"] = response["usage"]["totalTokens"]

                result["usage"] = usage_dict

            return result

        except (NoCredentialsError, ClientError) as e:
            if is_fatal_aws_error(e):
                raise FatalError(f"AWS error: {e}") from e
            # Not fatal - propagate as-is
            logger.error(f"Bedrock API call failed for {video_source}: {e}")
            # Check for ModelErrorException - often indicates codec mismatch
            if isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") == "ModelErrorException":
                logger.warning(
                    "Note: ModelErrorException may indicate codec mismatch (e.g., AV1 with H.264-only model)"
                )
            raise

    def _summarize_with_converse_frames(
        self,
        frames_bytes: List[bytes],
        custom_system_prompt: Optional[str],
        custom_user_prompt: Optional[str],
        return_usage: bool,
    ) -> Dict[str, Any]:
        user_prompt = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        content_blocks = []
        for frame_bytes in frames_bytes:
            content_blocks.append(
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": frame_bytes},
                    }
                }
            )
        content_blocks.append({"text": user_prompt})

        messages = [{"role": "user", "content": content_blocks}]

        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template
        system = [{"text": system_prompt}]

        inference_config = {
            "maxTokens": self.inference_params.get("maxTokens", 1000),
            "temperature": self.inference_params.get("temperature", 0.1),
            "topP": self.inference_params.get("topP", 0.9),
        }

        additional_model_fields = self.inference_params.get(
            "additional_model_fields",
            {"inferenceConfig": {"topK": self.inference_params.get("topK", 20)}},
        )

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("BEDROCK SUMMARIZER - PROMPTS (FRAMES)")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt:\n{user_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"Number of frames: {len(frames_bytes)}")
        logger.debug("=" * 80)

        try:
            api_start = time.time()
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_model_fields,
            )
            api_time = time.time() - api_start

            logger.debug("=" * 80)
            logger.debug("BEDROCK SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Full Bedrock API response: {response}")
            logger.debug("-" * 80)

            response_text = response["output"]["message"]["content"][0]["text"]

            logger.debug("Raw response text:")
            logger.debug(f"Response length: {len(response_text)} characters")
            logger.debug(f"Response text:\n{response_text}")
            logger.debug("=" * 80)

            result = self._parse_summary_response(response_text, validate_schema=False)

            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                if "usage" in response:
                    usage_dict["input_tokens"] = response["usage"]["inputTokens"]
                    usage_dict["output_tokens"] = response["usage"]["outputTokens"]
                    usage_dict["total_tokens"] = response["usage"]["totalTokens"]

                result["usage"] = usage_dict

            return result

        except (NoCredentialsError, ClientError) as e:
            if is_fatal_aws_error(e):
                raise FatalError(f"AWS error: {e}") from e
            # Not fatal - propagate as-is
            logger.error(f"Bedrock API call failed for frame-based inference: {e}")
            # Check for ModelErrorException - often indicates codec mismatch
            if isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") == "ModelErrorException":
                logger.warning(
                    "Note: ModelErrorException may indicate codec mismatch (e.g., AV1 with H.264-only model)"
                )
            raise

    def _summarize_with_invoke(
        self,
        video_source: str,
        custom_system_prompt: Optional[str],
        custom_user_prompt: Optional[str],
        return_usage: bool,
    ) -> Dict[str, Any]:
        if not self.account_id:
            raise ValueError("account_id is required when using api_type='invoke'")

        user_prompt = custom_user_prompt if custom_user_prompt else self.user_prompt_template

        system_prompt = custom_system_prompt if custom_system_prompt else self.system_prompt_template

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Debug log: show the prompts being sent to the model
        logger.debug("=" * 80)
        logger.debug("BEDROCK SUMMARIZER - PROMPTS (S3 URI - invoke_model)")
        logger.debug("=" * 80)
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"User Prompt:\n{user_prompt}")
        logger.debug("-" * 80)
        logger.debug(f"Video Source: {video_source}")
        logger.debug("=" * 80)

        request_body = {
            "inputPrompt": full_prompt,
            "mediaSource": {
                "s3Location": {
                    "uri": video_source,
                    "bucketOwner": self.account_id,
                }
            },
            "temperature": self.inference_params.get("temperature", 0.1),
            "maxOutputTokens": self.inference_params.get("maxTokens", 1000),
        }

        try:
            api_start = time.time()
            response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(request_body))
            api_time = time.time() - api_start

            logger.debug("=" * 80)
            logger.debug("BEDROCK SUMMARIZER - RESPONSE")
            logger.debug("=" * 80)
            logger.debug(f"Full Bedrock API response: {response}")
            logger.debug("-" * 80)

            model_response = json.loads(response["body"].read())
            response_text = model_response["message"]

            logger.debug("Raw response text:")
            logger.debug(f"Response length: {len(response_text)} characters")
            logger.debug(f"Response text:\n{response_text}")
            logger.debug("=" * 80)

            result = self._parse_summary_response(response_text, validate_schema=False)

            if return_usage:
                usage_dict = {"api_call_time_seconds": api_time}

                headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
                output_tokens = headers.get("x-amzn-bedrock-output-token-count")

                if output_tokens:
                    output_tokens_int = int(output_tokens)
                    usage_dict["input_tokens"] = 0
                    usage_dict["output_tokens"] = output_tokens_int
                    usage_dict["total_tokens"] = output_tokens_int

                result["usage"] = usage_dict

            return result

        except (NoCredentialsError, ClientError) as e:
            if is_fatal_aws_error(e):
                raise FatalError(f"AWS error: {e}") from e
            # Not fatal - propagate as-is
            logger.error(f"Bedrock invoke_model API call failed for {video_source}: {e}")
            # Check for ModelErrorException - often indicates codec mismatch
            if isinstance(e, ClientError) and e.response.get("Error", {}).get("Code") == "ModelErrorException":
                logger.warning(
                    "Note: ModelErrorException may indicate codec mismatch (e.g., AV1 with H.264-only model)"
                )
            raise

    def _summarize_with_invoke_frames(
        self,
        frames_bytes: List[bytes],
        custom_system_prompt: Optional[str],
        custom_user_prompt: Optional[str],
        return_usage: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "invoke_model API does not support frame-based inference. Use api_type='converse' instead."
        )

    def get_model_name(self) -> str:
        return self.model_name
