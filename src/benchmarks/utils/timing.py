from typing import Dict, Optional


def merge_timing_data(
    video_output: Dict, preprocessing_time: Optional[float], total_inference_time: float
) -> Dict[str, float]:
    """
    Merge timing data from various sources into a single dict.

    Args:
        video_output: Tagger output containing 'usage' dict
        preprocessing_time: Time for preprocessing (None if no preprocessing)
        total_inference_time: Total time measured by pipeline

    Returns:
        Dict with timing fields: api_call_time_seconds, preprocessing_time_seconds,
        total_inference_time_seconds, overhead_time_seconds
    """
    timing_dict = {}

    if "usage" in video_output:
        usage = video_output["usage"]

        # Add API call time
        if "api_call_time_seconds" in usage:
            timing_dict["api_call_time_seconds"] = usage["api_call_time_seconds"]

        # Add preprocessing time if available
        if preprocessing_time is not None:
            timing_dict["preprocessing_time_seconds"] = preprocessing_time

        # Add total inference time
        timing_dict["total_inference_time_seconds"] = total_inference_time

        # Calculate overhead
        component_time = usage.get("api_call_time_seconds", 0)
        if preprocessing_time is not None:
            component_time += preprocessing_time
        timing_dict["overhead_time_seconds"] = total_inference_time - component_time

    return timing_dict


def extract_token_counts(video_output: Dict) -> Dict[str, int]:
    """
    Extract token counts from tagger output if available.

    Args:
        video_output: Tagger output containing 'usage' dict

    Returns:
        Dict with token count fields: prompt_tokens, completion_tokens, total_tokens
    """
    token_dict = {}

    if "usage" in video_output:
        usage = video_output["usage"]
        if "input_tokens" in usage:
            token_dict["prompt_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            token_dict["completion_tokens"] = usage["output_tokens"]
        if "total_tokens" in usage:
            token_dict["total_tokens"] = usage["total_tokens"]

    return token_dict
