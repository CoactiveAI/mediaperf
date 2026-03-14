"""Qwen-specific error handling utilities."""


def is_fatal_qwen_error(e: Exception) -> bool:
    """
    Check if Qwen exception is fatal (should halt pipeline).

    Qwen uses OpenAI-compatible client, so similar error patterns.

    Args:
        e: Exception to check

    Returns:
        True if error is credential or configuration related
    """
    error_str = str(e)

    # Model not found errors
    if "404" in error_str and "not found" in error_str.lower():
        return True
    if "could not find base model" in error_str.lower():
        return True
    if "model" in error_str.lower() and "does not exist" in error_str.lower():
        return True

    # Authentication errors
    if "401" in error_str or "unauthorized" in error_str.lower():
        return True
    if "authentication" in error_str.lower() or "api_key" in error_str.lower():
        return True

    return False
