"""Anthropic-specific error handling utilities."""

from anthropic import AuthenticationError, NotFoundError, PermissionDeniedError


def is_fatal_anthropic_error(e: Exception) -> bool:
    """
    Check if Anthropic exception is fatal (should halt pipeline).

    Args:
        e: Exception to check

    Returns:
        True if error is credential or configuration related
    """
    # Credential errors
    if isinstance(e, AuthenticationError):
        return True

    # Configuration errors (model not found, no permission)
    if isinstance(e, (NotFoundError, PermissionDeniedError)):
        return True

    # Fallback string matching for edge cases
    error_str = str(e)
    if "model_not_found" in error_str or "does not exist" in error_str:
        return True
    if "authentication" in error_str.lower() or "api_key" in error_str.lower():
        return True

    return False
