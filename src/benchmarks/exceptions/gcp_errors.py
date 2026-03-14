"""GCP-specific error handling utilities."""

from google.api_core.exceptions import Forbidden, NotFound, PermissionDenied, Unauthenticated
from google.auth.exceptions import DefaultCredentialsError

# Credential-related exceptions
GCP_CREDENTIAL_EXCEPTIONS = (DefaultCredentialsError, Forbidden, Unauthenticated)

# Configuration-related exceptions
GCP_CONFIGURATION_EXCEPTIONS = (NotFound, PermissionDenied)


def is_fatal_gcp_error(e: Exception) -> bool:
    """
    Check if GCP exception is fatal (should halt pipeline).

    Args:
        e: Exception to check

    Returns:
        True if error is credential or configuration related
    """
    # Check exception types
    if isinstance(e, GCP_CREDENTIAL_EXCEPTIONS + GCP_CONFIGURATION_EXCEPTIONS):
        return True

    # Check for google-genai library errors (string matching needed)
    error_str = str(e)
    if "NOT_FOUND" in error_str and ("model" in error_str.lower() or "publisher" in error_str.lower()):
        return True

    return False
