"""AWS-specific error handling utilities."""

from botocore.exceptions import ClientError, NoCredentialsError

# Credential-related error codes
AWS_CREDENTIAL_ERROR_CODES = {
    "ExpiredToken",
    "ExpiredTokenException",
    "InvalidAccessKeyId",
    "InvalidToken",
    "SignatureDoesNotMatch",
    "UnrecognizedClientException",
    "InvalidSignatureException",
}

# Configuration-related error codes (always fatal)
AWS_CONFIGURATION_ERROR_CODES = {
    "ResourceNotFoundException",
    "ModelNotFoundException",
    "AccessDeniedException",
}


def is_fatal_aws_error(e: Exception) -> bool:
    """
    Check if AWS exception is fatal (should halt pipeline).

    Args:
        e: Exception to check

    Returns:
        True if error is credential or configuration related
    """
    if isinstance(e, NoCredentialsError):
        return True

    if isinstance(e, ClientError):
        error_code = e.response.get("Error", {}).get("Code", "")

        # Always fatal errors
        if error_code in AWS_CREDENTIAL_ERROR_CODES or error_code in AWS_CONFIGURATION_ERROR_CODES:
            return True

        # ValidationException - check message to determine if fatal
        if error_code == "ValidationException":
            error_message = str(e).lower()

            # Fatal - configuration issues
            if "model identifier is invalid" in error_message:
                return True
            if "model" in error_message and "not found" in error_message:
                return True

            # Non-fatal - per-video issues (payload size, video format, etc.)
            if "length limit exceeded" in error_message:
                return False
            if "payload" in error_message:
                return False
            if "request body" in error_message:
                return False

            # Unknown ValidationException - default to non-fatal to be safe
            return False

    return False
