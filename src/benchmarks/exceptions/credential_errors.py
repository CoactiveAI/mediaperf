from typing import NoReturn

from botocore.exceptions import ClientError, NoCredentialsError
from google.api_core.exceptions import Forbidden, Unauthenticated
from google.auth.exceptions import DefaultCredentialsError
from loguru import logger


def handle_credential_error(e: Exception, context: str = "") -> NoReturn:
    """
    Log credential error with context and re-raise.

    Args:
        e: The credential-related exception
        context: Optional context describing where the error occurred

    Raises:
        The same exception that was passed in
    """
    context_msg = f" ({context})" if context else ""

    if isinstance(e, NoCredentialsError):
        logger.error(f"AWS credentials not found{context_msg}: {e}")
    elif isinstance(e, DefaultCredentialsError):
        logger.error(f"GCP credentials not found or invalid{context_msg}: {e}")
    elif isinstance(e, Unauthenticated):
        logger.error(f"GCP credentials invalid/unauthenticated{context_msg}: {e}")
    elif isinstance(e, Forbidden):
        logger.error(f"GCP credentials invalid/expired{context_msg}: {e}")
    elif isinstance(e, ClientError):
        logger.error(f"AWS credentials invalid/expired{context_msg}: {e}")
    else:
        logger.error(f"Credential error{context_msg}: {e}")

    logger.error("Please update your credentials and try again.")
    raise
