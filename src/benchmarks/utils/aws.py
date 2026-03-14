"""AWS S3 utilities for cloud storage operations."""

from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from ..exceptions.aws_errors import AWS_CREDENTIAL_ERROR_CODES
from ..exceptions.credential_errors import handle_credential_error


def get_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: str = "us-east-1",
):
    """
    Create and return an S3 client.

    Args:
        aws_access_key_id: AWS access key ID. If None, uses environment variables or AWS config.
        aws_secret_access_key: AWS secret access key. If None, uses environment variables or AWS config.
        aws_session_token: AWS session token for temporary credentials. If None, uses environment variables.
        region_name: AWS region name (default: us-east-1).

    Returns:
        boto3 S3 client
    """
    if aws_access_key_id and aws_secret_access_key:
        client_kwargs = {
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "region_name": region_name,
        }
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token
        return boto3.client("s3", **client_kwargs)
    else:
        # Use default credentials (environment variables, ~/.aws/credentials, or IAM role)
        return boto3.client("s3", region_name=region_name)


def upload_file_to_s3(
    local_path: str,
    bucket_name: str,
    s3_key: str,
    s3_client=None,
    extra_args: Optional[dict] = None,
    verbose: bool = True,
) -> bool:
    """
    Upload a file to S3.

    Args:
        local_path: Path to local file to upload.
        bucket_name: Name of S3 bucket.
        s3_key: S3 object key (path within bucket).
        s3_client: Optional boto3 S3 client. If None, creates a new one.
        extra_args: Optional dictionary of extra arguments to pass to upload_file()
                   (e.g., {'ContentType': 'video/mp4', 'Metadata': {...}})
        verbose: If True, log success/error messages. Default True.

    Returns:
        True if upload successful, False otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    try:
        s3_client.upload_file(local_path, bucket_name, s3_key, ExtraArgs=extra_args)
        if verbose:
            logger.success(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
        return True
    except Exception as e:
        if verbose:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
        return False


def download_file_from_s3(bucket_name: str, s3_key: str, local_path: str, s3_client=None) -> bool:
    """
    Download a file from S3.

    Args:
        bucket_name: Name of S3 bucket.
        s3_key: S3 object key (path within bucket).
        local_path: Local path where file should be saved.
        s3_client: Optional boto3 S3 client. If None, creates a new one.

    Returns:
        True if download successful, False otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    # Ensure local directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except Exception as e:
        logger.error(f"Failed to download s3://{bucket_name}/{s3_key}: {e}")
        return False


def list_s3_objects(bucket_name: str, prefix: str = "", suffix: str = "", s3_client=None) -> List[str]:
    """
    List objects in an S3 bucket with optional prefix and suffix filtering.

    Args:
        bucket_name: Name of S3 bucket.
        prefix: Optional prefix to filter objects (e.g., "videos/").
        suffix: Optional suffix to filter objects (e.g., ".mp4").
        s3_client: Optional boto3 S3 client. If None, creates a new one.

    Returns:
        List of S3 object keys matching the criteria.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        objects = []
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if suffix and not key.endswith(suffix):
                        continue
                    objects.append(key)

        return objects
    except NoCredentialsError as e:
        handle_credential_error(e, context="listing S3 objects")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in AWS_CREDENTIAL_ERROR_CODES:
            handle_credential_error(e, context="listing S3 objects")
        logger.error(f"Failed to list S3 objects: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to list S3 objects: {e}")
        raise


def check_s3_object_exists(bucket_name: str, s3_key: str, s3_client=None) -> bool:
    """
    Check if an object exists in S3.

    Args:
        bucket_name: Name of S3 bucket.
        s3_key: S3 object key.
        s3_client: Optional boto3 S3 client. If None, creates a new one.

    Returns:
        True if object exists, False otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client()

    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except Exception:
        return False


def list_videos_in_s3(bucket_name: str, prefix: str, region_name: str = "us-east-1", s3_client=None) -> List[str]:
    """
    List all video files (.mp4) in an S3 bucket prefix.

    Args:
        bucket_name: Name of S3 bucket.
        prefix: Prefix to search under (e.g., "youtube_ads_dataset").
                Add trailing slash (e.g., "youtube_ads_dataset/") for exact prefix match only.
        region_name: AWS region name.
        s3_client: Optional boto3 S3 client. If None, creates a new one.

    Returns:
        List of video IDs (basenames without prefix and directory structure).
        Example: ["vid_-1yXdOufzKE.mp4", "vid_-44_igsZtgU.mp4", ...]
    """
    if s3_client is None:
        s3_client = get_s3_client(region_name=region_name)

    # Check if exact prefix matching is requested (trailing slash)
    exact_match = prefix.endswith("/")
    search_prefix = prefix.rstrip("/")

    # List all .mp4 files in the bucket with the given prefix
    s3_keys = list_s3_objects(bucket_name=bucket_name, prefix=search_prefix, suffix=".mp4", s3_client=s3_client)

    # Extract video IDs (basenames)
    video_ids = []
    for key in s3_keys:
        # If exact match requested, filter to only keys directly under the specified prefix
        if exact_match:
            expected_prefix = f"{search_prefix}/"
            if not key.startswith(expected_prefix):
                continue

        basename = Path(key).name
        # Handle both "vid_xxx.mp4" and "YoutubeAdvertisements_vid_xxx.mp4"
        if basename.startswith("YoutubeAdvertisements_"):
            video_id = basename.replace("YoutubeAdvertisements_", "")
        else:
            video_id = basename
        video_ids.append(video_id)

    return sorted(video_ids)


def download_video_with_cache(
    bucket_name: str, s3_key: str, cache_dir: str, region_name: str = "us-east-1", s3_client=None
) -> Optional[str]:
    """
    Download a video from S3 to local cache. Skip download if already cached.

    Args:
        bucket_name: Name of S3 bucket.
        s3_key: S3 object key (full path in bucket).
        cache_dir: Local directory for caching videos.
        region_name: AWS region name.
        s3_client: Optional boto3 S3 client. If None, creates a new one.

    Returns:
        Path to local cached file if successful, None otherwise.
    """
    if s3_client is None:
        s3_client = get_s3_client(region_name=region_name)

    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Determine local file path
    basename = Path(s3_key).name
    local_path = cache_path / basename

    # Check if already cached
    if local_path.exists():
        return str(local_path)

    # Download from S3
    success = download_file_from_s3(
        bucket_name=bucket_name, s3_key=s3_key, local_path=str(local_path), s3_client=s3_client
    )

    return str(local_path) if success else None


def get_video_paths_from_s3(
    video_ids: List[str], bucket_name: str, prefix: str, cache_dir: str, region_name: str = "us-east-1"
) -> List[str]:
    """
    Get local paths for videos, downloading from S3 if necessary.

    Args:
        video_ids: List of video IDs (e.g., ["vid_-1yXdOufzKE.mp4", ...])
        bucket_name: Name of S3 bucket.
        prefix: S3 prefix where videos are stored.
        cache_dir: Local directory for caching videos.
        region_name: AWS region name.

    Returns:
        List of local file paths to videos.
    """
    s3_client = get_s3_client(region_name=region_name)
    local_paths = []

    # Normalize prefix (remove trailing slash for consistent path construction)
    normalized_prefix = prefix.rstrip("/") if prefix else ""

    for video_id in video_ids:
        # S3 key is just the video_id (e.g., "vid_123.mp4") under the prefix
        # No need to prepend "YoutubeAdvertisements_" since files in S3 are named "vid_*.mp4"
        s3_key = f"{normalized_prefix}/{video_id}" if normalized_prefix else video_id

        # Download with caching
        local_path = download_video_with_cache(
            bucket_name=bucket_name, s3_key=s3_key, cache_dir=cache_dir, region_name=region_name, s3_client=s3_client
        )

        if local_path:
            local_paths.append(local_path)
        else:
            logger.warning(f"   Failed to download {video_id}")

    return local_paths


def build_s3_uri(bucket_name: str, s3_key: str) -> str:
    """Build S3 URI from bucket and key."""
    return f"s3://{bucket_name}/{s3_key}"


def get_s3_uris_for_videos(
    video_ids: List[str],
    bucket_name: str,
    prefix: str = "",
) -> List[str]:
    """Build S3 URIs for a list of video IDs."""
    # Normalize prefix (remove trailing slash for consistent path construction)
    normalized_prefix = prefix.rstrip("/") if prefix else ""

    uris = []
    for video_id in video_ids:
        s3_key = f"{normalized_prefix}/{video_id}" if normalized_prefix else video_id
        uris.append(build_s3_uri(bucket_name, s3_key))
    return uris
