"""GCP Cloud Storage utilities for cloud storage operations."""

import os
from pathlib import Path
from typing import List, Optional

from google.api_core.exceptions import Forbidden, Unauthenticated
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import storage
from loguru import logger

from ..exceptions.credential_errors import handle_credential_error


def get_gcs_client(project_id: Optional[str] = None, credentials_path: Optional[str] = None):
    """
    Create and return a Google Cloud Storage client.

    Args:
        project_id: GCP project ID. If None, uses default project from credentials.
        credentials_path: Path to service account JSON key file. If None, uses
                         GOOGLE_APPLICATION_CREDENTIALS environment variable or
                         default credentials.

    Returns:
        google.cloud.storage.Client instance
    """
    try:
        if credentials_path:
            return storage.Client.from_service_account_json(credentials_path, project=project_id)
        else:
            # Use default credentials (environment variable or application default)
            return storage.Client(project=project_id)
    except (DefaultCredentialsError, Forbidden, Unauthenticated) as e:
        handle_credential_error(e, context="creating GCS client")
    except Exception as e:
        logger.error(f"Failed to create GCS client: {e}")
        raise


def download_blob_from_gcs(
    bucket_name: str, blob_name: str, local_path: str, gcs_client=None, verbose: bool = True
) -> bool:
    """
    Download a blob from GCS.

    Args:
        bucket_name: Name of GCS bucket.
        blob_name: Name of blob (object key) in bucket.
        local_path: Local path where file should be saved.
        gcs_client: Optional GCS client. If None, creates a new one.
        verbose: If True, log success/error messages. Default True.

    Returns:
        True if download successful, False otherwise.
    """
    if gcs_client is None:
        gcs_client = get_gcs_client()

    # Ensure local directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        if verbose:
            logger.error(f"Failed to download gs://{bucket_name}/{blob_name}: {e}")
        return False


def upload_blob_to_gcs(
    local_path: str, bucket_name: str, blob_name: str, gcs_client=None, content_type: Optional[str] = None
) -> bool:
    """
    Upload a file to GCS.

    Args:
        local_path: Path to local file to upload.
        bucket_name: Name of GCS bucket.
        blob_name: Name for blob (object key) in bucket.
        gcs_client: Optional GCS client. If None, creates a new one.
        content_type: Optional content type (e.g., 'video/mp4').

    Returns:
        True if upload successful, False otherwise.
    """
    if gcs_client is None:
        gcs_client = get_gcs_client()

    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if content_type:
            blob.upload_from_filename(local_path, content_type=content_type)
        else:
            blob.upload_from_filename(local_path)

        logger.success(f"Uploaded {local_path} to gs://{bucket_name}/{blob_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return False


def list_gcs_blobs(bucket_name: str, prefix: str = "", suffix: str = "", gcs_client=None) -> List[str]:
    """
    List blobs in a GCS bucket with optional prefix and suffix filtering.

    Args:
        bucket_name: Name of GCS bucket.
        prefix: Optional prefix to filter blobs (e.g., "videos/").
        suffix: Optional suffix to filter blobs (e.g., ".mp4").
        gcs_client: Optional GCS client. If None, creates a new one.

    Returns:
        List of blob names matching the criteria.
    """
    if gcs_client is None:
        gcs_client = get_gcs_client()

    try:
        bucket = gcs_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        blob_names = []
        for blob in blobs:
            if suffix and not blob.name.endswith(suffix):
                continue
            blob_names.append(blob.name)

        return blob_names
    except (DefaultCredentialsError, Forbidden, Unauthenticated) as e:
        handle_credential_error(e, context="listing GCS blobs")
    except Exception as e:
        logger.error(f"Failed to list GCS blobs: {e}")
        raise


def check_gcs_blob_exists(bucket_name: str, blob_name: str, gcs_client=None) -> bool:
    """
    Check if a blob exists in GCS.

    Args:
        bucket_name: Name of GCS bucket.
        blob_name: Name of blob to check.
        gcs_client: Optional GCS client. If None, creates a new one.

    Returns:
        True if blob exists, False otherwise.
    """
    if gcs_client is None:
        gcs_client = get_gcs_client()

    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking blob existence: {e}")
        return False


def get_blob_metadata(bucket_name: str, blob_name: str, gcs_client=None) -> Optional[dict]:
    """
    Get metadata for a GCS blob.

    Args:
        bucket_name: Name of GCS bucket.
        blob_name: Name of blob.
        gcs_client: Optional GCS client. If None, creates a new one.

    Returns:
        Dictionary containing blob metadata (size, content_type, etc.) or None if error.
    """
    if gcs_client is None:
        gcs_client = get_gcs_client()

    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.reload()

        return {
            "name": blob.name,
            "size": blob.size,
            "content_type": blob.content_type,
            "updated": blob.updated,
            "md5_hash": blob.md5_hash,
            "crc32c": blob.crc32c,
        }
    except Exception as e:
        logger.error(f"Failed to get blob metadata: {e}")
        return None


# Video-specific utilities


def list_videos_in_gcs(bucket_name: str, prefix: str, project_id: Optional[str] = None, gcs_client=None) -> List[str]:
    """
    List all video files (.mp4) in a GCS bucket prefix.

    Args:
        bucket_name: Name of GCS bucket.
        prefix: Prefix to search under (e.g., "YoutubeAdvertisements").
        project_id: Optional GCP project ID.
        gcs_client: Optional GCS client. If None, creates a new one.

    Returns:
        List of video IDs (basenames without prefix and directory structure).
        Example: ["vid_-1yXdOufzKE.mp4", "vid_-44_igsZtgU.mp4", ...]
    """
    if gcs_client is None:
        gcs_client = get_gcs_client(project_id=project_id)

    # List all .mp4 files in the bucket with the given prefix
    blob_names = list_gcs_blobs(bucket_name=bucket_name, prefix=prefix, suffix=".mp4", gcs_client=gcs_client)

    # Extract video IDs (basenames)
    video_ids = []
    for blob_name in blob_names:
        basename = Path(blob_name).name
        # Handle both "vid_xxx.mp4" and "YoutubeAdvertisements_vid_xxx.mp4"
        if basename.startswith("YoutubeAdvertisements_"):
            video_id = basename.replace("YoutubeAdvertisements_", "")
        else:
            video_id = basename
        video_ids.append(video_id)

    return sorted(video_ids)


def get_gcs_uris_for_videos(
    video_ids: List[str],
    bucket_name: str,
    prefix: str = "",
) -> List[str]:
    """
    Build GCS URIs for a list of video IDs.

    Args:
        video_ids: List of video IDs (e.g., ["vid_-1yXdOufzKE.mp4", ...])
        bucket_name: Name of GCS bucket.
        prefix: GCS prefix where videos are stored.

    Returns:
        List of GCS URIs (e.g., ["gs://bucket/prefix/vid_123.mp4", ...])
    """
    uris = []
    for video_id in video_ids:
        blob_name = f"{prefix}/{video_id}" if prefix else video_id
        uri = f"gs://{bucket_name}/{blob_name}"
        uris.append(uri)
    return uris


def get_video_paths_from_gcs(
    video_ids: List[str], bucket_name: str, prefix: str, cache_dir: str, project_id: Optional[str] = None
) -> List[str]:
    """
    Get local paths for videos, downloading from GCS if necessary.

    Args:
        video_ids: List of video IDs (e.g., ["vid_-1yXdOufzKE.mp4", ...])
        bucket_name: Name of GCS bucket.
        prefix: GCS prefix where videos are stored.
        cache_dir: Local directory for caching videos.
        project_id: Optional GCP project ID.

    Returns:
        List of local file paths to videos.
    """
    gcs_client = get_gcs_client(project_id=project_id)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    local_paths = []

    for video_id in video_ids:
        # GCS blob name is just the video_id under the prefix
        blob_name = f"{prefix}/{video_id}" if prefix else video_id
        local_path = str(cache_path / video_id)

        # Check if already cached
        if os.path.exists(local_path):
            local_paths.append(local_path)
            continue

        # Download from GCS
        success = download_blob_from_gcs(
            bucket_name=bucket_name, blob_name=blob_name, local_path=local_path, gcs_client=gcs_client
        )

        if success:
            local_paths.append(local_path)
        else:
            logger.warning(f"   Failed to download {video_id}")

    return local_paths
