import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from ..constants import MP4_GLOB, YOUTUBE_ADS_PREFIX
from ..enums import VideoSelectionMode, VideoSourceMode, VideoSourceType
from .aws import get_s3_uris_for_videos, list_videos_in_s3
from .gcp import get_gcs_uris_for_videos, list_videos_in_gcs


def discover_videos(config: dict, video_dir: Path = None) -> List[str]:
    """
    Discover available video IDs from the configured source.

    Args:
        config: Configuration dictionary with video_source settings
        video_dir: Directory for local videos (only used if source type is "local")

    Returns:
        List of available video IDs (e.g., ["vid_123.mp4", ...])
    """
    source_type = config["video_source"]["type"]

    if source_type == VideoSourceType.S3.value:
        s3_config = config["video_source"]["s3"]
        video_ids = list_videos_in_s3(
            bucket_name=s3_config["bucket"], prefix=s3_config["prefix"], region_name=s3_config["region"]
        )
        logger.info(f"   Found {len(video_ids)} videos in S3")

    elif source_type == VideoSourceType.GCS.value:
        gcs_config = config["video_source"]["gcs"]
        video_ids = list_videos_in_gcs(
            bucket_name=gcs_config["bucket"], prefix=gcs_config["prefix"], project_id=gcs_config.get("project_id")
        )
        logger.info(f"   Found {len(video_ids)} videos in GCS")

    elif source_type == VideoSourceType.LOCAL.value:
        video_files = list(video_dir.glob(MP4_GLOB))
        video_ids = []
        for vf in video_files:
            if vf.name.startswith(YOUTUBE_ADS_PREFIX):
                video_id = vf.name.replace(YOUTUBE_ADS_PREFIX, "")
            else:
                video_id = vf.name
            video_ids.append(video_id)
        logger.info(f"   Found {len(video_ids)} videos locally")

    else:
        raise ValueError(f"Unknown video source type: {source_type}")

    return video_ids


def select_videos_by_duration(
    metadata_path: str,
    available_videos: List[str],
    target_minutes: float,
    order: str = "longest",
    resolution: Optional[str] = None,
) -> List[str]:
    """
    Args:
        metadata_path: Path to video metadata JSON file
        available_videos: List of available video IDs
        target_minutes: Target total duration in minutes
        order: "longest" to prioritize longest videos first, "shortest" for shortest first
        resolution: Optional resolution filter (e.g., "1920x1080", "854x480")

    Returns:
        List of selected video IDs
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Filter to available videos and extract metadata
    video_info = []
    for video_id in available_videos:
        if video_id in metadata:
            info = metadata[video_id]
            # Apply resolution filter if specified
            if resolution is None or info.get("resolution") == resolution:
                video_info.append(
                    {
                        "video_id": video_id,
                        "duration_seconds": info["duration_seconds"],
                        "resolution": info.get("resolution", "unknown"),
                    }
                )

    if not video_info:
        raise ValueError(f"No videos found matching criteria (resolution={resolution})")

    # Sort by duration
    reverse = order == "longest"
    video_info.sort(key=lambda x: x["duration_seconds"], reverse=reverse)

    # Select videos until target duration is reached
    target_seconds = target_minutes * 60
    selected = []
    total_duration = 0.0

    for video in video_info:
        selected.append(video["video_id"])
        total_duration += video["duration_seconds"]
        if total_duration >= target_seconds:
            break

    actual_minutes = total_duration / 60
    logger.info(f"   Selected {len(selected)} videos")
    logger.info(f"   Total duration: {actual_minutes:.2f} minutes ({actual_minutes / 60:.2f} hours)")
    logger.info(f"   Order: {order.upper()}")
    if resolution:
        logger.info(f"   Resolution filter: {resolution}")

    return selected


def select_video_ids(config: dict, available_videos: List[str]) -> List[str]:
    """
    Select video IDs based on mode (fixed, random, all, or duration_based).

    Args:
        config: Configuration dictionary with video_selection settings
        available_videos: List of available video IDs

    Returns:
        List of selected video IDs
    """
    mode = config["video_selection"]["mode"]

    if mode == VideoSelectionMode.FIXED.value:
        video_list = config["video_selection"]["fixed_videos"]
        logger.info(f"   Mode: FIXED - Using {len(video_list)} specified videos")
        return video_list

    elif mode == VideoSelectionMode.ALL.value:
        logger.info(f"   Mode: ALL - Using all {len(available_videos)} available videos")
        return available_videos

    elif mode == VideoSelectionMode.RANDOM.value:
        n = config["video_selection"]["random_count"]
        seed = config["video_selection"].get("random_seed")

        if seed is not None:
            random.seed(seed)
            logger.info(f"   Mode: RANDOM - Sampling {n} videos (seed={seed})")
        else:
            logger.info(f"   Mode: RANDOM - Sampling {n} videos (no seed)")

        if n > len(available_videos):
            logger.warning(f"   Warning: Requested {n} videos but only {len(available_videos)} available")
            return available_videos
        else:
            return random.sample(available_videos, n)

    elif mode == "duration_based":
        logger.info("   Mode: DURATION_BASED")
        duration_config = config["video_selection"]
        return select_videos_by_duration(
            metadata_path=duration_config["metadata_file"],
            available_videos=available_videos,
            target_minutes=duration_config["target_minutes"],
            order=duration_config.get("order", "longest"),
            resolution=duration_config.get("resolution"),
        )

    else:
        raise ValueError(f"Invalid video selection mode: {mode}")


def resolve_video_sources(config: dict, video_ids: List[str], video_dir: Path = None) -> List[str]:
    """
    Resolve video IDs to paths/URIs based on source type and mode.

    Args:
        config: Configuration dictionary with video_source settings
        video_ids: List of video IDs to resolve
        video_dir: Directory for local videos (only used if source type is "local")

    Returns:
        List of S3/GCS URIs or local file paths
    """
    source_type = config["video_source"]["type"]

    if source_type == VideoSourceType.S3.value:
        source_mode = config["video_source"]["mode"]
        s3_config = config["video_source"]["s3"]

        if source_mode == VideoSourceMode.FROM_S3.value:
            logger.info("   Using S3 URIs directly (no download)")
            return get_s3_uris_for_videos(
                video_ids=video_ids, bucket_name=s3_config["bucket"], prefix=s3_config["prefix"]
            )

        elif source_mode == VideoSourceMode.DOWNLOAD_LOCAL.value:
            logger.info("   Returning S3 URIs")
            return get_s3_uris_for_videos(
                video_ids=video_ids, bucket_name=s3_config["bucket"], prefix=s3_config["prefix"]
            )
        else:
            raise ValueError(f"Unknown S3 mode: {source_mode}")

    elif source_type == VideoSourceType.GCS.value:
        source_mode = config["video_source"]["mode"]
        gcs_config = config["video_source"]["gcs"]

        if source_mode == VideoSourceMode.FROM_GCS.value:
            logger.info("   Using GCS URIs directly (no download)")
            return get_gcs_uris_for_videos(
                video_ids=video_ids, bucket_name=gcs_config["bucket"], prefix=gcs_config["prefix"]
            )

        elif source_mode == VideoSourceMode.DOWNLOAD_LOCAL.value:
            logger.info("   Returning GCS URIs")
            return get_gcs_uris_for_videos(
                video_ids=video_ids, bucket_name=gcs_config["bucket"], prefix=gcs_config["prefix"]
            )
        else:
            raise ValueError(f"Unknown GCS mode: {source_mode}")

    elif source_type == VideoSourceType.LOCAL.value:
        logger.info("   Using local video paths")
        paths = []
        for vid in video_ids:
            # Check if file exists with or without YoutubeAdvertisements_ prefix
            path_with_prefix = video_dir / f"{YOUTUBE_ADS_PREFIX}{vid}"
            path_without_prefix = video_dir / vid

            if path_with_prefix.exists():
                paths.append(str(path_with_prefix))
            elif path_without_prefix.exists():
                paths.append(str(path_without_prefix))
            else:
                paths.append(str(path_without_prefix))
        return paths

    else:
        raise ValueError(f"Unknown video source type: {source_type}")


def sample_videos(config: dict, video_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Args:
        config: Configuration dictionary
        video_dir: Directory for local videos (used when source type is "local")

    Returns:
        Tuple of (video_ids, video_sources)
    """
    logger.info("Discovering available videos...")
    available_videos = discover_videos(config, video_dir)

    logger.info("Selecting videos...")
    selected_video_ids = select_video_ids(config, available_videos)

    logger.info("Resolving video sources...")
    video_sources = resolve_video_sources(config, selected_video_ids, video_dir)

    return selected_video_ids, video_sources
