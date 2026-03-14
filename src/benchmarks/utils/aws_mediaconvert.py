import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from loguru import logger

_MEDIACONVERT_ENDPOINT_CACHE = {}


def get_mediaconvert_client(region_name: str = "us-east-2"):
    """
    Initialize MediaConvert client with account-specific endpoint.

    Args:
        region_name: AWS region

    Returns:
        boto3 MediaConvert client
    """
    cache_key = region_name

    if cache_key in _MEDIACONVERT_ENDPOINT_CACHE:
        endpoint_url = _MEDIACONVERT_ENDPOINT_CACHE[cache_key]
        logger.debug(f"Using cached MediaConvert endpoint: {endpoint_url}")
    else:
        logger.info("Fetching MediaConvert endpoint for account...")
        temp_client = boto3.client("mediaconvert", region_name=region_name)
        endpoints = temp_client.describe_endpoints()
        endpoint_url = endpoints["Endpoints"][0]["Url"]
        _MEDIACONVERT_ENDPOINT_CACHE[cache_key] = endpoint_url
        logger.info(f"MediaConvert endpoint: {endpoint_url}")

    client = boto3.client("mediaconvert", region_name=region_name, endpoint_url=endpoint_url)

    return client


def get_video_codec(s3_client, bucket: str, key: str) -> Optional[str]:
    """
    Detect video codec using ffprobe on downloaded video.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        Codec name (e.g., "h264", "av1", "vp9") or None if detection fails
    """
    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        logger.debug(f"Downloading {key} for codec detection...")
        s3_client.download_file(bucket, key, tmp_path)

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            tmp_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        codec = result.stdout.strip().lower()
        logger.debug(f"Detected codec: {codec}")

        Path(tmp_path).unlink()

        return codec

    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {key}: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Codec detection failed for {key}: {e}")
        return None


def create_h264_job_settings(
    input_s3_uri: str,
    output_s3_prefix: str,
    preserve_resolution: bool = True,
    quality_tuning: str = "SINGLE_PASS_HQ",
    max_bitrate: int = 5000000,
    name_modifier: str = "",
    target_height: Optional[int] = None,
) -> Dict:
    """
    Create MediaConvert job settings for H.264 transcoding.

    Args:
        input_s3_uri: Full S3 URI (s3://bucket/key)
        output_s3_prefix: Output S3 prefix (s3://bucket/prefix/)
        preserve_resolution: If True, maintains source resolution (ignored if target_height is set)
        quality_tuning: SINGLE_PASS, SINGLE_PASS_HQ, or MULTI_PASS_HQ
        max_bitrate: Maximum bitrate in bps
        name_modifier: Optional suffix for output filename
        target_height: If set, downscale to this height (e.g., 720 for 720p).
            Width auto-calculated to maintain aspect ratio.

    Returns:
        Job settings dict
    """
    if not output_s3_prefix.endswith("/"):
        output_s3_prefix += "/"

    video_description = {
        "CodecSettings": {
            "Codec": "H_264",
            "H264Settings": {
                "RateControlMode": "QVBR",
                "MaxBitrate": max_bitrate,
                "QualityTuningLevel": quality_tuning,
                "CodecProfile": "HIGH",
                "CodecLevel": "AUTO",
                "InterlaceMode": "PROGRESSIVE",
                "GopSize": 48.0,
                "GopBReference": "DISABLED",
                "FramerateControl": "INITIALIZE_FROM_SOURCE",
                "ParControl": "INITIALIZE_FROM_SOURCE",
                "NumberReferenceFrames": 3,
                "Slices": 1,
            },
        }
    }

    if target_height:
        # Set target height, width auto-calculated to maintain aspect ratio
        video_description["Height"] = target_height
        video_description["ScalingBehavior"] = "DEFAULT"
    elif not preserve_resolution:
        video_description["Width"] = 1920
        video_description["Height"] = 1080
        video_description["ScalingBehavior"] = "DEFAULT"

    job_settings = {
        "Inputs": [
            {
                "FileInput": input_s3_uri,
                "VideoSelector": {},
                "AudioSelectors": {"Audio Selector 1": {"DefaultSelection": "DEFAULT"}},
            }
        ],
        "OutputGroups": [
            {
                "Name": "File Group",
                "OutputGroupSettings": {
                    "Type": "FILE_GROUP_SETTINGS",
                    "FileGroupSettings": {"Destination": output_s3_prefix},
                },
                "Outputs": [
                    {
                        "VideoDescription": video_description,
                        "AudioDescriptions": [
                            {
                                "CodecSettings": {
                                    "Codec": "AAC",
                                    "AacSettings": {
                                        "Bitrate": 128000,
                                        "CodingMode": "CODING_MODE_2_0",
                                        "SampleRate": 48000,
                                    },
                                }
                            }
                        ],
                        "ContainerSettings": {"Container": "MP4", "Mp4Settings": {}},
                    }
                ],
            }
        ],
    }

    if name_modifier:
        job_settings["OutputGroups"][0]["Outputs"][0]["NameModifier"] = name_modifier

    return job_settings


def submit_mediaconvert_job(
    client,
    role_arn: str,
    settings: Dict,
    tags: Optional[Dict[str, str]] = None,
    user_metadata: Optional[Dict[str, str]] = None,
    queue: str = "Default",
) -> Tuple[str, bool]:
    """
    Submit a MediaConvert job with error handling.

    Args:
        client: MediaConvert client
        role_arn: IAM role ARN for MediaConvert
        settings: Job settings dict
        tags: Cost allocation tags
        user_metadata: Job metadata (not for billing)
        queue: Queue name

    Returns:
        Tuple of (job_id, success)
    """
    try:
        kwargs = {"Role": role_arn, "Settings": settings, "Queue": queue}

        if tags:
            kwargs["Tags"] = tags

        if user_metadata:
            kwargs["UserMetadata"] = user_metadata

        response = client.create_job(**kwargs)
        job_id = response["Job"]["Id"]

        return job_id, True

    except client.exceptions.TooManyRequestsException:
        logger.warning("Rate limited by MediaConvert API")
        return None, False
    except client.exceptions.BadRequestException as e:
        logger.error(f"Invalid job settings: {e}")
        return None, False
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        return None, False


def get_job_status(client, job_id: str) -> Optional[Dict]:
    """
    Get MediaConvert job status.

    Args:
        client: MediaConvert client
        job_id: Job ID

    Returns:
        Job status dict or None if failed
    """
    try:
        response = client.get_job(Id=job_id)
        return response["Job"]
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        return None


def monitor_jobs_batch(
    client, job_ids: List[str], poll_interval: int = 30, progress_callback=None
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, Dict]]:
    """
    Monitor multiple MediaConvert jobs with batch polling.

    Args:
        client: MediaConvert client
        job_ids: List of job IDs to monitor
        poll_interval: Seconds between polls
        progress_callback: Optional callback(completed, failed, pending)

    Returns:
        Tuple of (completed_job_ids, failed_jobs_with_errors, timing_data)
    """
    pending = set(job_ids)
    completed = []
    failed = []
    timing_data = {}

    logger.info(f"Monitoring {len(job_ids)} jobs...")

    while pending:
        for job_id in list(pending):
            job = get_job_status(client, job_id)

            if not job:
                continue

            status = job["Status"]

            if status == "COMPLETE":
                completed.append(job_id)
                pending.remove(job_id)

                created_at = job.get("Timing", {}).get("SubmitTime")
                started_at = job.get("Timing", {}).get("StartTime")
                finished_at = job.get("Timing", {}).get("FinishTime")

                duration_seconds = None
                if started_at and finished_at:
                    duration_timedelta = finished_at - started_at
                    duration_seconds = duration_timedelta.total_seconds()

                timing_data[job_id] = {
                    "created_at": created_at.isoformat() if created_at else None,
                    "started_at": started_at.isoformat() if started_at else None,
                    "finished_at": finished_at.isoformat() if finished_at else None,
                    "duration_seconds": duration_seconds,
                }

                duration_str = f" ({duration_seconds:.1f}s)" if duration_seconds else ""
                logger.success(f"Job completed: {job_id}{duration_str}")

            elif status == "ERROR":
                error_msg = job.get("ErrorMessage", "Unknown error")
                failed.append((job_id, error_msg))
                pending.remove(job_id)
                logger.error(f"Job failed: {job_id} - {error_msg}")

            elif status == "CANCELED":
                failed.append((job_id, "Job canceled"))
                pending.remove(job_id)
                logger.warning(f"Job canceled: {job_id}")

        if progress_callback:
            progress_callback(len(completed), len(failed), len(pending))

        if pending:
            logger.info(f"Progress: {len(completed)} completed, {len(failed)} failed, {len(pending)} pending")
            time.sleep(poll_interval)

    logger.success(f"All jobs finished: {len(completed)} completed, {len(failed)} failed")

    return completed, failed, timing_data


def load_checkpoint(checkpoint_file: Path) -> Dict:
    """Load checkpoint data from file."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"completed_videos": [], "failed_videos": [], "job_mapping": {}}


def save_checkpoint(checkpoint_file: Path, data: Dict):
    """Save checkpoint data to file."""
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=2)


def estimate_cost(num_videos: int, avg_duration_seconds: float, pricing_per_minute: float = 0.030) -> float:
    """
    Estimate MediaConvert transcoding cost.

    Args:
        num_videos: Number of videos
        avg_duration_seconds: Average video duration
        pricing_per_minute: Cost per minute (default: $0.030 for HD)

    Returns:
        Estimated cost in USD
    """
    total_minutes = (num_videos * avg_duration_seconds) / 60
    cost = total_minutes * pricing_per_minute
    return cost
