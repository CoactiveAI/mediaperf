#!/usr/bin/env python3
"""
Convert videos to H.264 codec using AWS MediaConvert.

Optimized for batch processing of 2000+ short videos with:
- Codec detection (skip H.264 videos)
- Rate-limited concurrent submission
- Checkpoint/resume capability
- Batch monitoring
- Cost tracking with tags
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmarks.utils.aws import get_s3_client, list_s3_objects
from benchmarks.utils.aws_mediaconvert import (
    create_h264_job_settings,
    estimate_cost,
    get_mediaconvert_client,
    get_video_codec,
    load_checkpoint,
    monitor_jobs_batch,
    save_checkpoint,
    submit_mediaconvert_job,
)
from benchmarks.utils.video_selection import sample_videos


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0

    def wait(self):
        """Wait if necessary to maintain rate limit."""
        now = time.time()
        time_since_last = now - self.last_call
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        self.last_call = time.time()


def filter_videos_by_codec(
    s3_client,
    bucket: str,
    video_keys: List[str],
    target_codec: str = "h264",
    check_sample_size: Optional[int] = None,
) -> List[str]:
    """
    Filter out videos that already have the target codec.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        video_keys: List of video S3 keys
        target_codec: Codec to skip (default: h264)
        check_sample_size: If provided, only check first N videos

    Returns:
        List of video keys that need conversion
    """
    videos_to_convert = []
    videos_skipped = 0

    check_keys = video_keys[:check_sample_size] if check_sample_size else video_keys

    logger.info(f"Checking codecs for {len(check_keys)} videos...")

    for i, key in enumerate(check_keys, 1):
        if i % 50 == 0:
            logger.info(f"Checked {i}/{len(check_keys)} videos...")

        codec = get_video_codec(s3_client, bucket, key)

        if codec == target_codec:
            logger.debug(f"Skipping {key} (already {target_codec})")
            videos_skipped += 1
        elif codec is None:
            logger.warning(f"Could not detect codec for {key}, will attempt conversion")
            videos_to_convert.append(key)
        else:
            logger.debug(f"Will convert {key} ({codec} -> {target_codec})")
            videos_to_convert.append(key)

    logger.info(f"Codec check complete: {len(videos_to_convert)} to convert, {videos_skipped} already {target_codec}")

    return videos_to_convert


def submit_jobs_batch(
    mc_client,
    role_arn: str,
    source_bucket: str,
    video_keys: List[str],
    output_prefix: str,
    checkpoint_data: dict,
    rate_limiter: RateLimiter,
    tags: dict,
    quality_tuning: str = "SINGLE_PASS_HQ",
    max_bitrate: int = 5000000,
    progress_interval: int = 50,
    target_height: Optional[int] = None,
) -> dict:
    """
    Submit MediaConvert jobs for videos with rate limiting.

    Args:
        mc_client: MediaConvert client
        role_arn: IAM role ARN
        source_bucket: Source S3 bucket
        video_keys: List of video keys to convert
        output_prefix: Output S3 prefix
        checkpoint_data: Checkpoint dict to update
        rate_limiter: Rate limiter instance
        tags: Cost allocation tags
        quality_tuning: Quality setting
        max_bitrate: Max bitrate
        progress_interval: Log progress every N videos
        target_height: Target height for downscaling (e.g., 720)

    Returns:
        Updated checkpoint data
    """
    job_mapping = checkpoint_data.get("job_mapping", {})
    submitted_count = 0
    failed_submissions = []

    output_s3_prefix = f"s3://{source_bucket}/{output_prefix}"

    logger.info(f"Submitting {len(video_keys)} jobs...")

    for video_key in video_keys:
        if video_key in checkpoint_data.get("completed_videos", []):
            logger.debug(f"Skipping {video_key} (in checkpoint)")
            continue

        rate_limiter.wait()

        input_s3_uri = f"s3://{source_bucket}/{video_key}"

        job_settings = create_h264_job_settings(
            input_s3_uri=input_s3_uri,
            output_s3_prefix=output_s3_prefix,
            preserve_resolution=True,
            quality_tuning=quality_tuning,
            max_bitrate=max_bitrate,
            target_height=target_height,
        )

        user_metadata = {"source_key": video_key, "source_bucket": source_bucket}

        job_id, success = submit_mediaconvert_job(
            client=mc_client,
            role_arn=role_arn,
            settings=job_settings,
            tags=tags,
            user_metadata=user_metadata,
        )

        if success:
            job_mapping[job_id] = video_key
            submitted_count += 1

            if submitted_count % progress_interval == 0:
                logger.info(f"Submitted {submitted_count}/{len(video_keys)} jobs")
        else:
            failed_submissions.append(video_key)
            logger.warning(f"Failed to submit job for {video_key}")

    checkpoint_data["job_mapping"] = job_mapping

    logger.success(f"Job submission complete: {submitted_count} submitted, {len(failed_submissions)} failed")

    if failed_submissions:
        logger.warning(f"Failed submissions: {failed_submissions[:10]}...")

    return checkpoint_data


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Convert videos to H.264 using AWS MediaConvert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file (CLI args override config values)",
    )
    parser.add_argument(
        "--source-bucket",
        help="Source S3 bucket containing videos",
    )
    parser.add_argument(
        "--source-prefix",
        help="Source S3 prefix (folder path)",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output S3 prefix (e.g., 'videos_h264/')",
    )
    parser.add_argument(
        "--role-arn",
        help="IAM role ARN for MediaConvert",
    )
    parser.add_argument("--region", help="AWS region")
    parser.add_argument(
        "--check-codec",
        action="store_true",
        help="Check video codecs and skip H.264 videos",
    )
    parser.add_argument(
        "--quality",
        choices=["SINGLE_PASS", "SINGLE_PASS_HQ", "MULTI_PASS_HQ"],
        help="Quality tuning level",
    )
    parser.add_argument("--max-bitrate", type=int, help="Maximum bitrate in bps")
    parser.add_argument("--rate-limit", type=int, help="Job submissions per second")
    parser.add_argument(
        "--poll-interval",
        type=int,
        help="Job monitoring poll interval (seconds)",
    )
    parser.add_argument("--progress-interval", type=int, help="Log progress every N videos")
    parser.add_argument(
        "--checkpoint-file",
        help="Checkpoint file path (auto-generated if not provided)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List videos but don't submit jobs")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process (for testing)")
    parser.add_argument(
        "--avg-duration",
        type=float,
        help="Average video duration in seconds (for cost estimate)",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["all", "fixed", "random", "duration_based"],
        help="Video selection mode",
    )
    parser.add_argument(
        "--target-minutes",
        type=int,
        help="Target duration in minutes (required for duration_based mode)",
    )
    parser.add_argument(
        "--metadata-file",
        help="Video metadata file with durations (for duration_based mode)",
    )
    parser.add_argument(
        "--selection-order",
        choices=["longest", "shortest"],
        help="Order for duration_based selection",
    )
    parser.add_argument(
        "--resolution",
        help="Filter by resolution (e.g., '1920x1080') for duration_based mode",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        help="Maximum output height in pixels (e.g., 720 for 720p). "
        "Videos larger than this will be downscaled while maintaining aspect ratio.",
    )

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)

    # Extract values from config with CLI overrides (defaults defined here)
    video_source_config = config.get("video_source", {})
    s3_config = video_source_config.get("s3", {})
    source_bucket = args.source_bucket or s3_config.get("bucket", "benchmarks-project-dev")
    source_prefix = args.source_prefix or s3_config.get("prefix", "youtube_ads_dataset/")
    region = args.region or s3_config.get("region", "us-east-2")

    mediaconvert_config = config.get("mediaconvert", {})
    output_prefix = args.output_prefix or mediaconvert_config.get(
        "output_prefix", "youtube_ads_dataset_h264_mediaconvert/"
    )
    role_arn = args.role_arn or mediaconvert_config.get("role_arn", "arn:aws:iam::443102425374:role/MediaConvertRole")
    quality = args.quality or mediaconvert_config.get("quality", "SINGLE_PASS")
    max_bitrate = args.max_bitrate or mediaconvert_config.get("max_bitrate", 5000000)
    rate_limit = args.rate_limit or mediaconvert_config.get("rate_limit", 10)
    poll_interval = args.poll_interval or mediaconvert_config.get("poll_interval", 30)
    progress_interval = args.progress_interval or mediaconvert_config.get("progress_interval", 50)
    check_codec = args.check_codec or mediaconvert_config.get("check_codec", False)
    avg_duration = args.avg_duration or mediaconvert_config.get("avg_duration", 90.0)
    max_height = args.max_height or mediaconvert_config.get("max_height")

    # Video selection config
    video_selection = config.get("video_selection", {})
    selection_mode = args.selection_mode or video_selection.get("mode", "all")
    target_minutes = args.target_minutes or video_selection.get("target_minutes")
    default_metadata = "data/inputs/youtube_ads_video_metadata.json"
    metadata_file = args.metadata_file or video_selection.get("metadata_file", default_metadata)
    selection_order = args.selection_order or video_selection.get("order", "longest")
    resolution = args.resolution or video_selection.get("resolution")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get output paths from config
    outputs_dir = Path(config["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = outputs_dir / f"mediaconvert_checkpoint_{timestamp}.json"
    results_file = outputs_dir / f"mediaconvert_results_{timestamp}.json"

    # Validate duration_based mode requirements
    if selection_mode == "duration_based" and not target_minutes:
        logger.error("target_minutes is required when using selection_mode 'duration_based'")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("AWS MediaConvert Video Conversion")
    logger.info("=" * 80)
    logger.info(f"Source: s3://{source_bucket}/{source_prefix}")
    logger.info(f"Output: s3://{source_bucket}/{output_prefix}")
    logger.info(f"Role ARN: {role_arn}")
    logger.info(f"Quality: {quality}")
    logger.info(f"Max height: {max_height if max_height else 'preserve source'}")
    logger.info(f"Check codec: {check_codec}")
    logger.info(f"Selection mode: {selection_mode}")
    if selection_mode == "duration_based":
        logger.info(f"Target duration: {target_minutes} minutes")
        logger.info(f"Selection order: {selection_order}")
        if resolution:
            logger.info(f"Resolution filter: {resolution}")
    logger.info(f"Checkpoint: {checkpoint_file}")
    logger.info("=" * 80)

    s3_client = get_s3_client(region_name=region)
    mc_client = get_mediaconvert_client(region_name=region)

    logger.info("Selecting videos from S3...")

    if selection_mode == "all":
        # Use simple listing for "all" mode
        video_keys = list_s3_objects(bucket_name=source_bucket, prefix=source_prefix, suffix=".mp4")
    else:
        # Use sample_videos for other selection modes
        # Construct config dict in the format expected by sample_videos()
        video_selection_config = {
            "mode": selection_mode,
        }

        if selection_mode == "duration_based":
            video_selection_config.update(
                {
                    "target_minutes": target_minutes,
                    "metadata_file": metadata_file,
                    "order": selection_order,
                    "resolution": resolution,
                }
            )
        elif selection_mode == "random":
            # For random mode, use limit if provided
            if args.limit:
                video_selection_config["num_videos"] = args.limit
                video_selection_config["seed"] = 42  # Fixed seed for reproducibility

        # Build config dict for sample_videos()
        sample_config = {
            "video_source": {
                "type": "s3",
                "mode": "from_s3",  # Use S3 URIs directly
                "s3": {
                    "bucket": source_bucket,
                    "prefix": source_prefix,
                    "region": region,
                },
            },
            "video_selection": video_selection_config,
        }

        # Call sample_videos with proper signature
        video_ids, _ = sample_videos(sample_config, Path("."))

        # Convert video IDs to S3 keys (add prefix if not already present)
        video_keys = []
        for vid in video_ids:
            if vid.startswith(source_prefix):
                video_keys.append(vid)
            else:
                # Remove prefix from video_id if it was added, then add it back
                clean_vid = vid.replace(f"{source_prefix}/", "")
                video_keys.append(f"{source_prefix}{clean_vid}")

    if not video_keys:
        logger.error("No videos found in source bucket/prefix")
        sys.exit(1)

    logger.info(f"Selected {len(video_keys)} videos")

    if args.limit:
        video_keys = video_keys[: args.limit]
        logger.info(f"Limited to {len(video_keys)} videos for testing")

    checkpoint_data = load_checkpoint(checkpoint_file)
    completed_videos = set(checkpoint_data.get("completed_videos", []))

    videos_to_process = [k for k in video_keys if k not in completed_videos]

    if len(videos_to_process) < len(video_keys):
        logger.info(
            f"Resuming from checkpoint: {len(completed_videos)} already completed, {len(videos_to_process)} remaining"
        )

    if check_codec:
        logger.info("Checking video codecs (this may take a while)...")
        videos_to_process = filter_videos_by_codec(
            s3_client=s3_client,
            bucket=source_bucket,
            video_keys=videos_to_process,
            target_codec="h264",
        )

    if not videos_to_process:
        logger.success("No videos need conversion!")
        sys.exit(0)

    estimated_cost = estimate_cost(
        num_videos=len(videos_to_process),
        avg_duration_seconds=avg_duration,
        pricing_per_minute=0.030,
    )

    logger.info("=" * 80)
    logger.info("COST ESTIMATE")
    logger.info("=" * 80)
    logger.info(f"Videos to process: {len(videos_to_process)}")
    logger.info(f"Avg duration: {avg_duration}s")
    logger.info(f"Estimated cost: ${estimated_cost:.2f}")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("Dry run complete - no jobs submitted")
        logger.info(f"Videos to convert: {len(videos_to_process)}")
        if videos_to_process[:5]:
            logger.info("First 5 videos:")
            for key in videos_to_process[:5]:
                logger.info(f"  - {key}")
        sys.exit(0)

    input(f"\nPress ENTER to proceed with conversion of {len(videos_to_process)} videos, or Ctrl+C to cancel...")

    # Cost allocation tags
    tags = mediaconvert_config.get("tags", {"project": "benchmarks_project_mediaconvert"})
    logger.info(f"Cost allocation tags: {tags}")

    logger.info("\n" + "=" * 80)
    logger.info("SUBMITTING JOBS")
    logger.info("=" * 80)

    rate_limiter = RateLimiter(calls_per_second=rate_limit)

    script_start_time = time.time()

    try:
        checkpoint_data = submit_jobs_batch(
            mc_client=mc_client,
            role_arn=role_arn,
            source_bucket=source_bucket,
            video_keys=videos_to_process,
            output_prefix=output_prefix,
            checkpoint_data=checkpoint_data,
            rate_limiter=rate_limiter,
            tags=tags,
            quality_tuning=quality,
            max_bitrate=max_bitrate,
            progress_interval=progress_interval,
            target_height=max_height,
        )

        save_checkpoint(checkpoint_file, checkpoint_data)

        job_ids = list(checkpoint_data["job_mapping"].keys())

        if not job_ids:
            logger.warning("No jobs were submitted")
            sys.exit(1)

        logger.info("\n" + "=" * 80)
        logger.info("MONITORING JOBS")
        logger.info("=" * 80)
        logger.info(f"Tracking {len(job_ids)} jobs...")

        avg_processing_time_per_video = 8
        max_concurrent = 100

        if len(job_ids) <= max_concurrent:
            estimated_minutes = avg_processing_time_per_video
        else:
            batches = len(job_ids) / max_concurrent
            estimated_minutes = batches * avg_processing_time_per_video

        if estimated_minutes < 60:
            time_estimate = f"~{estimated_minutes:.0f} minutes"
        else:
            time_estimate = f"~{estimated_minutes / 60:.1f} hours"

        logger.info(f"Estimated completion time: {time_estimate}")

        def progress_callback(completed, failed, _pending):
            if (completed + failed) % 10 == 0 and (completed + failed) > 0:
                save_checkpoint(checkpoint_file, checkpoint_data)

        completed_jobs, failed_jobs, timing_data = monitor_jobs_batch(
            client=mc_client,
            job_ids=job_ids,
            poll_interval=poll_interval,
            progress_callback=progress_callback,
        )

        for job_id in completed_jobs:
            video_key = checkpoint_data["job_mapping"][job_id]
            if video_key not in checkpoint_data.get("completed_videos", []):
                checkpoint_data.setdefault("completed_videos", []).append(video_key)

        for job_id, error in failed_jobs:
            video_key = checkpoint_data["job_mapping"][job_id]
            checkpoint_data.setdefault("failed_videos", []).append(
                {"video_key": video_key, "job_id": job_id, "error": error}
            )

        save_checkpoint(checkpoint_file, checkpoint_data)

        script_end_time = time.time()
        total_duration = script_end_time - script_start_time

        video_timings = []
        for job_id in completed_jobs:
            video_key = checkpoint_data["job_mapping"][job_id]
            if job_id in timing_data and timing_data[job_id]["duration_seconds"]:
                video_timings.append(
                    {
                        "video_key": video_key,
                        "job_id": job_id,
                        "duration_seconds": timing_data[job_id]["duration_seconds"],
                    }
                )

        timing_stats = {}
        if video_timings:
            durations = [t["duration_seconds"] for t in video_timings]
            timing_stats = {
                "total_videos": len(video_timings),
                "avg_seconds": sum(durations) / len(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "total_conversion_seconds": sum(durations),
            }

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time_seconds": total_duration,
            "total_execution_time_formatted": f"{total_duration / 3600:.2f} hours",
            "completed_videos": len(completed_jobs),
            "failed_videos": len(failed_jobs),
            "timing_statistics": timing_stats,
            "per_video_timings": video_timings,
            "source_bucket": source_bucket,
            "source_prefix": source_prefix,
            "output_prefix": output_prefix,
            "quality": quality,
            "estimated_cost": estimated_cost,
        }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.success("CONVERSION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Completed: {len(completed_jobs)}")
        logger.info(f"Failed: {len(failed_jobs)}")
        logger.info(f"Total execution time: {total_duration / 3600:.2f} hours ({total_duration / 60:.1f} minutes)")

        if timing_stats:
            logger.info("\nConversion Time Statistics:")
            logger.info(f"  Average per video: {timing_stats['avg_seconds']:.1f}s")
            logger.info(f"  Min: {timing_stats['min_seconds']:.1f}s")
            logger.info(f"  Max: {timing_stats['max_seconds']:.1f}s")
            logger.info(f"  Total conversion time: {timing_stats['total_conversion_seconds'] / 3600:.2f} hours")

        logger.info(f"\nOutput location: s3://{source_bucket}/{output_prefix}")
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        logger.info(f"Results saved: {results_file}")
        logger.info("=" * 80)

        logger.info("\nTo view costs:")
        logger.info("1. Go to AWS Cost Explorer")
        logger.info("2. Filter by Service: AWS Elemental MediaConvert")
        logger.info("3. Filter by Tags:")
        for tag_key, tag_value in tags.items():
            logger.info(f"   - {tag_key} = {tag_value}")

        if failed_jobs:
            logger.warning(f"\n{len(failed_jobs)} jobs failed. See checkpoint file for details.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user. Saving checkpoint...")
        save_checkpoint(checkpoint_file, checkpoint_data)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        logger.info("Run again with the same parameters to resume.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        save_checkpoint(checkpoint_file, checkpoint_data)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
