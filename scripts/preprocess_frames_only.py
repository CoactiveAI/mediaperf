"""
Standalone script to run frame sampling preprocessing only.
Loads videos from S3, samples frames, and saves frames to S3.
Tracks timing and estimates costs.

Usage:
    python scripts/preprocess_frames_only.py --config configs/preprocess_config.yaml
    python scripts/preprocess_frames_only.py --num-videos 10 --x-frames 16
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# ruff: noqa: E402
from src.benchmarks.datasets.youtube_ads import extract_video_id
from src.benchmarks.preprocessing.video import sample_frames_numpy
from src.benchmarks.storage.s3 import S3FrameStorage
from src.benchmarks.utils.aws import (
    download_video_with_cache,
    get_s3_client,
    list_videos_in_s3,
)
from src.benchmarks.utils.video_selection import sample_videos


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run frame sampling preprocessing only")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--source-bucket",
        type=str,
        default="benchmarks-project-dev",
        help="Source S3 bucket with videos",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default="youtube_ads_dataset_h264",
        help="Source S3 prefix for videos",
    )
    parser.add_argument(
        "--dest-bucket",
        type=str,
        default="benchmarks-project-dev",
        help="Destination S3 bucket for frames",
    )
    parser.add_argument(
        "--dest-prefix",
        type=str,
        default="frames",
        help="Destination S3 prefix for frames",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-2",
        help="AWS region",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        help="Number of videos to process (if not using config)",
    )
    parser.add_argument(
        "--x-frames",
        type=int,
        default=16,
        help="Number of frames to sample per video",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/videos",
        help="Local cache directory for videos",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for organizing frames (default: auto-generated)",
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Extract parameters from config or args
    source_bucket = config.get("video_source", {}).get("s3", {}).get("bucket", args.source_bucket)
    source_prefix = config.get("video_source", {}).get("s3", {}).get("prefix", args.source_prefix)
    region = config.get("video_source", {}).get("s3", {}).get("region", args.region)

    storage_config = (
        config.get("pipeline", {}).get("preprocessor", {}).get("config", {}).get("storage", {}).get("config", {})
    )
    dest_bucket = storage_config.get("bucket", args.dest_bucket)
    dest_prefix = storage_config.get("prefix", args.dest_prefix)

    x_frames = config.get("pipeline", {}).get("preprocessor", {}).get("config", {}).get("x_frames", args.x_frames)
    cache_dir = config.get("paths", {}).get("cache_dir", args.cache_dir)

    # Generate run_id
    run_id = args.run_id or f"preprocess_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    logger.info("=== Frame Sampling Preprocessing ===")
    logger.info(f"Source: s3://{source_bucket}/{source_prefix}")
    logger.info(f"Destination: s3://{dest_bucket}/{dest_prefix}")
    logger.info(f"Frames per video: {x_frames}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Cache dir: {cache_dir}")

    # Initialize S3 client and storage
    s3_client = get_s3_client(region_name=region)
    storage = S3FrameStorage(
        bucket=dest_bucket,
        prefix=dest_prefix,
        region_name=region,
    )

    # Select videos to process
    logger.info("Selecting videos from S3...")
    video_selection = config.get("video_selection", {})

    # If video_selection config exists and has a mode, use sample_videos
    if video_selection and video_selection.get("mode") in [
        "fixed",
        "random",
        "duration_based",
        "all",
    ]:
        logger.info(f"Using video selection mode: {video_selection.get('mode')}")
        # Config is already in the correct format, just pass it directly
        video_ids, _ = sample_videos(config, Path(cache_dir))
    elif args.num_videos:
        # CLI override: list all and take first N
        logger.info(f"Using CLI override: first {args.num_videos} videos")
        all_video_ids = list_videos_in_s3(
            bucket_name=source_bucket,
            prefix=source_prefix,
            region_name=region,
            s3_client=s3_client,
        )
        video_ids = all_video_ids[: args.num_videos]
    else:
        # Default: process all videos
        logger.info("Processing all videos in bucket")
        all_video_ids = list_videos_in_s3(
            bucket_name=source_bucket,
            prefix=source_prefix,
            region_name=region,
            s3_client=s3_client,
        )
        video_ids = all_video_ids

    logger.info(f"Processing {len(video_ids)} videos")

    # Process videos
    total_start = time.time()
    results = []
    successful = 0
    failed = 0

    for i, video_id in enumerate(video_ids, 1):
        logger.info(f"[{i}/{len(video_ids)}] Processing {video_id}")

        try:
            # Download video
            download_start = time.time()
            s3_key = f"{source_prefix}/{video_id}"
            local_path = download_video_with_cache(
                bucket_name=source_bucket,
                s3_key=s3_key,
                cache_dir=cache_dir,
                region_name=region,
                s3_client=s3_client,
            )
            download_time = time.time() - download_start

            if not local_path:
                logger.error(f"Failed to download {video_id}")
                failed += 1
                continue

            # Sample frames
            sample_start = time.time()
            _, frames_np = sample_frames_numpy(local_path, x=x_frames)
            sample_time = time.time() - sample_start

            # Upload frames to S3
            # Extract video ID without extension to match tagging pipeline behavior
            clean_video_id = extract_video_id(video_id)
            upload_start = time.time()
            storage.save_frames(clean_video_id, frames_np, run_id)
            upload_time = time.time() - upload_start

            total_time = download_time + sample_time + upload_time

            results.append(
                {
                    "video_id": clean_video_id,
                    "num_frames": len(frames_np),
                    "download_time": download_time,
                    "sample_time": sample_time,
                    "upload_time": upload_time,
                    "total_time": total_time,
                }
            )

            successful += 1
            logger.success(
                f"✓ {clean_video_id}: {len(frames_np)} frames in {total_time:.2f}s "
                f"(download: {download_time:.2f}s, sample: {sample_time:.2f}s, upload: {upload_time:.2f}s)"
            )

        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            failed += 1

    total_time = time.time() - total_start

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos processed: {successful}/{len(video_ids)}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time / 60:.2f} min)")

    if successful > 0:
        avg_download = sum(r["download_time"] for r in results) / successful
        avg_sample = sum(r["sample_time"] for r in results) / successful
        avg_upload = sum(r["upload_time"] for r in results) / successful
        avg_total = sum(r["total_time"] for r in results) / successful

        logger.info("\nAverage per video:")
        logger.info(f"  Download time: {avg_download:.2f}s")
        logger.info(f"  Sample time: {avg_sample:.2f}s")
        logger.info(f"  Upload time: {avg_upload:.2f}s")
        logger.info(f"  Total time: {avg_total:.2f}s")

    logger.info("=" * 60)
    logger.info(f"Frames saved to: s3://{dest_bucket}/{dest_prefix}/{run_id}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
