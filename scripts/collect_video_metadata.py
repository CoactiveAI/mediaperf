import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from src.benchmarks.utils.aws import get_s3_client, list_s3_objects

load_dotenv()


def get_video_metadata_with_ffprobe(presigned_url: str) -> Optional[Dict]:
    """
    Args:
        presigned_url: Pre-signed S3 URL for the video

    Returns:
        Dictionary with duration, width, height, codec, or None if failed
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height,codec_name,codec_type",
            "-of",
            "json",
            presigned_url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"ffprobe failed: {result.stderr}")
            return None

        data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            logger.warning("No video stream found")
            return None

        format_info = data.get("format", {})

        metadata = {
            "duration_seconds": float(format_info.get("duration", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
        }

        return metadata

    except subprocess.TimeoutExpired:
        logger.error("ffprobe timeout")
        return None
    except Exception as e:
        logger.error(f"ffprobe error: {e}")
        return None


def collect_video_metadata(
    bucket_name: str,
    prefix: str,
    region_name: str = "us-east-1",
    output_path: Optional[str] = None,
    url_expiration: int = 3600,
) -> Dict[str, Dict]:
    """
    Args:
        bucket_name: S3 bucket name
        prefix: S3 prefix where videos are stored
        region_name: AWS region name
        output_path: Path to save metadata JSON (optional)
        url_expiration: Pre-signed URL expiration in seconds

    Returns:
        Dictionary mapping video_id to metadata
    """
    logger.info(f"Collecting metadata from s3://{bucket_name}/{prefix}")
    logger.info("Using ffprobe to extract actual video metadata from S3")

    s3_client = get_s3_client(region_name=region_name)

    s3_keys = list_s3_objects(bucket_name=bucket_name, prefix=prefix, suffix=".mp4", s3_client=s3_client)

    logger.info(f"Found {len(s3_keys)} video files")

    metadata = {}
    total_duration = 0.0
    failed_count = 0

    for i, s3_key in enumerate(s3_keys, 1):
        video_id = Path(s3_key).name

        if video_id.startswith("YoutubeAdvertisements_"):
            video_id = video_id.replace("YoutubeAdvertisements_", "")

        try:
            # Get file size from S3
            response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            file_size_bytes = response["ContentLength"]

            # Generate pre-signed URL
            presigned_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": s3_key},
                ExpiresIn=url_expiration,
            )

            # Get video metadata using ffprobe
            video_info = get_video_metadata_with_ffprobe(presigned_url)

            if video_info:
                metadata[video_id] = {
                    "file_size_bytes": file_size_bytes,
                    "duration_seconds": video_info["duration_seconds"],
                    "width": video_info["width"],
                    "height": video_info["height"],
                    "codec": video_info["codec"],
                    "resolution": f"{video_info['width']}x{video_info['height']}",
                    "s3_key": s3_key,
                    "s3_uri": f"s3://{bucket_name}/{s3_key}",
                }

                total_duration += video_info["duration_seconds"]

                if i % 50 == 0:
                    logger.info(f"Processed {i}/{len(s3_keys)} videos... ({failed_count} failed)")
            else:
                failed_count += 1
                logger.warning(f"Failed to extract metadata for {video_id}")

        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process {s3_key}: {e}")
            continue

    logger.success(f"Collected metadata for {len(metadata)} videos")
    logger.info(f"Failed: {failed_count} videos")
    logger.info(f"Total duration: {total_duration / 60:.2f} minutes ({total_duration / 3600:.2f} hours)")

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"Saved metadata to {output_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Collect video metadata from S3 using ffprobe")
    parser.add_argument("--bucket", type=str, default="benchmarks-project-dev", help="S3 bucket name")
    parser.add_argument(
        "--prefix",
        type=str,
        default="youtube_ads_dataset/",
        help="S3 prefix where videos are stored",
    )
    parser.add_argument("--region", type=str, default="us-east-2", help="AWS region name")
    parser.add_argument(
        "--output",
        type=str,
        default="data/inputs/youtube_ads_video_metadata.json",
        help="Output path for metadata JSON file",
    )
    parser.add_argument(
        "--url-expiration",
        type=int,
        default=3600,
        help="Pre-signed URL expiration in seconds (default: 3600)",
    )

    args = parser.parse_args()

    collect_video_metadata(
        bucket_name=args.bucket,
        prefix=args.prefix,
        region_name=args.region,
        output_path=args.output,
        url_expiration=args.url_expiration,
    )


if __name__ == "__main__":
    main()
