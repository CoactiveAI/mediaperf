import base64
import time
from typing import List, Tuple

from loguru import logger

from ..datasets.youtube_ads import extract_video_id
from ..utils.aws import download_video_with_cache
from .base import VideoPreprocessor


class VideoBase64Preprocessor(VideoPreprocessor):
    def __init__(
        self,
        cache_dir: str = "data/cache/videos",
        run_id: str = None,
        cache_run_id: str = None,
        config: dict = None,
    ):
        self.cache_dir = cache_dir
        self.config = config or {}
        # run_id, cache_run_id, and config not used for VideoBase64Preprocessor
        # but accepted for compatibility with build_preprocessor()

    def preprocess(self, video_sources: List[str]) -> List[Tuple[str, str, str, float]]:
        """
        Args:
            video_sources: List of video paths (local or S3 URIs)

        Returns:
            List of (video_id, video_path, base64_video_string, preprocessing_time_seconds)
        """
        video_data = []

        for video_source in video_sources:
            video_id = extract_video_id(video_source)

            try:
                start_time = time.time()

                # Download from S3 if needed
                if video_source.startswith("s3://"):
                    local_path = self._download_from_s3(video_source)
                    if local_path is None:
                        raise ValueError(f"Failed to download {video_source}")
                else:
                    local_path = video_source

                # Read and encode video
                with open(local_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

                preprocessing_time = time.time() - start_time

                video_data.append((video_id, video_source, video_base64, preprocessing_time))

            except Exception as e:
                logger.error(f"Failed to encode video {video_id}: {e}")
                video_data.append((video_id, video_source, "", 0.0))

        return video_data

    def _download_from_s3(self, s3_uri: str) -> str:
        """Download video from S3 to local cache."""
        # Parse S3 URI: s3://bucket/prefix/video.mp4
        s3_uri_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = s3_uri_parts[0]
        s3_key = s3_uri_parts[1]

        # Extract region from bucket name if present (e.g., bucket-us-east-2)
        # Default to us-east-1
        region_name = "us-east-2"

        local_path = download_video_with_cache(
            bucket_name=bucket_name,
            s3_key=s3_key,
            cache_dir=self.cache_dir,
            region_name=region_name,
        )

        return local_path
