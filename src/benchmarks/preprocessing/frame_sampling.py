import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from ..datasets.youtube_ads import extract_video_id
from ..enums import VideoSourceMode, VideoSourceType
from ..storage.base import FrameStorage
from ..utils.aws import download_video_with_cache
from ..utils.gcp import download_blob_from_gcs, get_gcs_client
from ..utils.preprocessing_stats import FrameSamplingTracker
from .base import VideoPreprocessor
from .video import sample_frames_base64, sample_frames_numpy


class FrameSamplingPreprocessor(VideoPreprocessor):
    def __init__(
        self,
        x_frames: int = 16,
        storage_backend: Optional[FrameStorage] = None,
        run_id: Optional[str] = None,
        cache_run_id: Optional[str] = None,
        output_format: str = "base64",
        config: Optional[Dict] = None,
    ):
        self.x_frames = x_frames
        self.storage_backend = storage_backend
        self.run_id = run_id
        self.cache_run_id = cache_run_id
        self.output_format = output_format
        self.config = config or {}

    def _download(self, video_source: str, source_type: str) -> str:
        """
        Download video from S3/GCS URI.

        Args:
            video_source: S3 or GCS URI
            source_type: "s3" or "gcs"

        Returns:
            Local file path
        """
        cache_dir = self.config.get("paths", {}).get("cache_dir", "data/cache/videos")

        if source_type == VideoSourceType.S3.value:
            s3_config = self.config.get("video_source", {}).get("s3", {})

            s3_uri_parts = video_source.replace("s3://", "").split("/", 1)
            bucket_name = s3_uri_parts[0]
            s3_key = s3_uri_parts[1] if len(s3_uri_parts) > 1 else ""

            local_path = download_video_with_cache(
                bucket_name=bucket_name,
                s3_key=s3_key,
                cache_dir=cache_dir,
                region_name=s3_config.get("region"),
            )
            return local_path

        elif source_type == VideoSourceType.GCS.value:
            gcs_config = self.config.get("video_source", {}).get("gcs", {})
            video_id = extract_video_id(video_source)

            gcs_uri_parts = video_source.replace("gs://", "").split("/", 1)
            bucket_name = gcs_uri_parts[0]
            blob_name = gcs_uri_parts[1] if len(gcs_uri_parts) > 1 else ""

            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            local_path = str(cache_path / video_id)

            if not Path(local_path).exists():
                gcs_client = get_gcs_client(project_id=gcs_config.get("project_id"))
                download_blob_from_gcs(
                    bucket_name=bucket_name, blob_name=blob_name, local_path=local_path, gcs_client=gcs_client
                )

            return local_path

        else:
            raise ValueError(f"Unsupported source_type for download: {source_type}")

    def preprocess(self, video_sources: List[str]) -> List[Tuple[str, str, List[str], float]]:
        """
        Args:
            video_sources: List of video file paths or URIs

        Returns:
            List of (video_id, video_path, frames_b64_or_refs, preprocessing_time_seconds)
        """
        if self.run_id is None:
            logger.warning("run_id not set for FrameSamplingPreprocessor, using 'default'")
            self.run_id = "default"

        # Memory mode: process inline (backward compatible)
        if self.storage_backend is None:
            return self._preprocess_memory_mode(video_sources)

        # Storage mode: batch operations (all saves, then all loads)
        return self._preprocess_storage_mode(video_sources)

    def _preprocess_memory_mode(self, video_sources: List[str]) -> List[Tuple[str, str, List[str], float]]:
        source_type = self.config.get("video_source", {}).get("type")
        source_mode = self.config.get("video_source", {}).get("mode")
        needs_download = (
            source_type in [VideoSourceType.S3.value, VideoSourceType.GCS.value]
            and source_mode == VideoSourceMode.DOWNLOAD_LOCAL.value
        )

        video_data = []
        tracker = FrameSamplingTracker(self.x_frames)

        for video_source in video_sources:
            video_id = extract_video_id(video_source)
            try:
                start_time = time.time()

                if needs_download:
                    local_video_path = self._download(video_source, source_type)
                else:
                    local_video_path = video_source

                _, frames_b64 = sample_frames_base64(local_video_path, x=self.x_frames)
                tracker.record(len(frames_b64))
                preprocessing_time = time.time() - start_time
                video_data.append((video_id, video_source, frames_b64, preprocessing_time))
            except Exception as e:
                logger.error(f"Failed to download/sample frames for {video_id}: {e}")
                tracker.record(0)
                video_data.append((video_id, video_source, [], 0.0))

        tracker.log_summary()
        return video_data

    def _preprocess_storage_mode(self, video_sources: List[str]) -> List[Tuple[str, str, List[str], float]]:
        video_metadata = []  # Store (video_id, video_path, sample_save_time, needs_load, actual_run_id)

        # Determine which run_id to use for storage
        use_cache = False
        storage_run_id = self.run_id

        if self.cache_run_id is not None:
            logger.info(
                f"Checking cache validity for {len(video_sources)} videos (cache_run_id: {self.cache_run_id})..."
            )
            all_cached = True
            for video_path in video_sources:
                video_id = extract_video_id(video_path)
                if not self.storage_backend.exists(video_id, self.cache_run_id, expected_count=self.x_frames):
                    all_cached = False
                    break

            if all_cached:
                logger.info(f"All {len(video_sources)} videos found in cache - using cache_run_id: {self.cache_run_id}")
                use_cache = True
                storage_run_id = self.cache_run_id
            else:
                logger.info(f"Cache miss - will sample all videos with run_id: {self.run_id}")
                storage_run_id = self.run_id
        else:
            logger.info(f"No cache_run_id provided - sampling all videos with run_id: {self.run_id}")

        # Phase 1: Load from cache OR download + sample and save
        if use_cache:
            for video_source in video_sources:
                video_id = extract_video_id(video_source)
                video_metadata.append((video_id, video_source, 0.0, True, storage_run_id))
        else:
            source_type = self.config.get("video_source", {}).get("type")
            source_mode = self.config.get("video_source", {}).get("mode")
            needs_download = (
                source_type in [VideoSourceType.S3.value, VideoSourceType.GCS.value]
                and source_mode == VideoSourceMode.DOWNLOAD_LOCAL.value
            )

            if needs_download:
                source_name = "S3" if source_type == VideoSourceType.S3.value else "GCS"
                logger.info(f"Downloading {len(video_sources)} videos from {source_name}...")

            storage_type = (
                self.config.get("pipeline", {})
                .get("preprocessor", {})
                .get("config", {})
                .get("storage", {})
                .get("type", "local")
            )
            logger.info(
                f"Sampling {self.x_frames} frames from {len(video_sources)} videos. "
                f"Saving results to {storage_type.upper()}."
            )

            for video_source in video_sources:
                video_id = extract_video_id(video_source)
                try:
                    start_time = time.time()

                    if needs_download:
                        local_video_path = self._download(video_source, source_type)
                    else:
                        local_video_path = video_source

                    _, frames_np = sample_frames_numpy(local_video_path, x=self.x_frames)
                    self.storage_backend.save_frames(video_id, frames_np, storage_run_id)
                    phase1_time = time.time() - start_time
                    video_metadata.append((video_id, video_source, phase1_time, True, storage_run_id))
                except Exception as e:
                    logger.error(f"Failed to download/sample/save frames for {video_id}: {e}")
                    video_metadata.append((video_id, video_source, 0.0, False, storage_run_id))

        # Phase 2: Load all frames
        load_format = "bytes" if self.output_format == "bytes" else "base64"
        logger.info(f"Phase 2: Loading frames as {load_format} for {len(video_metadata)} videos")
        video_data = []
        tracker = FrameSamplingTracker(self.x_frames)

        for video_id, video_path, phase1_time, needs_load, actual_run_id in video_metadata:
            if not needs_load:
                tracker.record(0)
                video_data.append((video_id, video_path, [], 0.0))
                continue

            try:
                start_time = time.time()
                if self.output_format == "bytes":
                    frames_data = self.storage_backend.load_frames_bytes(video_id, actual_run_id)
                else:
                    frames_data = self.storage_backend.load_frames_base64(video_id, actual_run_id)
                tracker.record(len(frames_data))
                phase2_time = time.time() - start_time
                total_preprocessing_time = phase1_time + phase2_time
                video_data.append((video_id, video_path, frames_data, total_preprocessing_time))
            except Exception as e:
                logger.error(f"Failed to load frames for {video_id}: {e}")
                tracker.record(0)
                video_data.append((video_id, video_path, [], phase1_time))

        tracker.log_summary()
        return video_data
