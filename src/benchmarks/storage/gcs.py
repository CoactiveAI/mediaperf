import base64
from typing import List, Optional

import cv2
import numpy as np
from google.api_core.exceptions import Forbidden, Unauthenticated
from google.auth.exceptions import DefaultCredentialsError
from loguru import logger

from ..exceptions.credential_errors import handle_credential_error
from ..utils.gcp import get_gcs_client
from .base import FrameStorage


class GCSFrameStorage(FrameStorage):
    def __init__(
        self,
        bucket: str,
        prefix: str = "frames",
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        self.bucket_name = bucket
        self.prefix = prefix.rstrip("/")
        self.gcs_client = get_gcs_client(project_id=project_id, credentials_path=credentials_path)
        self.bucket = self.gcs_client.bucket(bucket)
        logger.info(f"Initialized GCSFrameStorage at gs://{bucket}/{prefix}")

    def _get_blob_name(self, run_id: str, video_id: str, frame_idx: int = None) -> str:
        if frame_idx is not None:
            return f"{self.prefix}/{run_id}/{video_id}/frame_{frame_idx:03d}.jpg"
        return f"{self.prefix}/{run_id}/{video_id}/"

    def save_frames(self, video_id: str, frames: List[np.ndarray], run_id: str) -> List[str]:
        gcs_uris = []

        for idx, frame in enumerate(frames):
            blob_name = self._get_blob_name(run_id, video_id, idx)
            blob = self.bucket.blob(blob_name)

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            blob.upload_from_string(frame_bytes, content_type="image/jpeg")

            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            gcs_uris.append(gcs_uri)

        return gcs_uris

    def load_frames_base64(self, video_id: str, run_id: str) -> List[str]:
        frame_uris = self.list_frames(video_id, run_id)
        base64_frames = []

        for uri in frame_uris:
            blob_name = uri.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(blob_name)
            frame_bytes = blob.download_as_bytes()
            b64 = base64.b64encode(frame_bytes).decode("utf-8")
            base64_frames.append(b64)

        return base64_frames

    def load_frames_bytes(self, video_id: str, run_id: str) -> List[bytes]:
        frame_uris = self.list_frames(video_id, run_id)
        frames_bytes = []

        for uri in frame_uris:
            blob_name = uri.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(blob_name)
            frame_bytes = blob.download_as_bytes()
            frames_bytes.append(frame_bytes)

        return frames_bytes

    def exists(self, video_id: str, run_id: str, expected_count: int = None) -> bool:
        prefix = self._get_blob_name(run_id, video_id)

        try:
            blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=1))
            if not blobs:
                logger.debug(f"No frames found at gs://{self.bucket_name}/{prefix}")
                return False

            if expected_count is not None:
                frames = self.list_frames(video_id, run_id)
                actual_count = len(frames)
                if actual_count != expected_count:
                    logger.debug(
                        f"Frame count mismatch for {video_id}: expected {expected_count}, found {actual_count}"
                    )
                    return False

            return True
        except (DefaultCredentialsError, Forbidden, Unauthenticated) as e:
            handle_credential_error(e, context="checking GCS cache")
        except Exception as e:
            logger.error(f"Error checking existence in GCS: {e}")
            raise

    def list_frames(self, video_id: str, run_id: str) -> List[str]:
        prefix = self._get_blob_name(run_id, video_id)
        frame_uris = []

        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.name.endswith(".jpg"):
                    uri = f"gs://{self.bucket_name}/{blob.name}"
                    frame_uris.append(uri)

            frame_uris.sort()
            return frame_uris
        except (DefaultCredentialsError, Forbidden, Unauthenticated) as e:
            handle_credential_error(e, context="listing frames from GCS")
        except Exception as e:
            logger.error(f"Error listing frames from GCS: {e}")
            raise
