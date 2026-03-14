import base64
from typing import List, Optional

import cv2
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger

from ..exceptions.aws_errors import AWS_CREDENTIAL_ERROR_CODES
from ..exceptions.credential_errors import handle_credential_error
from ..utils.aws import get_s3_client
from .base import FrameStorage


class S3FrameStorage(FrameStorage):
    def __init__(
        self,
        bucket: str,
        prefix: str = "frames",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.s3_client = get_s3_client(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        logger.info(f"Initialized S3FrameStorage at s3://{bucket}/{prefix}")

    def _get_s3_key(self, run_id: str, video_id: str, frame_idx: int = None) -> str:
        if frame_idx is not None:
            return f"{self.prefix}/{run_id}/{video_id}/frame_{frame_idx:03d}.jpg"
        return f"{self.prefix}/{run_id}/{video_id}/"

    def save_frames(self, video_id: str, frames: List[np.ndarray], run_id: str) -> List[str]:
        s3_uris = []

        for idx, frame in enumerate(frames):
            s3_key = self._get_s3_key(run_id, video_id, idx)

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=frame_bytes,
                ContentType="image/jpeg",
            )

            s3_uri = f"s3://{self.bucket}/{s3_key}"
            s3_uris.append(s3_uri)

        return s3_uris

    def load_frames_base64(self, video_id: str, run_id: str) -> List[str]:
        frame_uris = self.list_frames(video_id, run_id)
        base64_frames = []

        for uri in frame_uris:
            s3_key = uri.replace(f"s3://{self.bucket}/", "")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            frame_bytes = response["Body"].read()
            b64 = base64.b64encode(frame_bytes).decode("utf-8")
            base64_frames.append(b64)

        return base64_frames

    def load_frames_bytes(self, video_id: str, run_id: str) -> List[bytes]:
        frame_uris = self.list_frames(video_id, run_id)
        frames_bytes = []

        for uri in frame_uris:
            s3_key = uri.replace(f"s3://{self.bucket}/", "")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            frame_bytes = response["Body"].read()
            frames_bytes.append(frame_bytes)

        return frames_bytes

    def exists(self, video_id: str, run_id: str, expected_count: int = None) -> bool:
        prefix = self._get_s3_key(run_id, video_id)

        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
            if "Contents" not in response:
                logger.debug(f"No frames found at s3://{self.bucket}/{prefix}")
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
        except NoCredentialsError as e:
            handle_credential_error(e, context="checking S3 cache")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in AWS_CREDENTIAL_ERROR_CODES:
                handle_credential_error(e, context="checking S3 cache")
            logger.error(f"Error checking existence in S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Error checking existence in S3: {e}")
            raise

    def list_frames(self, video_id: str, run_id: str) -> List[str]:
        prefix = self._get_s3_key(run_id, video_id)
        frame_uris = []

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".jpg"):
                        uri = f"s3://{self.bucket}/{obj['Key']}"
                        frame_uris.append(uri)

            frame_uris.sort()
            return frame_uris
        except NoCredentialsError as e:
            handle_credential_error(e, context="listing frames from S3")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in AWS_CREDENTIAL_ERROR_CODES:
                handle_credential_error(e, context="listing frames from S3")
            logger.error(f"Error listing frames from S3: {e}")
            raise
        except Exception as e:
            logger.error(f"Error listing frames from S3: {e}")
            raise
