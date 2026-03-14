import base64
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger

from .base import FrameStorage


class LocalFrameStorage(FrameStorage):
    def __init__(self, base_path: str = "./data/cache/frames"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized LocalFrameStorage at {self.base_path}")

    def _get_video_dir(self, run_id: str, video_id: str) -> Path:
        return self.base_path / run_id / video_id

    def _get_frame_path(self, run_id: str, video_id: str, frame_idx: int) -> Path:
        return self._get_video_dir(run_id, video_id) / f"frame_{frame_idx:03d}.jpg"

    def save_frames(self, video_id: str, frames: List[np.ndarray], run_id: str) -> List[str]:
        video_dir = self._get_video_dir(run_id, video_id)
        video_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for idx, frame in enumerate(frames):
            frame_path = self._get_frame_path(run_id, video_id, idx)
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

        return frame_paths

    def load_frames_base64(self, video_id: str, run_id: str) -> List[str]:
        frame_paths = self.list_frames(video_id, run_id)
        base64_frames = []

        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
                b64 = base64.b64encode(frame_bytes).decode("utf-8")
                base64_frames.append(b64)

        return base64_frames

    def load_frames_bytes(self, video_id: str, run_id: str) -> List[bytes]:
        frame_paths = self.list_frames(video_id, run_id)
        frames_bytes = []

        for frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
                frames_bytes.append(frame_bytes)

        return frames_bytes

    def exists(self, video_id: str, run_id: str, expected_count: int = None) -> bool:
        video_dir = self._get_video_dir(run_id, video_id)
        if not video_dir.exists():
            logger.debug(f"No frames found at {video_dir}")
            return False

        if expected_count is not None:
            frame_files = sorted(video_dir.glob("frame_*.jpg"))
            actual_count = len(frame_files)
            if actual_count != expected_count:
                logger.debug(f"Frame count mismatch for {video_id}: expected {expected_count}, found {actual_count}")
                return False

        return True

    def list_frames(self, video_id: str, run_id: str) -> List[str]:
        video_dir = self._get_video_dir(run_id, video_id)
        if not video_dir.exists():
            return []

        frame_files = sorted(video_dir.glob("frame_*.jpg"))
        return [str(f) for f in frame_files]
