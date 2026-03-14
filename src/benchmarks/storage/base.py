from abc import ABC, abstractmethod
from typing import List

import numpy as np


class FrameStorage(ABC):
    @abstractmethod
    def save_frames(self, video_id: str, frames: List[np.ndarray], run_id: str) -> List[str]:
        """
        Args:
            video_id: Unique identifier for the video
            frames: List of frames as numpy arrays
            run_id: Run identifier for organizing frames

        Returns:
            List of frame references (paths or URIs)
        """
        pass

    @abstractmethod
    def load_frames_base64(self, video_id: str, run_id: str) -> List[str]:
        """
        Args:
            video_id: Unique identifier for the video
            run_id: Run identifier for organizing frames

        Returns:
            List of base64-encoded frame strings
        """
        pass

    @abstractmethod
    def load_frames_bytes(self, video_id: str, run_id: str) -> List[bytes]:
        """
        Args:
            video_id: Unique identifier for the video
            run_id: Run identifier for organizing frames

        Returns:
            List of raw JPEG frame bytes
        """
        pass

    @abstractmethod
    def exists(self, video_id: str, run_id: str, expected_count: int = None) -> bool:
        """
        Args:
            video_id: Unique identifier for the video
            run_id: Run identifier for organizing frames
            expected_count: Expected number of frames (optional validation)

        Returns:
            True if frames exist for this video_id and run_id
        """
        pass

    @abstractmethod
    def list_frames(self, video_id: str, run_id: str) -> List[str]:
        """
        Args:
            video_id: Unique identifier for the video
            run_id: Run identifier for organizing frames

        Returns:
            List of frame references (paths or URIs)
        """
        pass
