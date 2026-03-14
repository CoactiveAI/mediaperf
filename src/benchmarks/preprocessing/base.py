from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class VideoPreprocessor(ABC):
    """Abstract base class for video preprocessing."""

    @abstractmethod
    def preprocess(self, video_sources: List[str]) -> List[Tuple[str, str, Any]]:
        """
        Preprocess videos for model input.

        Args:
            video_sources: List of video identifiers (paths, IDs, URIs, etc.)

        Returns:
            List of tuples (video_id, source_path, preprocessed_data)
            - video_id: Video identifier
            - source_path: Original source for logging
            - preprocessed_data: Model-ready input (frames, URI, etc.)
        """
        pass
