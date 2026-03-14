from abc import ABC, abstractmethod
from typing import Any, Dict


class ResultWriter(ABC):
    """Abstract base class for writing pipeline results."""

    @abstractmethod
    def write(self, result: Dict[str, Any]) -> None:
        """
        Write a single result record.

        Args:
            result: Dictionary containing result data (video_id, tags, scores, etc.)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
