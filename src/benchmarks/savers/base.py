from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class ResultSaver(ABC):
    """Abstract base class for saving benchmark results."""

    @abstractmethod
    def save(self, results: Dict[str, Any], model_name: str, output_dir: Path) -> Path:
        """
        Save benchmark results.

        Args:
            results: Dictionary containing benchmark metrics
            model_name: Name of the model being benchmarked
            output_dir: Directory to save results to

        Returns:
            Path to the saved file
        """
        pass
