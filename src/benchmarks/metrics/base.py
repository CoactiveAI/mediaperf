from abc import ABC, abstractmethod
from typing import Any, Dict


class MetricCalculator(ABC):
    """Abstract base class for calculating evaluation metrics."""

    @abstractmethod
    def calculate(
        self,
        predictions_path: str,
        ground_truth_path: str,
    ) -> Dict[str, Any]:
        """
        Calculate metrics by comparing predictions to ground truth.

        Args:
            predictions_path: Path to predictions file
            ground_truth_path: Path to ground truth file

        Returns:
            Dictionary containing calculated metrics
        """
        pass
