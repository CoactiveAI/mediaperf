from typing import Any, Dict, List

from .base import MetricCalculator
from .classification import calculate_multilabel_metrics


class MultilabelClassificationMetric(MetricCalculator):
    """Metric calculator for multi-label classification tasks."""

    def __init__(
        self,
        pred_key: str = "video_level_tags",
        gt_key: str = "tags",
        exclude_tags: List[str] = None,
        include_tags: List[str] = None,
        exclude_failures: bool = True,
    ):
        """
        Initialize multi-label classification metric calculator.

        Args:
            pred_key: Key name for predicted tags in predictions file
            gt_key: Key name for ground truth tags in ground truth file
            exclude_tags: List of tags to exclude from metric calculation
            include_tags: List of tags to include in metric calculation (if specified, only these tags are evaluated)
            exclude_failures: If True, exclude videos with "error" field from metrics (default: True)
        """
        self.pred_key = pred_key
        self.gt_key = gt_key
        self.exclude_tags = exclude_tags or []
        self.include_tags = include_tags or []
        self.exclude_failures = exclude_failures

    def calculate(
        self,
        predictions_path: str,
        ground_truth_path: str,
    ) -> Dict[str, Any]:
        """
        Calculate multi-label classification metrics.

        Args:
            predictions_path: Path to predictions JSONL file
            ground_truth_path: Path to ground truth JSONL file

        Returns:
            Dictionary containing calculated metrics
        """
        return calculate_multilabel_metrics(
            predictions_jsonl=predictions_path,
            ground_truth_jsonl=ground_truth_path,
            pred_key=self.pred_key,
            gt_key=self.gt_key,
            exclude_tags=self.exclude_tags,
            include_tags=self.include_tags,
            exclude_failures=self.exclude_failures,
        )
