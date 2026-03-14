from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from ..metrics.classification import save_metrics_to_json
from .base import ResultSaver


class JSONResultSaver(ResultSaver):
    """Saves benchmark results to JSON files with timestamp suffix."""

    def __init__(self, include_timestamp: bool = True):
        """
        Args:
            include_timestamp: Whether to include timestamp in filename
        """
        self.include_timestamp = include_timestamp
        self.timestamp = None  # Can be set externally to ensure consistency

    def save(self, results: Dict[str, Any], model_name: str, output_dir: Path, timestamp: str = None) -> Path:
        """Save benchmark results to JSON file with timestamp."""
        # Sanitize model name for filename
        model_short_name = model_name.replace("us.amazon.", "").replace(":", "_").replace(".", "_")

        # Build filename: {model_name}_{timestamp}.json (no "metrics_" prefix)
        if self.include_timestamp:
            # Use provided timestamp or create new one
            ts = timestamp if timestamp else datetime.now().strftime("%y%m%d_%H%M%S")
            filename = f"{model_short_name}_{ts}.json"
        else:
            filename = f"{model_short_name}.json"

        filepath = output_dir / filename
        save_metrics_to_json(results, str(filepath))
        logger.info(f"   Results saved to: {filepath}")

        return filepath
