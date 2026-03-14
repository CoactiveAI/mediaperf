from typing import Any, Dict

import jsonlines

from .base import ResultWriter


class JSONLResultWriter(ResultWriter):
    """Writes results to a JSONL file."""

    def __init__(self, output_path: str):
        """
        Initialize JSONL writer.

        Args:
            output_path: Path to output JSONL file
        """
        self.output_path = output_path
        self.writer = jsonlines.open(output_path, mode="w")

    def write(self, result: Dict[str, Any]) -> None:
        """
        Write a single result record to JSONL file.

        Args:
            result: Dictionary containing result data
        """
        self.writer.write(result)

    def close(self) -> None:
        """Close the JSONL writer."""
        if self.writer:
            self.writer.close()
