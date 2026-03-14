"""Custom exceptions for the benchmarks framework."""


class BenchmarkError(Exception):
    """Base exception for all benchmark-related errors."""

    pass


class FatalError(BenchmarkError):
    """
    Fatal error that should halt pipeline execution immediately.

    Raised for systemic issues where all videos will fail:
    - Invalid/expired/missing credentials
    - Invalid model ID or configuration
    - Missing permissions to access models/resources
    - Model not found or not available in region

    Pipeline should catch this and halt execution, not treat as per-video failure.
    """

    pass
