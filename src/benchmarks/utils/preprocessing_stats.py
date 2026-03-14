from statistics import median

from loguru import logger


class FrameSamplingTracker:
    """Track frame sampling statistics across videos."""

    def __init__(self, expected_frames: int):
        self.expected_frames = expected_frames
        self.frame_counts = []

    def record(self, frames_obtained: int):
        """Record frames obtained for a single video."""
        self.frame_counts.append(frames_obtained)

    def log_summary(self):
        """Log summary statistics."""
        if not self.frame_counts:
            return

        total_videos = len(self.frame_counts)
        incomplete_videos = sum(1 for count in self.frame_counts if count < self.expected_frames)
        incomplete_pct = (incomplete_videos / total_videos) * 100

        avg_frames = sum(self.frame_counts) / total_videos
        min_frames = min(self.frame_counts)
        median_frames = median(self.frame_counts)

        logger.info(
            f"Frame sampling summary: {total_videos} videos processed, "
            f"{incomplete_videos} ({incomplete_pct:.1f}%) had fewer frames than requested, "
            f"min: {min_frames}/{self.expected_frames}, "
            f"median: {median_frames:.1f}/{self.expected_frames}, "
            f"avg: {avg_frames:.1f}/{self.expected_frames} frames per video"
        )
