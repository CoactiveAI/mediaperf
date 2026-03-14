"""Video preprocessing utilities."""

import base64
import os
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

# Suppress ffmpeg warnings (like "mmco: unref short failure")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


def uniform_frame_indices(num_frames: int, x: int) -> List[int]:
    """
    Evenly sample x frame indices over [0, num_frames-1].
    Guarantees x unique, sorted integer indices (handles short videos too).
    """
    if num_frames <= 0 or x <= 0:
        return []
    if x >= num_frames:
        return list(range(num_frames))
    idxs = np.linspace(0, num_frames - 1, x)
    return sorted({int(round(i)) for i in idxs})


def sample_frames_numpy(video_path: str, x: int = 16) -> Tuple[List[int], List[np.ndarray]]:
    """
    Args:
        video_path: Path to the video file.
        x: Number of frames to sample uniformly.

    Returns:
        (frame_indices, frames)
    """
    cv2.setLogLevel(0)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = uniform_frame_indices(num_frames, x)
    frames: List[np.ndarray] = []

    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        success, frame = cap.read()
        if not success:
            continue
        frames.append(frame)

    cap.release()

    if len(frames) < len(indices):
        logger.warning(f"Extracted fewer frames than requested from {video_path}: got {len(frames)}/{len(indices)}")

    return indices, frames


def sample_frames_base64(video_path: str, x: int = 16) -> Tuple[List[int], List[str]]:
    """
    Samples x frames uniformly from a video and returns them as base64-encoded JPEG strings.

    Args:
        video_path: Path to the video file.
        x: Number of frames to sample uniformly.

    Returns:
        (frame_indices, base64_frames)
        - frame_indices: List of sampled frame indices.
        - base64_frames: List of base64-encoded JPEG strings (no data URL prefix).
    """
    cv2.setLogLevel(0)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = uniform_frame_indices(num_frames, x)
    base64_frames: List[str] = []

    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        success, frame = cap.read()
        if not success:
            continue
        # Encode frame as JPEG and convert to Base64
        _, buffer = cv2.imencode(".jpg", frame)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        base64_frames.append(base64_str)

    cap.release()

    if len(base64_frames) < len(indices):
        logger.warning(
            f"Extracted fewer frames than requested from {video_path}: got {len(base64_frames)}/{len(indices)}"
        )

    return indices, base64_frames
