from .base import FrameStorage
from .gcs import GCSFrameStorage
from .local import LocalFrameStorage
from .s3 import S3FrameStorage

__all__ = ["FrameStorage", "LocalFrameStorage", "S3FrameStorage", "GCSFrameStorage"]
