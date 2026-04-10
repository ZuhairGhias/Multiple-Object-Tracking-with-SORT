"""Detection interfaces and shared detection types."""

from .base import Detection, Detector
from .mot import MOT17Detector

__all__ = ["Detection", "Detector", "MOT17Detector"]
