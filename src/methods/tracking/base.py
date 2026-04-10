from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.methods.detection import Detection


@dataclass(frozen=True)
class Track:
    """A single tracked object state for one frame."""

    track_id: int
    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Tracker(ABC):
    """Common tracker contract for frame-wise multi-object trackers."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
    ) -> list[Track]:
        """Update tracker state from detections and return active tracks."""
