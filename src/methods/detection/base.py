from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """A single detector output for one frame."""

    frame_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    class_id: int | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


class Detector(ABC):
    """Common detector contract for frame-wise detection providers."""

    @abstractmethod
    def get_detections(
        self,
        frame: Any | None = None,
        *,
        frame_index: int | None = None,
    ) -> list[Detection]:
        """Return detections for a frame image and/or its frame index."""
