from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlaceholderTracker:
    """Minimal tracker scaffold until SORT logic is implemented."""

    max_age: int = 10
    min_hits: int = 3
    iou_threshold: float = 0.3
    _initialized: bool = field(default=True, init=False, repr=False)

    def describe(self) -> str:
        return (
            "PlaceholderTracker("
            f"max_age={self.max_age}, "
            f"min_hits={self.min_hits}, "
            f"iou_threshold={self.iou_threshold})"
        )

    def process_video(self, video_path: str | None) -> str:
        if not video_path:
            return "No video provided. Upload a file to exercise the scaffold."
        return (
            f"Received video at '{video_path}'. "
            "Frame processing and SORT association are not implemented yet."
        )
