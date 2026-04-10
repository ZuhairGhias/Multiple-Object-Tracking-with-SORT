from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import Track, Tracker


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOT17_DIR = PROJECT_ROOT / "data" / "MOT17"


@dataclass
class MOTGroundTruthTracker(Tracker):
    """Tracker-like wrapper that replays MOT ground-truth IDs and boxes."""

    sequence_id: str
    root_dir: str | Path | None = None
    _tracks_by_frame: dict[int, list[Track]] = field(default_factory=dict, init=False, repr=False)
    _sequence_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sequence_path = self._resolve_sequence_path()
        self._tracks_by_frame = self._load_tracks()

    @property
    def sequence_path(self) -> Path:
        return self._sequence_path

    @property
    def ground_truth_path(self) -> Path:
        return self._sequence_path / "gt" / "gt.txt"

    def update(self, detections, *, frame_index: int) -> list[Track]:
        del detections
        return list(self._tracks_by_frame.get(frame_index, []))

    def _resolve_sequence_path(self) -> Path:
        if self.root_dir is not None:
            base_dir = Path(self.root_dir)
            direct_match = base_dir / self.sequence_id
            if direct_match.is_dir():
                return direct_match
            raise FileNotFoundError(
                f"Could not find MOT17 sequence '{self.sequence_id}' under '{base_dir}'."
            )

        candidate_paths = [
            DEFAULT_MOT17_DIR / "train" / self.sequence_id,
            DEFAULT_MOT17_DIR / "test" / self.sequence_id,
        ]
        for candidate_path in candidate_paths:
            if candidate_path.is_dir():
                return candidate_path

        raise FileNotFoundError(
            f"Could not resolve MOT17 ground-truth sequence '{self.sequence_id}' in '{DEFAULT_MOT17_DIR}'."
        )

    def _load_tracks(self) -> dict[int, list[Track]]:
        ground_truth_path = self.ground_truth_path
        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                f"Could not find MOT17 ground-truth file at '{ground_truth_path}'."
            )

        tracks_by_frame: dict[int, list[Track]] = {}
        with ground_truth_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 6:
                    raise ValueError(
                        f"Invalid MOT17 ground-truth row on line {line_number}: '{line}'."
                    )

                frame_index = int(parts[0])
                track_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                score = float(parts[6]) if len(parts) > 6 else None

                track = Track(
                    track_id=track_id,
                    frame_index=frame_index,
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    score=score,
                )
                tracks_by_frame.setdefault(frame_index, []).append(track)

        return tracks_by_frame
