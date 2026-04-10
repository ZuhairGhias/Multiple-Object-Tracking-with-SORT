from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import Detection, Detector


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOT17_DIR = PROJECT_ROOT / "data" / "MOT17"
DEFAULT_MIN_SCORE: float | None = None


@dataclass
class MOT17Detector(Detector):
    """Detector backed by MOT17 det.txt files for a chosen sequence."""

    sequence_id: str
    root_dir: str | Path | None = None
    min_score: float | None = field(default=DEFAULT_MIN_SCORE, init=False)
    _detections_by_frame: dict[int, list[Detection]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _sequence_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sequence_path = self._resolve_sequence_path()
        self._detections_by_frame = self._load_detections()

    @property
    def sequence_path(self) -> Path:
        return self._sequence_path

    @property
    def detections_path(self) -> Path:
        return self._sequence_path / "det" / "det.txt"

    def get_detections(
        self,
        frame=None,
        *,
        frame_index: int | None = None,
    ) -> list[Detection]:
        del frame
        if frame_index is None:
            raise ValueError("MOT17Detector requires frame_index for detection lookup.")
        return list(self._detections_by_frame.get(frame_index, []))

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
            "Could not resolve MOT17 sequence "
            f"'{self.sequence_id}' in '{DEFAULT_MOT17_DIR}'."
        )

    def _load_detections(self) -> dict[int, list[Detection]]:
        detections_path = self.detections_path
        if not detections_path.is_file():
            raise FileNotFoundError(
                f"Could not find MOT17 detections file at '{detections_path}'."
            )

        detections_by_frame: dict[int, list[Detection]] = {}
        with detections_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 7:
                    raise ValueError(
                        f"Invalid MOT17 detection row on line {line_number}: '{line}'."
                    )

                frame_index = int(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                width = float(parts[4])
                height = float(parts[5])
                score = float(parts[6])

                if self.min_score is not None and score < self.min_score:
                    continue

                detection = Detection(
                    frame_index=frame_index,
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    score=score,
                )
                detections_by_frame.setdefault(frame_index, []).append(detection)

        return detections_by_frame
