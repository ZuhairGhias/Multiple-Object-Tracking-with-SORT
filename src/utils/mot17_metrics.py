"""Batch MOT17 scoring and CSV report persistence."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.metrics import MOTMetrics, compare_mot_metrics, evaluate_mot_metrics
from src.methods.detection import MOT17Detector
from src.methods.tracking import MOTGroundTruthTracker, NaiveIOUTracker, SORT, Track, Tracker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_ROOT = PROJECT_ROOT / "data" / "MOT17" / "train"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
DEFAULT_METRICS_CSV = DEFAULT_METRICS_DIR / "MOT17_tracking_metrics.csv"

TRACKER_BUILDERS: dict[str, Callable[[], Tracker]] = {
    "naive_iou": NaiveIOUTracker,
    "sort": SORT,
}
TRACKER_LABELS = {
    "naive_iou": "Naive IoU",
    "sort": "SORT",
}
METADATA_COLUMNS = ("Frames", "GroundTruthCount", "PredictionCount", "Matches")
METRIC_COLUMNS = ("MOTA", "MOTP", "IDF1", "FAF", "MT", "ML", "FP", "FN", "IDSW", "Frag")
CSV_COLUMNS = (
    "Scope",
    "Example",
    "Sequence",
    "Detector",
    "Tracker",
    *METADATA_COLUMNS,
    *METRIC_COLUMNS,
)
@dataclass(frozen=True)
class SequenceScore:
    """Tracks and computed metrics for one MOT17 sequence/detector pair."""

    example: str
    sequence_id: str
    detector: str
    frame_count: int
    metrics_by_tracker: dict[str, MOTMetrics]
    ground_truth_tracks: list[Track]
    predictions_by_tracker: dict[str, list[Track]]


@dataclass(frozen=True)
class MOT17MetricsOutputs:
    """Files and summary counts produced by the batch metrics command."""

    csv_path: Path
    sequence_count: int


@dataclass(frozen=True)
class MOT17MetricsRow:
    """One persisted metrics row from the MOT17 batch report.

    The same row class is used for CSV writing and reading so downstream
    reporting utilities do not reimplement the report schema.
    """

    scope: str
    example: str
    sequence: str
    detector: str
    tracker: str
    frames: int
    ground_truth_count: int
    prediction_count: int
    matches: int
    mota: float
    motp: float
    idf1: float
    faf: float
    mostly_tracked: int
    mostly_lost: int
    false_positives: int
    false_negatives: int
    id_switches: int
    fragmentations: int

    def to_csv_row(self) -> dict[str, str | float | int]:
        """Serialize this row using the public CSV column names."""

        return {
            "Scope": self.scope,
            "Example": self.example,
            "Sequence": self.sequence,
            "Detector": self.detector,
            "Tracker": self.tracker,
            "Frames": self.frames,
            "GroundTruthCount": self.ground_truth_count,
            "PredictionCount": self.prediction_count,
            "Matches": self.matches,
            "MOTA": self.mota,
            "MOTP": self.motp,
            "IDF1": self.idf1,
            "FAF": self.faf,
            "MT": self.mostly_tracked,
            "ML": self.mostly_lost,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "IDSW": self.id_switches,
            "Frag": self.fragmentations,
        }

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "MOT17MetricsRow":
        """Parse one CSV row produced by :meth:`to_csv_row`."""

        return cls(
            scope=row["Scope"],
            example=row["Example"],
            sequence=row["Sequence"],
            detector=row["Detector"],
            tracker=row["Tracker"],
            frames=int(row["Frames"]),
            ground_truth_count=int(row["GroundTruthCount"]),
            prediction_count=int(row["PredictionCount"]),
            matches=int(row["Matches"]),
            mota=float(row["MOTA"]),
            motp=float(row["MOTP"]),
            idf1=float(row["IDF1"]),
            faf=float(row["FAF"]),
            mostly_tracked=int(row["MT"]),
            mostly_lost=int(row["ML"]),
            false_positives=int(row["FP"]),
            false_negatives=int(row["FN"]),
            id_switches=int(row["IDSW"]),
            fragmentations=int(row["Frag"]),
        )

    def metric_value(self, metric_name: str) -> float | int:
        """Return a metric by its report column name."""

        return {
            "MOTA": self.mota,
            "MOTP": self.motp,
            "IDF1": self.idf1,
            "FAF": self.faf,
            "MT": self.mostly_tracked,
            "ML": self.mostly_lost,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "IDSW": self.id_switches,
            "Frag": self.fragmentations,
        }[metric_name]


@dataclass
class AggregateTracks:
    """Pooled tracks and frame count for one aggregate report group."""

    ground_truth: list[Track] = field(default_factory=list)
    predictions: list[Track] = field(default_factory=list)
    frame_count: int = 0


def generate_mot17_metrics_report(
    sequence_root: str | Path = DEFAULT_SEQUENCE_ROOT,
) -> MOT17MetricsOutputs:
    """Score every GT-backed MOT17 sequence and write the canonical CSV report.

    The local MOT17 test split does not contain `gt/gt.txt`, so the default
    root points at `data/MOT17/train`. Each detector-specific sequence is
    scored independently, then aggregate rows are recomputed from pooled raw
    tracks instead of averaging per-sequence percentages.
    """

    sequence_dirs = discover_scored_sequence_dirs(sequence_root)
    sequence_scores = []
    for sequence_index, sequence_dir in enumerate(sequence_dirs, start=1):
        print(f"[{sequence_index}/{len(sequence_dirs)}] Scoring {sequence_dir.name}")
        sequence_scores.append(score_sequence(sequence_dir))
    aggregate_rows = build_aggregate_rows(sequence_scores)
    per_sequence_rows = build_per_sequence_rows(sequence_scores)
    rows = per_sequence_rows + aggregate_rows

    DEFAULT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(DEFAULT_METRICS_CSV, rows)
    return MOT17MetricsOutputs(
        csv_path=DEFAULT_METRICS_CSV,
        sequence_count=len(sequence_scores),
    )


def discover_scored_sequence_dirs(sequence_root: str | Path) -> list[Path]:
    """Find MOT17 sequence directories that contain detections and GT."""

    root = Path(sequence_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Could not find MOT17 sequence root '{root}'.")

    sequence_dirs = [
        sequence_dir
        for sequence_dir in sorted(root.iterdir())
        if sequence_dir.is_dir()
        and (sequence_dir / "seqinfo.ini").is_file()
        and (sequence_dir / "det" / "det.txt").is_file()
        and (sequence_dir / "gt" / "gt.txt").is_file()
    ]
    if not sequence_dirs:
        raise FileNotFoundError(f"No GT-backed MOT17 sequences found under '{root}'.")
    return sequence_dirs


def score_sequence(sequence_dir: Path) -> SequenceScore:
    """Run the configured trackers over one MOT17 sequence and score them."""

    sequence_id = sequence_dir.name
    example, detector_name = split_sequence_id(sequence_id)
    frame_count = infer_sequence_length(sequence_dir)
    detector = MOT17Detector(sequence_id=sequence_id, root_dir=sequence_dir.parent)
    ground_truth_tracker = MOTGroundTruthTracker(
        sequence_id=sequence_id,
        root_dir=sequence_dir.parent,
    )
    trackers = {
        tracker_name: builder()
        for tracker_name, builder in TRACKER_BUILDERS.items()
    }
    ground_truth_tracks: list[Track] = []
    predictions_by_tracker: dict[str, list[Track]] = {
        tracker_name: []
        for tracker_name in trackers
    }

    for frame_index in range(1, frame_count + 1):
        detections = detector.get_detections(frame_index=frame_index)
        ground_truth_tracks.extend(
            ground_truth_tracker.update(detections, frame_index=frame_index)
        )
        for tracker_name, tracker in trackers.items():
            predictions_by_tracker[tracker_name].extend(
                tracker.update(detections, frame_index=frame_index)
            )

    metrics_by_tracker = compare_mot_metrics(
        ground_truth_tracks,
        predictions_by_tracker,
        frame_count=frame_count,
    )
    return SequenceScore(
        example=example,
        sequence_id=sequence_id,
        detector=detector_name,
        frame_count=frame_count,
        metrics_by_tracker=metrics_by_tracker,
        ground_truth_tracks=ground_truth_tracks,
        predictions_by_tracker=predictions_by_tracker,
    )


def build_per_sequence_rows(sequence_scores: list[SequenceScore]) -> list[MOT17MetricsRow]:
    """Convert independent sequence scores into report rows."""

    rows: list[MOT17MetricsRow] = []
    for sequence_score in sequence_scores:
        for tracker_name, metrics in sequence_score.metrics_by_tracker.items():
            rows.append(
                build_metrics_row(
                    scope="sequence",
                    example=sequence_score.example,
                    sequence=sequence_score.sequence_id,
                    detector=sequence_score.detector,
                    tracker_name=tracker_name,
                    frame_count=sequence_score.frame_count,
                    metrics=metrics,
                )
            )
    return rows


def build_aggregate_rows(sequence_scores: list[SequenceScore]) -> list[MOT17MetricsRow]:
    """Recompute aggregate metrics per detector/tracker and overall tracker."""

    tracks_by_group: dict[tuple[str, str], AggregateTracks] = defaultdict(AggregateTracks)

    for sequence_score in sequence_scores:
        for tracker_name, predictions in sequence_score.predictions_by_tracker.items():
            aggregate_tracks(
                tracks_by_group[(sequence_score.detector, tracker_name)],
                sequence_score,
                predictions,
            )
            aggregate_tracks(
                tracks_by_group[("ALL", tracker_name)],
                sequence_score,
                predictions,
            )

    rows: list[MOT17MetricsRow] = []
    for (detector_name, tracker_name), grouped_tracks in sorted(tracks_by_group.items()):
        metrics = evaluate_mot_metrics(
            grouped_tracks.ground_truth,
            grouped_tracks.predictions,
            frame_count=grouped_tracks.frame_count,
        )
        rows.append(
            build_metrics_row(
                scope="aggregate",
                example="ALL",
                sequence="ALL",
                detector=detector_name,
                tracker_name=tracker_name,
                frame_count=grouped_tracks.frame_count,
                metrics=metrics,
            )
        )
    return rows


def aggregate_tracks(
    grouped_tracks: AggregateTracks,
    sequence_score: SequenceScore,
    predictions: list[Track],
) -> None:
    """Append one sequence into an aggregate timeline without ID collisions."""

    frame_offset = grouped_tracks.frame_count
    gt_track_offset = max_track_id(grouped_tracks.ground_truth)
    prediction_track_offset = max_track_id(grouped_tracks.predictions)
    grouped_tracks.ground_truth.extend(
        offset_tracks(
            sequence_score.ground_truth_tracks,
            frame_offset=frame_offset,
            track_id_offset=gt_track_offset,
        )
    )
    grouped_tracks.predictions.extend(
        offset_tracks(
            predictions,
            frame_offset=frame_offset,
            track_id_offset=prediction_track_offset,
        )
    )
    grouped_tracks.frame_count = frame_offset + sequence_score.frame_count


def offset_tracks(
    tracks: list[Track],
    *,
    frame_offset: int,
    track_id_offset: int,
) -> list[Track]:
    """Shift tracks into a pooled timeline used for aggregate evaluation."""

    return [
        Track(
            track_id=track.track_id + track_id_offset,
            frame_index=track.frame_index + frame_offset,
            x1=track.x1,
            y1=track.y1,
            x2=track.x2,
            y2=track.y2,
            score=track.score,
        )
        for track in tracks
    ]


def max_track_id(tracks: list[Track]) -> int:
    if not tracks:
        return 0
    return max(track.track_id for track in tracks)


def build_metrics_row(
    *,
    scope: str,
    example: str,
    sequence: str,
    detector: str,
    tracker_name: str,
    frame_count: int,
    metrics: MOTMetrics,
) -> MOT17MetricsRow:
    """Create one persisted report row from a computed metrics object."""

    return MOT17MetricsRow(
        scope=scope,
        example=example,
        sequence=sequence,
        detector=detector,
        tracker=TRACKER_LABELS.get(tracker_name, tracker_name),
        frames=frame_count,
        ground_truth_count=metrics.ground_truth_count,
        prediction_count=metrics.prediction_count,
        matches=metrics.matches,
        mota=metrics.mota,
        motp=metrics.motp,
        idf1=metrics.idf1,
        faf=metrics.faf,
        mostly_tracked=metrics.mostly_tracked,
        mostly_lost=metrics.mostly_lost,
        false_positives=metrics.false_positives,
        false_negatives=metrics.false_negatives,
        id_switches=metrics.id_switches,
        fragmentations=metrics.fragmentations,
    )


def write_metrics_csv(
    csv_path: Path,
    rows: list[MOT17MetricsRow],
) -> None:
    """Write typed report rows to the canonical CSV schema."""

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(row.to_csv_row() for row in rows)


def load_metrics_csv(csv_path: str | Path) -> list[MOT17MetricsRow]:
    """Load a metrics CSV written by :func:`write_metrics_csv`."""

    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return [
            MOT17MetricsRow.from_csv_row(row)
            for row in csv.DictReader(handle)
        ]


def split_sequence_id(sequence_id: str) -> tuple[str, str]:
    """Split `MOT17-02-FRCNN` into example and detector labels."""

    if "-" not in sequence_id:
        return sequence_id, "UNKNOWN"
    example, detector = sequence_id.rsplit("-", maxsplit=1)
    return example, detector


def infer_sequence_length(sequence_dir: Path) -> int:
    """Read the expected frame count from one sequence `seqinfo.ini`."""

    parser = ConfigParser()
    parser.read(sequence_dir / "seqinfo.ini")
    try:
        return parser.getint("Sequence", "seqLength")
    except Exception as exc:
        raise ValueError(f"Could not read seqLength from '{sequence_dir / 'seqinfo.ini'}'.") from exc


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Score every GT-backed MOT17 sequence and generate aggregate tracker reports."
    )
    parser.add_argument(
        "--sequence-root",
        default=str(DEFAULT_SEQUENCE_ROOT),
        help="Directory containing MOT17 sequence directories with det/ and gt/ subdirectories.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = generate_mot17_metrics_report(args.sequence_root)
    print(f"sequence_count={outputs.sequence_count}")
    print(f"metrics_csv={outputs.csv_path}")


if __name__ == "__main__":
    main()
