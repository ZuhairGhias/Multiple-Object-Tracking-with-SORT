"""Batch MOT17 scoring and CSV report persistence."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
import csv
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from src.metrics import MOTMetrics, compare_mot_metrics, evaluate_mot_metrics
from src.methods.detection import MOT17Detector
from src.methods.detection.mot import DEFAULT_MIN_SCORE
from src.methods.tracking import (
    DeepSORT,
    MOTGroundTruthTracker,
    MyDeepSORT2,
    NaiveIOUTracker,
    SORT,
    Track,
    Tracker,
)
from src.methods.tracking.deep_SORT import (
    ENCODER_CNN_COLOR,
    ENCODER_COLOR_HISTOGRAM,
    ENCODER_SIMPLE_CNN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE_ROOT = PROJECT_ROOT / "data" / "MOT17" / "train"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
DEFAULT_METRICS_CSV = DEFAULT_METRICS_DIR / "MOT17_tracking_metrics.csv"
DEFAULT_TRACKER_NAMES = ("naive_iou", "sort", "deep_sort", "my_deep_sort2")

TRACKER_BUILDERS: dict[str, Callable[[], Tracker]] = {
    "naive_iou": NaiveIOUTracker,
    "sort": SORT,
    "deep_sort": DeepSORT,
    "my_deep_sort2": MyDeepSORT2,
    "deep_sort_cnn": lambda: DeepSORT(encoder_name=ENCODER_SIMPLE_CNN),
    "deep_sort_color": lambda: DeepSORT(encoder_name=ENCODER_COLOR_HISTOGRAM),
    "deep_sort_cnn_color": lambda: DeepSORT(encoder_name=ENCODER_CNN_COLOR),
}
TRACKER_LABELS = {
    "naive_iou": "Naive IoU",
    "sort": "SORT",
    "deep_sort": "DeepSORT",
    "my_deep_sort2": "MyDeepSORT2",
    "deep_sort_cnn": "DeepSORT CNN",
    "deep_sort_color": "DeepSORT Color",
    "deep_sort_cnn_color": "DeepSORT CNN+Color",
}
METADATA_COLUMNS = (
    "Frames",
    "Detections",
    "GroundTruthCount",
    "IgnoredCount",
    "PredictionCount",
    "Matches",
    "RuntimeSeconds",
    "MsPerFrame",
    "PredictionsPerFrame",
    "MeanActiveTracks",
    "MaxActiveTracks",
)
METRIC_COLUMNS = ("MOTA", "MOTP", "IDF1", "FAF", "MT", "ML", "FP", "FN", "IDSW", "Frag")
CSV_COLUMNS = (
    "Scope",
    "Example",
    "Sequence",
    "Detector",
    "DetectorMinScore",
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
    detector_min_score: float | None
    frame_count: int
    metrics_by_tracker: dict[str, MOTMetrics]
    performance_by_tracker: dict[str, "TrackerPerformance"]
    ground_truth_tracks: list[Track]
    ignored_tracks: list[Track]
    predictions_by_tracker: dict[str, list[Track]]


@dataclass(frozen=True)
class MOT17MetricsOutputs:
    """Files and summary counts produced by the batch metrics command."""

    csv_path: Path
    sequence_count: int
    tracker_count: int


@dataclass
class TrackerPerformance:
    """Runtime and state-size counters collected while scoring one tracker."""

    frame_count: int = 0
    detection_count: int = 0
    runtime_seconds: float = 0.0
    active_track_sum: int = 0
    max_active_tracks: int = 0

    @property
    def ms_per_frame(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return 1000 * self.runtime_seconds / self.frame_count

    @property
    def mean_active_tracks(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.active_track_sum / self.frame_count

    def add_frame(
        self,
        *,
        detection_count: int,
        runtime_seconds: float,
        active_track_count: int,
    ) -> None:
        """Accumulate one tracker update measurement."""

        self.frame_count += 1
        self.detection_count += detection_count
        self.runtime_seconds += runtime_seconds
        self.active_track_sum += active_track_count
        self.max_active_tracks = max(self.max_active_tracks, active_track_count)

    def merge(self, other: "TrackerPerformance") -> None:
        """Accumulate another performance summary into this one."""

        self.frame_count += other.frame_count
        self.detection_count += other.detection_count
        self.runtime_seconds += other.runtime_seconds
        self.active_track_sum += other.active_track_sum
        self.max_active_tracks = max(self.max_active_tracks, other.max_active_tracks)


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
    detector_min_score: float | None
    tracker: str
    frames: int
    detections: int
    ground_truth_count: int
    ignored_count: int
    prediction_count: int
    matches: int
    runtime_seconds: float
    ms_per_frame: float
    predictions_per_frame: float
    mean_active_tracks: float
    max_active_tracks: int
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
            "DetectorMinScore": format_optional_float(self.detector_min_score),
            "Tracker": self.tracker,
            "Frames": self.frames,
            "Detections": self.detections,
            "GroundTruthCount": self.ground_truth_count,
            "IgnoredCount": self.ignored_count,
            "PredictionCount": self.prediction_count,
            "Matches": self.matches,
            "RuntimeSeconds": self.runtime_seconds,
            "MsPerFrame": self.ms_per_frame,
            "PredictionsPerFrame": self.predictions_per_frame,
            "MeanActiveTracks": self.mean_active_tracks,
            "MaxActiveTracks": self.max_active_tracks,
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
            detector_min_score=parse_optional_float(row.get("DetectorMinScore", "")),
            tracker=row["Tracker"],
            frames=int(row["Frames"]),
            detections=int(row.get("Detections", "0")),
            ground_truth_count=int(row["GroundTruthCount"]),
            ignored_count=int(row.get("IgnoredCount", "0")),
            prediction_count=int(row["PredictionCount"]),
            matches=int(row["Matches"]),
            runtime_seconds=float(row.get("RuntimeSeconds", "0")),
            ms_per_frame=float(row.get("MsPerFrame", "0")),
            predictions_per_frame=float(row.get("PredictionsPerFrame", "0")),
            mean_active_tracks=float(row.get("MeanActiveTracks", "0")),
            max_active_tracks=int(float(row.get("MaxActiveTracks", "0"))),
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
    ignored: list[Track] = field(default_factory=list)
    predictions: list[Track] = field(default_factory=list)
    performance: TrackerPerformance = field(default_factory=TrackerPerformance)
    frame_count: int = 0
    detector_min_score: float | None = None


def generate_mot17_metrics_report(
    sequence_root: str | Path = DEFAULT_SEQUENCE_ROOT,
    *,
    sequence_filter: str | None = None,
    tracker_names: list[str] | None = None,
    frame_limit: int | None = None,
    detector_min_score: float | None = DEFAULT_MIN_SCORE,
    output_csv: str | Path = DEFAULT_METRICS_CSV,
) -> MOT17MetricsOutputs:
    """Score every GT-backed MOT17 sequence and write the canonical CSV report.

    The local MOT17 test split does not contain `gt/gt.txt`, so the default
    root points at `data/MOT17/train`. Each detector-specific sequence is
    scored independently, then aggregate rows are recomputed from pooled raw
    tracks instead of averaging per-sequence percentages.
    """

    selected_tracker_names = resolve_tracker_names(tracker_names)
    sequence_dirs = filter_sequence_dirs(
        discover_scored_sequence_dirs(sequence_root),
        sequence_filter=sequence_filter,
    )
    sequence_scores = []
    for sequence_index, sequence_dir in enumerate(sequence_dirs, start=1):
        print(f"[{sequence_index}/{len(sequence_dirs)}] Scoring {sequence_dir.name}")
        sequence_scores.append(
            score_sequence(
                sequence_dir,
                tracker_names=selected_tracker_names,
                frame_limit=frame_limit,
                detector_min_score=detector_min_score,
            )
        )
    aggregate_rows = build_aggregate_rows(sequence_scores)
    per_sequence_rows = build_per_sequence_rows(sequence_scores)
    rows = per_sequence_rows + aggregate_rows

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(csv_path, rows)
    return MOT17MetricsOutputs(
        csv_path=csv_path,
        sequence_count=len(sequence_scores),
        tracker_count=len(selected_tracker_names),
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


def filter_sequence_dirs(
    sequence_dirs: list[Path],
    *,
    sequence_filter: str | None,
) -> list[Path]:
    """Filter discovered sequences by comma-separated name fragments."""

    if sequence_filter is None:
        return sequence_dirs

    filters = [
        item.strip().lower()
        for item in sequence_filter.split(",")
        if item.strip()
    ]
    if not filters:
        return sequence_dirs

    filtered = [
        sequence_dir
        for sequence_dir in sequence_dirs
        if any(filter_value in sequence_dir.name.lower() for filter_value in filters)
    ]
    if not filtered:
        raise FileNotFoundError(f"No MOT17 sequences matched filter '{sequence_filter}'.")
    return filtered


def resolve_tracker_names(tracker_names: list[str] | None) -> list[str]:
    """Resolve optional tracker keys into a stable configured tracker list."""

    if tracker_names is None:
        return list(DEFAULT_TRACKER_NAMES)

    resolved = []
    for tracker_name in tracker_names:
        key = tracker_name.strip()
        if not key:
            continue
        if key not in TRACKER_BUILDERS:
            known = ", ".join(TRACKER_BUILDERS)
            raise ValueError(f"Unknown tracker '{key}'. Expected one of: {known}.")
        resolved.append(key)

    if not resolved:
        raise ValueError("At least one tracker is required.")
    return resolved


def score_sequence(
    sequence_dir: Path,
    *,
    tracker_names: list[str] | None = None,
    frame_limit: int | None = None,
    detector_min_score: float | None = DEFAULT_MIN_SCORE,
) -> SequenceScore:
    """Run the configured trackers over one MOT17 sequence and score them."""

    selected_tracker_names = resolve_tracker_names(tracker_names)
    sequence_id = sequence_dir.name
    example, detector_name = split_sequence_id(sequence_id)
    frame_count = infer_sequence_length(sequence_dir)
    if frame_limit is not None:
        if frame_limit <= 0:
            raise ValueError("frame_limit must be greater than zero.")
        frame_count = min(frame_count, frame_limit)
    detector = MOT17Detector(
        sequence_id=sequence_id,
        root_dir=sequence_dir.parent,
        min_score=detector_min_score,
    )
    ground_truth_tracker = MOTGroundTruthTracker(
        sequence_id=sequence_id,
        root_dir=sequence_dir.parent,
    )
    trackers = {
        tracker_name: builder()
        for tracker_name, builder in TRACKER_BUILDERS.items()
        if tracker_name in selected_tracker_names
    }
    needs_frame = any("deep_sort" in tracker_name for tracker_name in trackers)
    ground_truth_tracks: list[Track] = []
    ignored_tracks: list[Track] = []
    predictions_by_tracker: dict[str, list[Track]] = {
        tracker_name: []
        for tracker_name in trackers
    }
    performance_by_tracker = {
        tracker_name: TrackerPerformance()
        for tracker_name in trackers
    }

    for frame_index in range(1, frame_count + 1):
        detections = detector.get_detections(frame_index=frame_index)
        frame = None
        if needs_frame:
            frame = cv2.imread(str(sequence_dir / "img1" / f"{frame_index:06d}.jpg"))
            if frame is None:
                raise ValueError(f"Could not read frame {frame_index} for '{sequence_id}'.")

        ground_truth_tracks.extend(
            ground_truth_tracker.update(detections, frame_index=frame_index)
        )
        ignored_tracks.extend(
            ground_truth_tracker.ignored_regions(frame_index=frame_index)
        )
        for tracker_name, tracker in trackers.items():
            started_at = perf_counter()
            if "deep_sort" in tracker_name:
                tracks = tracker.update(detections, frame_index=frame_index, frame=frame)
            else:
                tracks = tracker.update(detections, frame_index=frame_index)
            runtime_seconds = perf_counter() - started_at
            active_track_count = len(getattr(tracker, "current_tracks", []))
            performance_by_tracker[tracker_name].add_frame(
                detection_count=len(detections),
                runtime_seconds=runtime_seconds,
                active_track_count=active_track_count,
            )
            predictions_by_tracker[tracker_name].extend(tracks)

    metrics_by_tracker = compare_mot_metrics(
        ground_truth_tracks,
        predictions_by_tracker,
        ignored_tracks=ignored_tracks,
        frame_count=frame_count,
    )
    return SequenceScore(
        example=example,
        sequence_id=sequence_id,
        detector=detector_name,
        detector_min_score=detector_min_score,
        frame_count=frame_count,
        metrics_by_tracker=metrics_by_tracker,
        performance_by_tracker=performance_by_tracker,
        ground_truth_tracks=ground_truth_tracks,
        ignored_tracks=ignored_tracks,
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
                    detector_min_score=sequence_score.detector_min_score,
                    tracker_name=tracker_name,
                    frame_count=sequence_score.frame_count,
                    ignored_count=len(sequence_score.ignored_tracks),
                    performance=sequence_score.performance_by_tracker[tracker_name],
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
                tracker_name,
                predictions,
            )
            aggregate_tracks(
                tracks_by_group[("ALL", tracker_name)],
                sequence_score,
                tracker_name,
                predictions,
            )

    rows: list[MOT17MetricsRow] = []
    for (detector_name, tracker_name), grouped_tracks in sorted(tracks_by_group.items()):
        metrics = evaluate_mot_metrics(
            grouped_tracks.ground_truth,
            grouped_tracks.predictions,
            ignored_tracks=grouped_tracks.ignored,
            frame_count=grouped_tracks.frame_count,
        )
        rows.append(
            build_metrics_row(
                scope="aggregate",
                example="ALL",
                sequence="ALL",
                detector=detector_name,
                detector_min_score=grouped_tracks.detector_min_score,
                tracker_name=tracker_name,
                frame_count=grouped_tracks.frame_count,
                ignored_count=len(grouped_tracks.ignored),
                performance=grouped_tracks.performance,
                metrics=metrics,
            )
        )
    return rows


def aggregate_tracks(
    grouped_tracks: AggregateTracks,
    sequence_score: SequenceScore,
    tracker_name: str,
    predictions: list[Track],
) -> None:
    """Append one sequence into an aggregate timeline without ID collisions."""

    frame_offset = grouped_tracks.frame_count
    grouped_tracks.detector_min_score = sequence_score.detector_min_score
    gt_track_offset = max_track_id(grouped_tracks.ground_truth)
    prediction_track_offset = max_track_id(grouped_tracks.predictions)
    grouped_tracks.ground_truth.extend(
        offset_tracks(
            sequence_score.ground_truth_tracks,
            frame_offset=frame_offset,
            track_id_offset=gt_track_offset,
        )
    )
    grouped_tracks.ignored.extend(
        offset_tracks(
            sequence_score.ignored_tracks,
            frame_offset=frame_offset,
            track_id_offset=0,
        )
    )
    grouped_tracks.predictions.extend(
        offset_tracks(
            predictions,
            frame_offset=frame_offset,
            track_id_offset=prediction_track_offset,
        )
    )
    grouped_tracks.performance.merge(sequence_score.performance_by_tracker[tracker_name])
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
    detector_min_score: float | None,
    tracker_name: str,
    frame_count: int,
    ignored_count: int,
    performance: TrackerPerformance,
    metrics: MOTMetrics,
) -> MOT17MetricsRow:
    """Create one persisted report row from a computed metrics object."""

    return MOT17MetricsRow(
        scope=scope,
        example=example,
        sequence=sequence,
        detector=detector,
        detector_min_score=detector_min_score,
        tracker=TRACKER_LABELS.get(tracker_name, tracker_name),
        frames=frame_count,
        detections=performance.detection_count,
        ground_truth_count=metrics.ground_truth_count,
        ignored_count=ignored_count,
        prediction_count=metrics.prediction_count,
        matches=metrics.matches,
        runtime_seconds=performance.runtime_seconds,
        ms_per_frame=performance.ms_per_frame,
        predictions_per_frame=_safe_divide(metrics.prediction_count, frame_count),
        mean_active_tracks=performance.mean_active_tracks,
        max_active_tracks=performance.max_active_tracks,
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


def parse_optional_float(value: str) -> float | None:
    """Parse a CSV optional float field.

    >>> parse_optional_float("")
    >>> parse_optional_float("0.25")
    0.25
    """

    if value == "":
        return None
    return float(value)


def format_optional_float(value: float | None) -> str | float:
    """Format an optional float for CSV output.

    >>> format_optional_float(None)
    ''
    >>> format_optional_float(0.0)
    0.0
    """

    if value is None:
        return ""
    return value


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
    parser.add_argument(
        "--sequence-filter",
        default=None,
        help="Comma-separated sequence name fragments, for example MOT17-04-SDP.",
    )
    parser.add_argument(
        "--trackers",
        default=None,
        help="Comma-separated tracker keys. Defaults to all configured trackers.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Score only the first N frames of each selected sequence.",
    )
    parser.add_argument(
        "--detector-min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum MOT17 detection confidence to keep before tracking.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="CSV output path for the generated metrics report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    tracker_names = None
    if args.trackers is not None:
        tracker_names = [tracker_name.strip() for tracker_name in args.trackers.split(",")]
    outputs = generate_mot17_metrics_report(
        args.sequence_root,
        sequence_filter=args.sequence_filter,
        tracker_names=tracker_names,
        frame_limit=args.frame_limit,
        detector_min_score=args.detector_min_score,
        output_csv=args.output_csv,
    )
    print(f"sequence_count={outputs.sequence_count}")
    print(f"tracker_count={outputs.tracker_count}")
    print(f"metrics_csv={outputs.csv_path}")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()
