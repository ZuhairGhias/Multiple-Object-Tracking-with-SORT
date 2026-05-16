"""Static plots generated from the persisted MOT17 metrics CSV report."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.mot17_metrics import (
    DEFAULT_METRICS_CSV,
    DEFAULT_METRICS_DIR,
    MOT17MetricsRow,
    TRACKER_LABELS,
    load_metrics_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRIC_IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "metrics"
DETECTOR_ORDER = ("DPM", "FRCNN", "SDP")
TRACKER_ORDER = (
    "Naive IoU",
    "SORT",
    "DeepSORT",
    "MyDeepSORT2",
)
TRACKER_PALETTE = {
    "Naive IoU": "#147d76",
    "SORT": "#c86b1f",
    "DeepSORT": "#5b5fc7",
    "MyDeepSORT2": "#3f8f62",
}
DETECTOR_MARKERS = {
    "DPM": "o",
    "FRCNN": "s",
    "SDP": "^",
}
HEADLINE_METRICS = ("MOTA", "IDF1", "MOTP")
ERROR_METRICS = ("FP", "FN", "IDSW", "Frag")
PERFORMANCE_METRICS = (
    ("RuntimeSeconds", "Runtime", "Seconds (log scale)", True),
    ("MsPerFrame", "Frame time", "Milliseconds/frame (log scale)", True),
    ("MeanActiveTracks", "Mean active tracks", "Tracks", False),
    ("PredictionsPerFrame", "Predictions per frame", "Tracks/frame", False),
)
@dataclass(frozen=True)
class MOT17AggregatePlotOutputs:
    """Static aggregate plot files created from the saved metrics CSV."""

    scores_by_detector_image_path: Path
    errors_by_detector_image_path: Path
    performance_by_detector_image_path: Path
    mota_idf1_motp_bubbles_image_path: Path
    metrics_table_path: Path
    aggregate_row_count: int


def generate_mot17_aggregate_plots(
    *,
    tracker_names: list[str] | None = None,
    detector_names: list[str] | None = None,
    output_stem: str | None = None,
) -> MOT17AggregatePlotOutputs:
    """Generate detector/tracker aggregate plots from the canonical CSV report.

    Run `python -m src.utils.mot17_metrics` first if the metrics CSV does not
    exist yet. This plotting step intentionally reads the persisted report
    rather than rerunning tracker evaluation.
    """

    if not DEFAULT_METRICS_CSV.is_file():
        raise FileNotFoundError(
            "Could not find metrics report "
            f"'{DEFAULT_METRICS_CSV}'. "
            "Run `python -m src.utils.mot17_metrics` first."
        )

    rows = load_metrics_csv(DEFAULT_METRICS_CSV)
    tracker_labels = resolve_tracker_labels(tracker_names)
    detector_labels = resolve_detector_labels(detector_names)
    aggregate_rows = [
        row
        for row in rows
        if row.scope == "aggregate" and row.detector in DETECTOR_ORDER
    ]
    if tracker_labels is not None:
        aggregate_rows = [
            row
            for row in aggregate_rows
            if row.tracker in tracker_labels
        ]
    if detector_labels is not None:
        aggregate_rows = [
            row
            for row in aggregate_rows
            if row.detector in detector_labels
        ]
    if not aggregate_rows:
        raise ValueError("No detector-specific aggregate metric rows matched the requested filters.")

    DEFAULT_METRIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_stem or default_output_stem(
        tracker_labels=tracker_labels,
        detector_labels=detector_labels,
    )

    if stem is None:
        scores_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_metric_scores_by_detector.png"
        errors_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_metric_errors_by_detector.png"
        performance_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_performance_by_detector.png"
        bubble_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_mota_idf1_motp_bubbles.png"
        table_path = DEFAULT_METRICS_DIR / "MOT17_tracking_metric_table.md"
    else:
        scores_image_path = DEFAULT_METRIC_IMAGE_DIR / f"{stem}_metric_scores_by_detector.png"
        errors_image_path = DEFAULT_METRIC_IMAGE_DIR / f"{stem}_metric_errors_by_detector.png"
        performance_image_path = DEFAULT_METRIC_IMAGE_DIR / f"{stem}_performance_by_detector.png"
        bubble_image_path = DEFAULT_METRIC_IMAGE_DIR / f"{stem}_mota_idf1_motp_bubbles.png"
        table_path = DEFAULT_METRICS_DIR / f"{stem}_metric_table.md"

    write_metric_grid(
        scores_image_path,
        aggregate_rows,
        metric_names=HEADLINE_METRICS,
        title="MOT17 aggregate scores by detector and tracker",
        y_label="Score (%)",
        y_as_percent=True,
    )
    write_metric_grid(
        errors_image_path,
        aggregate_rows,
        metric_names=ERROR_METRICS,
        title="MOT17 aggregate errors by detector and tracker",
        y_label="Count",
    )
    write_mota_idf1_motp_bubble_chart(
        bubble_image_path,
        aggregate_rows,
    )
    write_performance_grid(
        performance_image_path,
        aggregate_rows,
    )
    write_metrics_markdown_table(
        table_path,
        aggregate_rows,
    )
    return MOT17AggregatePlotOutputs(
        scores_by_detector_image_path=scores_image_path,
        errors_by_detector_image_path=errors_image_path,
        performance_by_detector_image_path=performance_image_path,
        mota_idf1_motp_bubbles_image_path=bubble_image_path,
        metrics_table_path=table_path,
        aggregate_row_count=len(aggregate_rows),
    )


def write_metric_grid(
    image_path: Path,
    rows: list[MOT17MetricsRow],
    *,
    metric_names: tuple[str, ...],
    title: str,
    y_label: str,
    y_as_percent: bool = False,
) -> None:
    """Write a small-multiple bar-chart grid for aggregate metrics."""

    sns.set_theme(style="whitegrid", context="talk")
    tracker_order = tracker_order_for_rows(rows)
    detector_order = detector_order_for_rows(rows)
    column_count = 2 if len(metric_names) > 3 else len(metric_names)
    row_count = (len(metric_names) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.2 * column_count, 4.8 * row_count),
        squeeze=False,
    )

    for axis_index, metric_name in enumerate(metric_names):
        axis = axes[axis_index // column_count][axis_index % column_count]
        chart_data = build_metric_chart_data(rows, metric_name)
        sns.barplot(
            data=chart_data,
            x="Detector",
            y="Value",
            hue="Tracker",
            order=detector_order,
            hue_order=tracker_order,
            palette=palette_for_trackers(tracker_order),
            ax=axis,
        )
        axis.set_title(metric_name, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel(y_label)
        if y_as_percent:
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.legend(title="")
        sns.despine(ax=axis)

    for axis_index in range(len(metric_names), row_count * column_count):
        axis = axes[axis_index // column_count][axis_index % column_count]
        axis.axis("off")

    figure.suptitle(title, y=1.02, fontweight="bold")
    figure.tight_layout()
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_metric_chart_data(
    rows: list[MOT17MetricsRow],
    metric_name: str,
) -> dict[str, list[str | float]]:
    """Convert typed metric rows into Seaborn's column-oriented input."""

    chart_data: dict[str, list[str | float]] = {
        "Detector": [],
        "Tracker": [],
        "Value": [],
    }
    for row in rows:
        chart_data["Detector"].append(row.detector)
        chart_data["Tracker"].append(row.tracker)
        chart_data["Value"].append(float(row.metric_value(metric_name)))
    return chart_data


def write_performance_grid(
    image_path: Path,
    rows: list[MOT17MetricsRow],
) -> None:
    """Write runtime and output-rate plots for aggregate tracker comparison."""

    sns.set_theme(style="whitegrid", context="talk")
    tracker_order = tracker_order_for_rows(rows)
    detector_order = detector_order_for_rows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), squeeze=False)

    for axis_index, (field_name, title, y_label, use_log_scale) in enumerate(PERFORMANCE_METRICS):
        axis = axes[axis_index // 2][axis_index % 2]
        chart_data = build_performance_chart_data(rows, field_name)
        sns.barplot(
            data=chart_data,
            x="Detector",
            y="Value",
            hue="Tracker",
            order=detector_order,
            hue_order=tracker_order,
            palette=palette_for_trackers(tracker_order),
            ax=axis,
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel(y_label)
        if use_log_scale:
            axis.set_yscale("log")
        axis.legend(title="")
        sns.despine(ax=axis)

    figure.suptitle("MOT17 aggregate runtime characteristics by detector and tracker", y=1.02, fontweight="bold")
    figure.tight_layout()
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_performance_chart_data(
    rows: list[MOT17MetricsRow],
    field_name: str,
) -> dict[str, list[str | float]]:
    """Convert persisted performance fields into Seaborn input."""

    chart_data: dict[str, list[str | float]] = {
        "Detector": [],
        "Tracker": [],
        "Value": [],
    }
    for row in rows:
        chart_data["Detector"].append(row.detector)
        chart_data["Tracker"].append(row.tracker)
        chart_data["Value"].append(float(getattr(row, snake_case(field_name))))
    return chart_data


def write_metrics_markdown_table(
    table_path: Path,
    rows: list[MOT17MetricsRow],
) -> None:
    """Write aggregate metrics as Markdown tables grouped by detector."""

    tracker_order = tracker_order_for_rows(rows)
    detector_order = detector_order_for_rows(rows)
    lines = ["# MOT17 Aggregate Metrics", ""]
    for detector_name in detector_order:
        detector_rows = [
            row
            for row in rows
            if row.detector == detector_name
        ]
        if not detector_rows:
            continue

        rows_by_tracker = {row.tracker: row for row in detector_rows}
        lines.extend(
            [
                f"## {detector_name}",
                "",
                "| Tracker | MOTA | IDF1 | MOTP | FP | FN | IDSW | Frag | ms/frame |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tracker_name in tracker_order:
            row = rows_by_tracker.get(tracker_name)
            if row is None:
                continue
            lines.append(
                "| "
                f"{row.tracker} | "
                f"{format_percent(row.mota)} | "
                f"{format_percent(row.idf1)} | "
                f"{format_percent(row.motp)} | "
                f"{row.false_positives} | "
                f"{row.false_negatives} | "
                f"{row.id_switches} | "
                f"{row.fragmentations} | "
                f"{row.ms_per_frame:.2f} |"
            )
        lines.append("")

    table_path.write_text("\n".join(lines), encoding="utf-8")


def format_percent(value: float) -> str:
    """Format a metric stored as a fraction for Markdown tables.

    >>> format_percent(0.1234)
    '12.3%'
    """

    return f"{value * 100:.1f}%"


def snake_case(field_name: str) -> str:
    """Map selected report column names to `MOT17MetricsRow` attributes.

    >>> snake_case("MsPerFrame")
    'ms_per_frame'
    """

    return {
        "RuntimeSeconds": "runtime_seconds",
        "MsPerFrame": "ms_per_frame",
        "MeanActiveTracks": "mean_active_tracks",
        "PredictionsPerFrame": "predictions_per_frame",
    }[field_name]


def tracker_order_for_rows(rows: list[MOT17MetricsRow]) -> list[str]:
    """Return the known tracker order filtered to rows present in a report.

    >>> tracker_order_for_rows([])
    []
    """

    row_trackers = {row.tracker for row in rows}
    ordered_trackers = [
        tracker_name
        for tracker_name in TRACKER_ORDER
        if tracker_name in row_trackers
    ]
    extra_trackers = sorted(row_trackers - set(ordered_trackers))
    return [*ordered_trackers, *extra_trackers]


def detector_order_for_rows(rows: list[MOT17MetricsRow]) -> list[str]:
    """Return the canonical detector order filtered to rows present in a report.

    >>> detector_order_for_rows([])
    []
    """

    row_detectors = {row.detector for row in rows}
    ordered_detectors = [
        detector_name
        for detector_name in DETECTOR_ORDER
        if detector_name in row_detectors
    ]
    extra_detectors = sorted(row_detectors - set(ordered_detectors))
    return [*ordered_detectors, *extra_detectors]


def palette_for_trackers(tracker_order: list[str]) -> list[str]:
    """Return plot colors in the same order as the tracker labels."""

    return [TRACKER_PALETTE.get(tracker_name, "#777777") for tracker_name in tracker_order]


def bubble_tracker_label(tracker_name: str) -> str:
    """Return a short label that fits inside a bubble marker.

    >>> bubble_tracker_label("MyDeepSORT2")
    'My\\nDeep2'
    """

    return {
        "Naive IoU": "Naive\nIoU",
        "SORT": "SORT",
        "DeepSORT": "Deep\nSORT",
        "MyDeepSORT2": "My\nDeep2",
    }.get(tracker_name, tracker_name)


def resolve_tracker_labels(tracker_names: list[str] | None) -> list[str] | None:
    """Resolve optional tracker keys or labels into report labels.

    >>> resolve_tracker_labels(["naive_iou", "SORT"])
    ['Naive IoU', 'SORT']
    """

    if tracker_names is None:
        return None

    labels_by_lowercase = {
        label.lower(): label
        for label in TRACKER_LABELS.values()
    }
    resolved = []
    for tracker_name in tracker_names:
        tracker_key = tracker_name.strip()
        if not tracker_key:
            continue

        if tracker_key in TRACKER_LABELS:
            resolved.append(TRACKER_LABELS[tracker_key])
            continue

        tracker_label = labels_by_lowercase.get(tracker_key.lower())
        if tracker_label is not None:
            resolved.append(tracker_label)
            continue

        known = ", ".join([*TRACKER_LABELS, *TRACKER_LABELS.values()])
        raise ValueError(f"Unknown tracker '{tracker_key}'. Expected one of: {known}.")

    if not resolved:
        raise ValueError("At least one tracker is required when --trackers is provided.")
    return resolved


def resolve_detector_labels(detector_names: list[str] | None) -> list[str] | None:
    """Resolve optional detector names into canonical report labels.

    Detector order is canonicalized so the same filter always produces the same
    plot order and default filename.

    >>> resolve_detector_labels(["sdp", "DPM"])
    ['DPM', 'SDP']
    """

    if detector_names is None:
        return None

    labels_by_lowercase = {
        detector_name.lower(): detector_name
        for detector_name in DETECTOR_ORDER
    }
    requested = set()
    for detector_name in detector_names:
        detector_key = detector_name.strip()
        if not detector_key:
            continue

        detector_label = labels_by_lowercase.get(detector_key.lower())
        if detector_label is None:
            known = ", ".join(DETECTOR_ORDER)
            raise ValueError(f"Unknown detector '{detector_key}'. Expected one of: {known}.")
        requested.add(detector_label)

    if not requested:
        raise ValueError("At least one detector is required when --detectors is provided.")
    return [
        detector_name
        for detector_name in DETECTOR_ORDER
        if detector_name in requested
    ]


def default_output_stem(
    *,
    tracker_labels: list[str] | None,
    detector_labels: list[str] | None,
) -> str | None:
    """Return the default filtered output stem, or `None` for full-report plots.

    >>> default_output_stem(tracker_labels=["Naive IoU"], detector_labels=["SDP"])
    'MOT17_tracking_trackers-naive_iou_detectors-sdp'
    >>> default_output_stem(tracker_labels=None, detector_labels=None) is None
    True
    """

    if tracker_labels is None and detector_labels is None:
        return None

    tracker_part = "all"
    if tracker_labels is not None:
        tracker_part = "_".join(
            sanitize_filename_part(tracker_name)
            for tracker_name in tracker_order_for_labels(tracker_labels)
        )

    detector_part = "all"
    if detector_labels is not None:
        detector_part = "_".join(
            sanitize_filename_part(detector_name)
            for detector_name in detector_labels
        )

    return f"MOT17_tracking_trackers-{tracker_part}_detectors-{detector_part}"


def tracker_order_for_labels(tracker_labels: list[str]) -> list[str]:
    """Return canonical tracker order for a selected label list."""

    selected_trackers = set(tracker_labels)
    ordered_trackers = [
        tracker_name
        for tracker_name in TRACKER_ORDER
        if tracker_name in selected_trackers
    ]
    extra_trackers = sorted(selected_trackers - set(ordered_trackers))
    return [*ordered_trackers, *extra_trackers]


def sanitize_filename_part(value: str) -> str:
    """Return a safe filename fragment for filtered plot outputs.

    >>> sanitize_filename_part("Naive IoU")
    'naive_iou'
    """

    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return sanitized.lower()


def write_mota_idf1_motp_bubble_chart(
    image_path: Path,
    rows: list[MOT17MetricsRow],
) -> None:
    """Write a paper-style MOTA/IDF1 scatter plot with MOTP-sized bubbles.

    The StrongSORT figure uses HOTA as bubble radius. This project does not
    compute HOTA yet, so this chart uses MOTP instead and states that directly
    in the title and legend.
    """

    sns.set_theme(style="whitegrid", context="talk")
    figure, axis = plt.subplots(figsize=(11.6, 7.2))
    tracker_order = tracker_order_for_rows(rows)
    detector_order = detector_order_for_rows(rows)

    for tracker_name in tracker_order:
        for detector_name in detector_order:
            row = next(
                (
                    item
                    for item in rows
                    if item.tracker == tracker_name and item.detector == detector_name
                ),
                None,
            )
            if row is None:
                continue
            bubble_size = 2200 * row.motp
            axis.scatter(
                row.mota,
                row.idf1,
                s=bubble_size,
                color=TRACKER_PALETTE.get(tracker_name, "#777777"),
                marker=DETECTOR_MARKERS.get(detector_name, "o"),
                alpha=0.78,
                edgecolor="white",
                linewidth=1.5,
            )
            axis.text(
                row.mota,
                row.idf1,
                bubble_tracker_label(tracker_name),
                ha="center",
                va="center",
                fontsize=8,
                weight="semibold",
                color="white",
            )

    axis.set_title("MOT17 aggregate IDF1-MOTA comparison; bubble area scales with MOTP", pad=16, fontweight="bold")
    axis.set_xlabel("MOTA (%)")
    axis.set_ylabel("IDF1 (%)")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.margins(x=0.08, y=0.08)
    tracker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=TRACKER_PALETTE.get(tracker_name, "#777777"),
            markeredgecolor="white",
            markersize=11,
            label=tracker_name,
        )
        for tracker_name in tracker_order
    ]
    detector_handles = [
        Line2D(
            [0],
            [0],
            marker=DETECTOR_MARKERS.get(detector_name, "o"),
            color="#444",
            linestyle="none",
            markersize=9,
            label=detector_name,
        )
        for detector_name in detector_order
    ]
    tracker_legend = axis.legend(
        handles=tracker_handles,
        title="Tracker",
        frameon=True,
        loc="lower right",
        fontsize=10,
        title_fontsize=11,
    )
    axis.add_artist(tracker_legend)
    axis.legend(
        handles=detector_handles,
        title="Detector",
        frameon=True,
        loc="upper left",
        fontsize=10,
        title_fontsize=11,
    )
    sns.despine(ax=axis)
    figure.tight_layout()
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """CLI entry point for generating aggregate report plots."""

    parser = build_parser()
    args = parser.parse_args()
    tracker_names = None
    if args.trackers is not None:
        tracker_names = [tracker_name.strip() for tracker_name in args.trackers.split(",")]
    detector_names = None
    if args.detectors is not None:
        detector_names = [detector_name.strip() for detector_name in args.detectors.split(",")]

    outputs = generate_mot17_aggregate_plots(
        tracker_names=tracker_names,
        detector_names=detector_names,
        output_stem=args.output_stem,
    )
    print(f"aggregate_row_count={outputs.aggregate_row_count}")
    print(f"scores_by_detector_image={outputs.scores_by_detector_image_path}")
    print(f"errors_by_detector_image={outputs.errors_by_detector_image_path}")
    print(f"performance_by_detector_image={outputs.performance_by_detector_image_path}")
    print(f"mota_idf1_motp_bubbles_image={outputs.mota_idf1_motp_bubbles_image_path}")
    print(f"metrics_table={outputs.metrics_table_path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate MOT17 aggregate plots and Markdown metric tables."
    )
    parser.add_argument(
        "--trackers",
        default=None,
        help="Comma-separated tracker keys or labels to include, for example naive_iou,sort.",
    )
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated detector labels to include, for example SDP or DPM,FRCNN,SDP.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Optional output filename stem. Defaults to standard names, or a filter-derived stem when filtered.",
    )
    return parser


if __name__ == "__main__":
    main()
