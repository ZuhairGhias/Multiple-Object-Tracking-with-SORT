"""Static plots generated from the persisted MOT17 metrics CSV report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.mot17_metrics import (
    DEFAULT_METRICS_CSV,
    MOT17MetricsRow,
    load_metrics_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRIC_IMAGE_DIR = PROJECT_ROOT / "data" / "images" / "metrics"
DETECTOR_ORDER = ("DPM", "FRCNN", "SDP")
TRACKER_ORDER = ("Naive IoU", "SORT")
HEADLINE_METRICS = ("MOTA", "IDF1", "MOTP")
ERROR_METRICS = ("FP", "FN", "IDSW", "Frag")
POINT_LABEL_OFFSETS = {
    ("DPM", "Naive IoU"): (8, -10),
    ("DPM", "SORT"): (8, 8),
    ("FRCNN", "Naive IoU"): (10, -12),
    ("FRCNN", "SORT"): (10, 10),
    ("SDP", "Naive IoU"): (10, 12),
    ("SDP", "SORT"): (10, 16),
}


@dataclass(frozen=True)
class MOT17AggregatePlotOutputs:
    """Static aggregate plot files created from the saved metrics CSV."""

    scores_by_detector_image_path: Path
    errors_by_detector_image_path: Path
    mota_idf1_motp_bubbles_image_path: Path
    aggregate_row_count: int


def generate_mot17_aggregate_plots(
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
    aggregate_rows = [
        row
        for row in rows
        if row.scope == "aggregate" and row.detector in DETECTOR_ORDER
    ]
    if not aggregate_rows:
        raise ValueError("No detector-specific aggregate metric rows were found in the CSV report.")

    DEFAULT_METRIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    scores_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_metric_scores_by_detector.png"
    errors_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_metric_errors_by_detector.png"
    bubble_image_path = DEFAULT_METRIC_IMAGE_DIR / "MOT17_tracking_mota_idf1_motp_bubbles.png"

    write_metric_grid(
        scores_image_path,
        aggregate_rows,
        metric_names=HEADLINE_METRICS,
        title="MOT17 aggregate scores by detector and tracker",
        y_label="Score",
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
    return MOT17AggregatePlotOutputs(
        scores_by_detector_image_path=scores_image_path,
        errors_by_detector_image_path=errors_image_path,
        mota_idf1_motp_bubbles_image_path=bubble_image_path,
        aggregate_row_count=len(aggregate_rows),
    )


def write_metric_grid(
    image_path: Path,
    rows: list[MOT17MetricsRow],
    *,
    metric_names: tuple[str, ...],
    title: str,
    y_label: str,
) -> None:
    """Write a small-multiple bar-chart grid for aggregate metrics."""

    sns.set_theme(style="whitegrid", context="talk")
    column_count = 2 if len(metric_names) > 3 else len(metric_names)
    row_count = (len(metric_names) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.4 * column_count, 4.8 * row_count),
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
            order=DETECTOR_ORDER,
            hue_order=TRACKER_ORDER,
            palette=("#147d76", "#c86b1f"),
            ax=axis,
        )
        axis.set_title(metric_name, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel(y_label)
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
    figure, axis = plt.subplots(figsize=(10.4, 6.6))
    palette = {
        "Naive IoU": "#147d76",
        "SORT": "#c86b1f",
    }
    marker_by_detector = {
        "DPM": "o",
        "FRCNN": "s",
        "SDP": "^",
    }

    for tracker_name in TRACKER_ORDER:
        for detector_name in DETECTOR_ORDER:
            row = next(
                item
                for item in rows
                if item.tracker == tracker_name and item.detector == detector_name
            )
            bubble_size = 2200 * row.motp
            axis.scatter(
                row.mota,
                row.idf1,
                s=bubble_size,
                color=palette[tracker_name],
                marker=marker_by_detector[detector_name],
                alpha=0.78,
                edgecolor="white",
                linewidth=1.5,
                label=tracker_name if detector_name == DETECTOR_ORDER[0] else None,
            )
            offset = POINT_LABEL_OFFSETS[(detector_name, tracker_name)]
            axis.annotate(
                f"{detector_name} {tracker_name}",
                (row.mota, row.idf1),
                xytext=offset,
                textcoords="offset points",
                fontsize=10,
                weight="semibold",
            )

    axis.set_title("MOT17 aggregate IDF1-MOTA comparison; bubble area scales with MOTP", pad=16, fontweight="bold")
    axis.set_xlabel("MOTA")
    axis.set_ylabel("IDF1")
    axis.margins(x=0.08, y=0.08)
    axis.legend(title="Tracker", frameon=True, loc="lower right", fontsize=11, title_fontsize=12)
    sns.despine(ax=axis)
    figure.tight_layout()
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """CLI entry point for generating aggregate report plots."""

    outputs = generate_mot17_aggregate_plots()
    print(f"aggregate_row_count={outputs.aggregate_row_count}")
    print(f"scores_by_detector_image={outputs.scores_by_detector_image_path}")
    print(f"errors_by_detector_image={outputs.errors_by_detector_image_path}")
    print(f"mota_idf1_motp_bubbles_image={outputs.mota_idf1_motp_bubbles_image_path}")


if __name__ == "__main__":
    main()
