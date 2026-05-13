"""Public metrics API used by MOT17 scoring utilities."""

from src.metrics.matcher import BoxLike, Match, iou, match_by_iou
from src.metrics.mot import MOTMetrics, compare_mot_metrics, evaluate_mot_metrics

__all__ = [
    "BoxLike",
    "MOTMetrics",
    "Match",
    "compare_mot_metrics",
    "evaluate_mot_metrics",
    "iou",
    "match_by_iou",
]
