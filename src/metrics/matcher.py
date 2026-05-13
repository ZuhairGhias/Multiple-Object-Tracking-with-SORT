from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment


class BoxLike(Protocol):
    """Object with an id, frame index, and bounding box."""

    track_id: int
    frame_index: int

    def as_xyxy(self) -> tuple[float, float, float, float]:
        ...


@dataclass(frozen=True)
class Match:
    ground_truth_index: int
    prediction_index: int
    iou: float


def match_by_iou(
    ground_truth: list[BoxLike],
    predictions: list[BoxLike],
    *,
    iou_threshold: float,
) -> list[Match]:
    """Match one frame of ground truth and predictions by maximum IoU."""

    if not ground_truth or not predictions:
        return []

    cost_matrix = np.ones((len(ground_truth), len(predictions)), dtype=np.float32)
    for gt_index, gt_box in enumerate(ground_truth):
        for pred_index, pred_box in enumerate(predictions):
            cost_matrix[gt_index, pred_index] = 1 - iou(
                gt_box.as_xyxy(),
                pred_box.as_xyxy(),
            )

    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
    matches: list[Match] = []
    for gt_index, pred_index in zip(gt_indices, pred_indices):
        match_iou = 1 - float(cost_matrix[gt_index, pred_index])
        if match_iou >= iou_threshold:
            matches.append(
                Match(
                    ground_truth_index=int(gt_index),
                    prediction_index=int(pred_index),
                    iou=match_iou,
                )
            )

    return matches


def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    intersection = intersection_area(box_a, box_b)
    union = area(box_a) + area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
