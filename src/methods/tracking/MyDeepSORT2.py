"""Project DeepSORT variant using color histograms and IoU association."""

from __future__ import annotations

import numpy as np

from src.metrics.matcher import iou
from src.methods.detection import Detection
from src.methods.tracking.deep_SORT import (
    ENCODER_COLOR_HISTOGRAM,
    DeepSORT,
    _DeepSortTrack,
    nearest_cosine_distance,
)


DEFAULT_IOU_APPEARANCE_WEIGHT = 0.5


class MyDeepSORT2(DeepSORT):
    """DeepSORT variant with color-histogram features and IoU-gated matching.

    This is the single project variant kept after baseline `DeepSORT`. It keeps
    DeepSORT's Kalman prediction, matching cascade, feature gallery, and track
    lifecycle. The two intentional changes are:

    - appearance is an HSV color histogram instead of the fixed CNN descriptor
    - cascade association is gated by SORT-style box IoU instead of Mahalanobis
    """

    def __init__(self, *, appearance_weight: float = DEFAULT_IOU_APPEARANCE_WEIGHT):
        super().__init__(encoder_name=ENCODER_COLOR_HISTOGRAM)
        self.appearance_weight = appearance_weight

    def build_association_cost_matrix(
        self,
        tracks: list[_DeepSortTrack],
        detections: list[Detection],
        features: list[np.ndarray],
    ) -> np.ndarray:
        """Return IoU-gated color-appearance costs for cascade matching."""

        return build_iou_appearance_cost_matrix(
            tracks,
            detections,
            features,
            iou_threshold=self.iou_threshold,
            appearance_weight=self.appearance_weight,
        )


def build_iou_appearance_cost_matrix(
    tracks: list[_DeepSortTrack],
    detections: list[Detection],
    features: list[np.ndarray],
    *,
    iou_threshold: float,
    appearance_weight: float,
) -> np.ndarray:
    """Return SORT-style IoU-gated costs blended with appearance distance.

    `appearance_weight=0.0` makes the cost pure SORT-style IoU distance.
    `appearance_weight=1.0` uses IoU only as a gate and appearance as the cost.

    >>> detection = Detection(frame_index=1, x1=0, y1=0, x2=10, y2=20)
    >>> track = _DeepSortTrack(detection, np.array([1.0, 0.0]), track_id=1)
    >>> costs = build_iou_appearance_cost_matrix(
    ...     [track],
    ...     [detection, detection],
    ...     [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    ...     iou_threshold=0.3,
    ...     appearance_weight=0.5,
    ... )
    >>> costs.tolist()
    [[0.0, 0.5]]
    """

    if len(detections) != len(features):
        raise ValueError("Feature count must match detection count.")
    if not 0.0 <= appearance_weight <= 1.0:
        raise ValueError("Appearance weight must be between 0 and 1.")

    cost_matrix = np.full((len(tracks), len(detections)), np.inf, dtype=np.float32)

    for track_index, track in enumerate(tracks):
        track_box = track.to_xyxy()
        for detection_index, detection in enumerate(detections):
            overlap = iou(track_box, detection.as_xyxy())
            if overlap < iou_threshold:
                continue

            appearance_distance = nearest_cosine_distance(
                track.features,
                features[detection_index],
            )
            iou_distance = 1.0 - overlap
            cost_matrix[track_index, detection_index] = (
                appearance_weight * appearance_distance
                + (1.0 - appearance_weight) * iou_distance
            )

    return cost_matrix
