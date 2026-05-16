"""SORT tracker aligned with the original paper/reference implementation."""

from __future__ import annotations

from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.methods.detection import Detection
from src.methods.tracking.base import Track, Tracker


SORT_MAX_AGE = 1
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3


class _SortTrack:
    """Internal SORT track state.

    SORT uses a seven-dimensional constant-velocity Kalman state:
    `(u, v, s, r, du, dv, ds)`, where `u/v` are box center coordinates,
    `s` is area, and `r` is aspect ratio. The reference implementation keeps
    aspect ratio fixed and only predicts center and scale velocity.
    """

    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = xyxy_to_sort_z(detection.as_xyxy())

    def predict(self) -> tuple[float, float, float, float]:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return sort_x_to_xyxy(self.kf.x)

    def update(self, detection: Detection) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(xyxy_to_sort_z(detection.as_xyxy()))

    def to_track(self, frame_index: int) -> Track:
        x1, y1, x2, y2 = sort_x_to_xyxy(self.kf.x)
        return Track(
            track_id=self.track_id,
            frame_index=frame_index,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=None,
        )


class SORT(Tracker):
    """SORT tracker using Kalman prediction and IoU/Hungarian association.

    This class intentionally follows the original SORT lifecycle: predictions
    can remain alive briefly for future association, but only tracks updated in
    the current frame are emitted. New tracks are emitted during the initial
    warm-up frames; afterward they must reach `min_hits` consecutive matches.

    >>> tracker = SORT()
    >>> detection = Detection(frame_index=1, x1=0, y1=0, x2=10, y2=10)
    >>> len(tracker.update([detection], frame_index=1))
    1
    >>> tracker.update([], frame_index=2)
    []
    >>> len(tracker.current_tracks)
    1
    >>> tracker.update([], frame_index=3)
    []
    >>> len(tracker.current_tracks)
    0
    """

    def __init__(
        self,
        *,
        max_age: int = SORT_MAX_AGE,
        min_hits: int = SORT_MIN_HITS,
        iou_threshold: float = SORT_IOU_THRESHOLD,
    ):
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.next_track_id = 1
        self.current_tracks: list[_SortTrack] = []

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
    ) -> list[Track]:
        self.frame_count += 1
        predicted_boxes, invalid_track_indices = self.predict_tracks()
        for track_index in reversed(invalid_track_indices):
            self.current_tracks.pop(track_index)

        matches, _, unmatched_detections = match_sort_detections(
            predicted_boxes,
            detections,
            iou_threshold=self.iou_threshold,
        )

        for track_index, detection_index in matches:
            self.current_tracks[track_index].update(detections[detection_index])

        for detection_index in unmatched_detections:
            self.current_tracks.append(
                _SortTrack(
                    detection=detections[detection_index],
                    track_id=self.next_track_id,
                )
            )
            self.next_track_id += 1

        output_tracks = [
            track.to_track(frame_index)
            for track in self.current_tracks
            if (
                track.time_since_update < 1
                and (
                    track.hit_streak >= self.min_hits
                    or self.frame_count <= self.min_hits
                )
            )
        ]
        self.current_tracks = [
            track
            for track in self.current_tracks
            if track.time_since_update <= self.max_age
        ]
        return output_tracks

    def predict_tracks(self) -> tuple[list[tuple[float, float, float, float]], list[int]]:
        predicted_boxes = []
        invalid_track_indices = []
        for track_index, track in enumerate(self.current_tracks):
            predicted_box = track.predict()
            if not np.isfinite(predicted_box).all():
                invalid_track_indices.append(track_index)
                continue
            predicted_boxes.append(predicted_box)
        return predicted_boxes, invalid_track_indices


def match_sort_detections(
    predicted_boxes: list[tuple[float, float, float, float]],
    detections: list[Detection],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match SORT predictions to detections using IoU and Hungarian assignment.

    >>> detections = [Detection(frame_index=1, x1=0, y1=0, x2=10, y2=10)]
    >>> match_sort_detections([(0, 0, 10, 10)], detections, iou_threshold=0.3)
    ([(0, 0)], [], [])
    """

    if len(predicted_boxes) == 0 or len(detections) == 0:
        return [], list(range(len(predicted_boxes))), list(range(len(detections)))

    cost_matrix = np.ones((len(predicted_boxes), len(detections)), dtype=np.float32)
    for track_index, predicted_box in enumerate(predicted_boxes):
        for detection_index, detection in enumerate(detections):
            cost_matrix[track_index, detection_index] = 1 - box_iou(
                predicted_box,
                detection.as_xyxy(),
            )

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(len(predicted_boxes)))
    unmatched_detections = set(range(len(detections)))
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= 1 - iou_threshold:
            matches.append((int(row), int(col)))
            unmatched_tracks.discard(row)
            unmatched_detections.discard(col)

    return matches, list(unmatched_tracks), list(unmatched_detections)


def xyxy_to_sort_z(box: tuple[float, float, float, float]) -> np.ndarray:
    """Convert `(x1, y1, x2, y2)` into SORT's `(u, v, s, r)` measurement.

    >>> xyxy_to_sort_z((0, 0, 10, 20)).reshape(-1).tolist()
    [5.0, 10.0, 200.0, 0.5]
    """

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if height <= 0:
        height = 1e-6
    u = x1 + width / 2
    v = y1 + height / 2
    scale = width * height
    ratio = width / height
    return np.array([u, v, scale, ratio]).reshape((4, 1))


def sort_x_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert SORT's Kalman state into `(x1, y1, x2, y2)`.

    >>> sort_x_to_xyxy(np.array([5, 10, 200, 0.5, 0, 0, 0]))
    (0.0, 0.0, 10.0, 20.0)
    """

    u, v, scale, ratio = state[:4].reshape(-1)
    if scale <= 0:
        scale = 1e-6
    if ratio <= 0:
        ratio = 1e-6

    width = np.sqrt(scale * ratio)
    height = scale / width
    return (
        float(u - width / 2),
        float(v - height / 2),
        float(u + width / 2),
        float(v + height / 2),
    )


def box_iou(
    first_box: tuple[float, float, float, float],
    second_box: tuple[float, float, float, float],
) -> float:
    """Return intersection-over-union for two `(x1, y1, x2, y2)` boxes.

    >>> box_iou((0, 0, 10, 10), (5, 5, 15, 15))
    0.14285714285714285
    """

    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
