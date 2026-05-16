"""Dormant original-project SORT variant for manual experiments."""

from __future__ import annotations

from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.methods.detection import Detection
from src.methods.tracking.base import Track, Tracker
from src.methods.tracking.SORT import box_iou, sort_x_to_xyxy, xyxy_to_sort_z


class _MySORTTrack:
    """Original project SORT-style track state."""

    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.hits = 1
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
        self.kf.x[:4] = xyxy_to_sort_z(detection.as_xyxy())

    def predict_xyxy(self) -> tuple[float, float, float, float]:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return sort_x_to_xyxy(self.kf.x)

    def update(self, detection: Detection) -> None:
        self.time_since_update = 0
        self.hits += 1
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


class MySORT(Tracker):
    """Original project SORT implementation kept out of metrics by default.

    This class is intentionally not exported from `src.methods.tracking` and
    not registered in the MOT17 metrics utilities. It remains here as a manual
    comparison point while `SORT` represents the paper-aligned implementation.
    """

    def __init__(self):
        super().__init__()
        self.iou_threshold = 0.3
        self.next_track_id = 1
        self.current_tracks: list[_MySORTTrack] = []
        self.max_age = 5

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
    ) -> list[Track]:
        matches, _, unmatched_detections = self.match_detections(
            self.current_tracks,
            detections,
        )

        for track_index, detection_index in matches:
            self.current_tracks[track_index].update(detections[detection_index])

        for detection_index in unmatched_detections:
            self.current_tracks.append(
                _MySORTTrack(
                    detection=detections[detection_index],
                    track_id=self.next_track_id,
                )
            )
            self.next_track_id += 1

        self.current_tracks = [
            track
            for track in self.current_tracks
            if track.time_since_update <= self.max_age
        ]

        return [
            track.to_track(frame_index)
            for track in self.current_tracks
        ]

    def match_detections(
        self,
        tracks: list[_MySORTTrack],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        predicted_boxes = [
            track.predict_xyxy()
            for track in tracks
        ]

        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)
        for track_index, predicted_box in enumerate(predicted_boxes):
            for detection_index, detection in enumerate(detections):
                cost_matrix[track_index, detection_index] = 1 - box_iou(
                    predicted_box,
                    detection.as_xyxy(),
                )

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1 - self.iou_threshold:
                matches.append((int(row), int(col)))
                unmatched_tracks.discard(row)
                unmatched_detections.discard(col)

        return matches, list(unmatched_tracks), list(unmatched_detections)
