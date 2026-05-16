"""Dormant pre-performance DeepSORT variant for manual experiments."""

from __future__ import annotations

from filterpy.kalman import KalmanFilter
import numpy as np

from src.methods.detection import Detection
from src.methods.tracking.deep_SORT import (
    DeepSORT,
    ENCODER_SIMPLE_CNN,
    State,
    _DeepSortTrack,
    append_normalized_feature,
    xyxy_to_uvah,
)


class _MyDeepSORTTrack(_DeepSortTrack):
    """Pre-performance-branch DeepSORT track state.

    This restores the older track-level behavior from commit `10b72a8`:
    FilterPy's default Kalman covariance/noise values are used instead of the
    current DeepSORT-style height-scaled covariance tuning.
    """

    def __init__(
        self,
        detection: Detection,
        feature: np.ndarray,
        *,
        track_id: int,
    ) -> None:
        self.track_id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = State.TENTATIVE
        self.features = [feature]

        self.kf = kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        kf.x[:4] = xyxy_to_uvah(detection.as_xyxy()).reshape((4, 1))

    def project(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.kf.H @ self.kf.x
        covariance = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        return mean.reshape(-1), covariance

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        detection: Detection,
        feature: np.ndarray,
        *,
        max_features: int,
        hits_to_confirm: int,
    ) -> None:
        measurement = xyxy_to_uvah(detection.as_xyxy()).reshape((4, 1))
        self.kf.update(measurement)

        self.features = append_normalized_feature(
            self.features,
            feature,
            max_features=max_features,
        )

        self.hits += 1
        self.time_since_update = 0

        if self.state == State.TENTATIVE and self.hits >= hits_to_confirm:
            self.state = State.CONFIRMED


class MyDeepSORT(DeepSORT):
    """Pre-performance DeepSORT variant kept out of metrics by default.

    This class is intentionally not exported from `src.methods.tracking` and
    not registered in MOT17 metrics or plotting utilities.
    """

    def __init__(self):
        super().__init__(encoder_name=ENCODER_SIMPLE_CNN)

    def create_track(
        self,
        detection: Detection,
        feature: np.ndarray,
        *,
        track_id: int,
    ) -> _MyDeepSORTTrack:
        return _MyDeepSORTTrack(detection, feature, track_id=track_id)
