"""DeepSORT tracker and association helpers.

The implementation follows the main DeepSORT flow: Kalman prediction,
Mahalanobis-gated appearance association, matching cascade, IoU fallback, and
track lifecycle management. The appearance encoder is a lightweight fixed CNN
baseline, not the learned ReID network from the paper.
"""

from __future__ import annotations

from enum import Enum

import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics.matcher import iou
from src.methods.detection import Detection
from src.methods.tracking.base import Track, Tracker


MAHALANOBIS_THRESHOLD = 9.4877
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_AGE = 30
DEFAULT_MAX_FEATURES = 100
DEFAULT_HITS_TO_CONFIRM = 3
DEFAULT_ASSOCIATION_LAMBDA = 0.0
DEEPSORT_STD_WEIGHT_POSITION = 1 / 20
DEEPSORT_STD_WEIGHT_VELOCITY = 1 / 160
DEFAULT_ENCODER_SIZE = (64, 128)
DEFAULT_CNN_POOL_SIZE = (4, 4)
DEFAULT_HISTOGRAM_BINS = (8, 8, 4)
ENCODER_SIMPLE_CNN = "cnn"
ENCODER_COLOR_HISTOGRAM = "color"
ENCODER_CNN_COLOR = "cnn_color"
DEFAULT_ENCODER_NAME = ENCODER_SIMPLE_CNN
ENCODER_NAMES = {
    ENCODER_SIMPLE_CNN,
    ENCODER_COLOR_HISTOGRAM,
    ENCODER_CNN_COLOR,
}
SIMPLE_CNN_KERNELS = np.array(
    [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
        [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
    dtype=np.float32,
)


class State(Enum):
    """Track lifecycle states used by DeepSORT."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"

class _DeepSortTrack:
    """Internal DeepSORT track state.

    The track stores Kalman state, lifecycle counters, and an appearance
    gallery of recent normalized features.
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

        # create Kalman filter
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

        measurement = xyxy_to_uvah(detection.as_xyxy())
        kf.x[:4] = measurement.reshape((4, 1))
        kf.P = initial_kalman_covariance(measurement)

    def to_uvah(self) -> np.ndarray:
        """Return the current Kalman box state as `(u, v, a, h)`."""

        return self.kf.x[:4].reshape(-1)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return the current Kalman box state as `(x1, y1, x2, y2)`."""

        return uvah_to_xyxy(self.to_uvah())

    def to_track(self, frame_index: int) -> Track:
        """Convert this internal track into the public `Track` type."""

        x1, y1, x2, y2 = self.to_xyxy()
        return Track(
            track_id=self.track_id,
            frame_index=frame_index,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=None,
        )

    # TODO: understand this better
    def project(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.kf.H @ self.kf.x
        covariance = (
            self.kf.H @ self.kf.P @ self.kf.H.T
            + measurement_kalman_covariance(mean.reshape(-1))
        )
        return mean.reshape(-1), covariance

    def predict(self) -> None:
        self.kf.Q = motion_kalman_covariance(self.to_uvah())
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
        self.kf.R = measurement_kalman_covariance(self.to_uvah())
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

    def mark_missed(self, *, max_age: int) -> None:
        if self.state == State.TENTATIVE:
            self.state = State.DELETED
        elif self.time_since_update > max_age:
            self.state = State.DELETED


class DeepSORT(Tracker):
    """A Deep SORT tracker implementation."""

    def __init__(self, *, encoder_name: str = DEFAULT_ENCODER_NAME):
        super().__init__()
        if encoder_name not in ENCODER_NAMES:
            known = ", ".join(sorted(ENCODER_NAMES))
            raise ValueError(f"Unknown DeepSORT encoder '{encoder_name}'. Expected one of: {known}.")

        self.encoder_name = encoder_name
        self.max_age = DEFAULT_MAX_AGE
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.mahalanobis_threshold = MAHALANOBIS_THRESHOLD
        self.lamb = DEFAULT_ASSOCIATION_LAMBDA
        self.max_features = DEFAULT_MAX_FEATURES
        self.hits_to_confirm = DEFAULT_HITS_TO_CONFIRM
        self.current_tracks: list[_DeepSortTrack] = []
        self.next_track_id = 1

    def create_track(
        self,
        detection: Detection,
        feature: np.ndarray,
        *,
        track_id: int,
    ) -> _DeepSortTrack:
        return _DeepSortTrack(detection, feature, track_id=track_id)

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> list[Track]:
        """Extract appearance features, then run the unfinished tracker update."""

        features = self.extract_normalized_features(frame, detections)

        for track in self.current_tracks:
            track.predict()

        # TODO: This works like hard priority assignment.
        #  We could try something softer, like blending staleness into the cost
        matches: list[tuple[int, int]] = []
        unmatched_track_indices_set = set(range(len(self.current_tracks)))
        unmatched_detection_indices = list(range(len(detections)))

        # Cascade over confirmed tracks, prioritizing the most current ones.
        confirmed_track_indices = {
            track_index
            for track_index, track in enumerate(self.current_tracks)
            if track.state == State.CONFIRMED
        }
        for time_since_update in range(1, self.max_age + 1):
            if not unmatched_track_indices_set or not unmatched_detection_indices:
                break

            track_indices_of_age = [
                track_index
                for track_index in sorted(unmatched_track_indices_set) # TODO: We shouldn't be sorting every loop
                if (
                    track_index in confirmed_track_indices
                    and self.current_tracks[track_index].time_since_update == time_since_update
                )
            ]

            if not track_indices_of_age:
                continue

            cost_matrix = self.build_association_cost_matrix(
                [self.current_tracks[track_index] for track_index in track_indices_of_age],
                [detections[detection_index] for detection_index in unmatched_detection_indices],
                [features[detection_index] for detection_index in unmatched_detection_indices],
            )

            # Remember that tracks were a subset, so we need to map them back
            level_matches_relative = linear_assignment_from_cost_matrix(cost_matrix)

            level_matches = [(track_indices_of_age[t], unmatched_detection_indices[d]) for (t, d) in level_matches_relative]

            matches.extend(level_matches)
            unmatched_track_indices_set -= set([t for (t, d) in level_matches])
            # slightly different handling for
            matched_detection_indices_set = set([d for (t, d) in level_matches])
            unmatched_detection_indices = [d for d in unmatched_detection_indices if d not in matched_detection_indices_set]

        iou_track_indices = [
            track_index
            for track_index in sorted(unmatched_track_indices_set)
            if (
                self.current_tracks[track_index].state == State.TENTATIVE
                or self.current_tracks[track_index].time_since_update == 1
            )
        ]
        iou_matches = match_by_iou(
            self.current_tracks,
            detections,
            track_indices=iou_track_indices,
            detection_indices=unmatched_detection_indices,
            iou_threshold=self.iou_threshold,
        )
        matches.extend(iou_matches)
        unmatched_track_indices_set -= {track_index for track_index, _ in iou_matches}
        matched_detection_indices_set = {detection_index for _, detection_index in iou_matches}
        unmatched_detection_indices = [
            detection_index
            for detection_index in unmatched_detection_indices
            if detection_index not in matched_detection_indices_set
        ]

        # Update matched tracks
        for track_index, detection_index in matches:
            self.current_tracks[track_index].update(
                detections[detection_index],
                features[detection_index],
                max_features=self.max_features,
                hits_to_confirm=self.hits_to_confirm,
            )

        # mark old tracks as missed
        for track_index in unmatched_track_indices_set:
            self.current_tracks[track_index].mark_missed(max_age=self.max_age)

        # create new tracks
        for detection_index in unmatched_detection_indices:
            self.current_tracks.append(
                self.create_track(
                    detections[detection_index],
                    features[detection_index],
                    track_id=self.next_track_id,
                )
            )
            self.next_track_id += 1

        # cleanup old tracks
        self.current_tracks = [
            track for track in self.current_tracks
            if track.state != State.DELETED
        ]

        # return only confirmed tracks
        return [
            track.to_track(frame_index)
            for track in self.current_tracks
            if track.state == State.CONFIRMED and track.time_since_update <= 1
        ]



    def extract_normalized_features(
        self,
        frame: np.ndarray | None,
        detections: list[Detection],
    ) -> list[np.ndarray]:
        """Crop detections from a frame and return one normalized feature each."""

        if frame is None:
            raise ValueError("A frame image is required to extract features.")

        features = []
        for detection in detections:
            crop = crop_frame_to_detection(frame, detection)
            feature = self.encode_crop(crop)
            features.append(normalize_feature(feature))
        return features

    def encode_crop(self, crop: np.ndarray) -> np.ndarray:
        """Encode one detection crop into an appearance feature.

        The default dependency-free baseline is a small fixed-weight
        convolutional descriptor. It is not the learned appearance descriptor
        from the DeepSORT paper. The return value does not need to be
        normalized; `extract_normalized_features` handles that boundary.
        """

        return encode_crop_feature(crop, encoder_name=self.encoder_name)

    def build_association_cost_matrix(
        self,
        tracks: list[_DeepSortTrack],
        detections: list[Detection],
        features: list[np.ndarray],
    ) -> np.ndarray:
        """Return Mahalanobis-gated appearance costs for cascade matching."""

        return build_association_cost_matrix(
            tracks,
            detections,
            features,
            lamb=self.lamb,
            mahalanobis_threshold=self.mahalanobis_threshold,
        )


def crop_frame_to_detection(frame: np.ndarray, detection: Detection) -> np.ndarray:
    """Crop one detection from an image frame with image-bound clamping.

    Fully out-of-frame detections fail fast instead of creating synthetic
    appearance features. MOT17 public detections should only hit this for
    malformed boxes, and the metrics utility should surface that data issue.

    >>> frame = np.arange(4 * 5).reshape(4, 5)
    >>> detection = Detection(frame_index=1, x1=1, y1=1, x2=4, y2=3)
    >>> crop_frame_to_detection(frame, detection).tolist()
    [[6, 7, 8], [11, 12, 13]]
    """

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.as_xyxy()

    x1 = max(0, min(width, int(round(x1))))
    x2 = max(0, min(width, int(round(x2))))
    y1 = max(0, min(height, int(round(y1))))
    y2 = max(0, min(height, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection crop has no area.")

    return frame[y1:y2, x1:x2]


def encode_crop_feature(crop: np.ndarray, *, encoder_name: str) -> np.ndarray:
    """Encode a crop with one of the configured appearance baselines.

    >>> encode_crop_feature(
    ...     np.full((4, 4, 3), 255, dtype=np.uint8),
    ...     encoder_name=ENCODER_CNN_COLOR,
    ... ).shape
    (384,)
    """

    if encoder_name == ENCODER_SIMPLE_CNN:
        return encode_simple_cnn_feature(crop)
    if encoder_name == ENCODER_COLOR_HISTOGRAM:
        return encode_color_histogram(crop)
    if encoder_name == ENCODER_CNN_COLOR:
        return np.concatenate(
            [
                normalize_feature(encode_simple_cnn_feature(crop)),
                normalize_feature(encode_color_histogram(crop)),
            ]
        ).astype(np.float32)

    known = ", ".join(sorted(ENCODER_NAMES))
    raise ValueError(f"Unknown DeepSORT encoder '{encoder_name}'. Expected one of: {known}.")


def encode_simple_cnn_feature(crop: np.ndarray) -> np.ndarray:
    """Encode a crop with a small fixed convolutional feature extractor.

    This is a lightweight CNN-style baseline: convolution kernels, rectified
    activations, and spatial pooling. The weights are fixed, not learned.

    >>> feature = encode_simple_cnn_feature(np.full((4, 4, 3), 255, dtype=np.uint8))
    >>> feature.shape
    (128,)
    >>> bool(np.isfinite(feature).all())
    True
    """

    bgr_crop = _as_bgr_crop(crop)
    resized_crop = cv2.resize(
        bgr_crop,
        DEFAULT_ENCODER_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    gray_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_crop /= 255.0

    pooled_maps = []
    for kernel in SIMPLE_CNN_KERNELS:
        response = cv2.filter2D(gray_crop, cv2.CV_32F, kernel)
        activation = np.maximum(response, 0.0)
        pooled = cv2.resize(
            activation,
            DEFAULT_CNN_POOL_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        pooled_maps.append(pooled.reshape(-1))

    feature = np.concatenate(pooled_maps).astype(np.float32)
    if not np.any(feature):
        feature[0] = 1.0
    return feature


def encode_color_histogram(crop: np.ndarray) -> np.ndarray:
    """Encode a crop as an HSV color histogram.

    The input is expected to use OpenCV image channel order when it has color
    channels. TODO: evaluate this color-bin encoder later as an appearance
    ablation against the default convolutional encoder.

    >>> feature = encode_color_histogram(np.full((4, 4, 3), 255, dtype=np.uint8))
    >>> feature.shape
    (256,)
    >>> float(feature.sum())
    8192.0
    """

    bgr_crop = _as_bgr_crop(crop)
    resized_crop = cv2.resize(
        bgr_crop,
        DEFAULT_ENCODER_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    hsv_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv_crop],
        channels=[0, 1, 2],
        mask=None,
        histSize=list(DEFAULT_HISTOGRAM_BINS),
        ranges=[0, 180, 0, 256, 0, 256],
    )
    return histogram.reshape(-1).astype(np.float32)


def _as_bgr_crop(crop: np.ndarray) -> np.ndarray:
    """Return a non-empty crop with three OpenCV BGR channels."""

    if crop.size == 0:
        raise ValueError("Cannot encode an empty crop.")
    if crop.ndim == 2:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if crop.ndim != 3 or crop.shape[2] not in (3, 4):
        raise ValueError("Crop must be grayscale, BGR, or BGRA.")
    if crop.shape[2] == 4:
        return crop[:, :, :3]
    return crop


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """Return an L2-normalized appearance feature.

    >>> normalize_feature(np.array([3.0, 4.0])).tolist()
    [0.6, 0.8]
    """

    norm = np.linalg.norm(feature)
    if norm == 0:
        raise ValueError("Appearance feature cannot be zero.")
    return feature / norm


def nearest_cosine_distance(
    gallery: list[np.ndarray],
    feature: np.ndarray,
) -> float:
    """Return the nearest cosine distance from a normalized gallery.

    Features are assumed to be normalized at extraction time.

    >>> nearest_cosine_distance([np.array([1.0, 0.0])], np.array([1.0, 0.0]))
    0.0
    >>> nearest_cosine_distance([np.array([1.0, 0.0])], np.array([0.0, 1.0]))
    1.0
    """

    if not gallery:
        return 1.0

    distances = [
        1 - float(np.dot(gallery_feature, feature))
        for gallery_feature in gallery
    ]
    return min(distances)


def append_normalized_feature(
    gallery: list[np.ndarray],
    feature: np.ndarray,
    *,
    max_features: int,
) -> list[np.ndarray]:
    """Append a normalized feature and retain the newest `max_features` items.

    >>> updated = append_normalized_feature(
    ...     [np.array([1]), np.array([2])],
    ...     np.array([3]),
    ...     max_features=2,
    ... )
    >>> [feature.tolist() for feature in updated]
    [[2], [3]]
    """

    return [*gallery, feature][-max_features:]


def initial_kalman_covariance(measurement: np.ndarray) -> np.ndarray:
    """Return DeepSORT-style initial covariance for `(u, v, a, h)`.

    >>> covariance = initial_kalman_covariance(np.array([5.0, 10.0, 0.5, 20.0]))
    >>> covariance.shape
    (8, 8)
    >>> bool(np.all(np.diag(covariance) > 0))
    True
    """

    height = _positive_height(measurement)
    standard_deviations = np.array(
        [
            2 * DEEPSORT_STD_WEIGHT_POSITION * height,
            2 * DEEPSORT_STD_WEIGHT_POSITION * height,
            1e-2,
            2 * DEEPSORT_STD_WEIGHT_POSITION * height,
            10 * DEEPSORT_STD_WEIGHT_VELOCITY * height,
            10 * DEEPSORT_STD_WEIGHT_VELOCITY * height,
            1e-5,
            10 * DEEPSORT_STD_WEIGHT_VELOCITY * height,
        ],
        dtype=np.float32,
    )
    return np.diag(np.square(standard_deviations))


def motion_kalman_covariance(measurement: np.ndarray) -> np.ndarray:
    """Return DeepSORT-style process noise for the current track height.

    >>> covariance = motion_kalman_covariance(np.array([5.0, 10.0, 0.5, 20.0]))
    >>> covariance.shape
    (8, 8)
    """

    height = _positive_height(measurement)
    standard_deviations = np.array(
        [
            DEEPSORT_STD_WEIGHT_POSITION * height,
            DEEPSORT_STD_WEIGHT_POSITION * height,
            1e-2,
            DEEPSORT_STD_WEIGHT_POSITION * height,
            DEEPSORT_STD_WEIGHT_VELOCITY * height,
            DEEPSORT_STD_WEIGHT_VELOCITY * height,
            1e-5,
            DEEPSORT_STD_WEIGHT_VELOCITY * height,
        ],
        dtype=np.float32,
    )
    return np.diag(np.square(standard_deviations))


def measurement_kalman_covariance(measurement: np.ndarray) -> np.ndarray:
    """Return DeepSORT-style measurement noise for `(u, v, a, h)`.

    >>> covariance = measurement_kalman_covariance(np.array([5.0, 10.0, 0.5, 20.0]))
    >>> covariance.shape
    (4, 4)
    """

    height = _positive_height(measurement)
    standard_deviations = np.array(
        [
            DEEPSORT_STD_WEIGHT_POSITION * height,
            DEEPSORT_STD_WEIGHT_POSITION * height,
            1e-1,
            DEEPSORT_STD_WEIGHT_POSITION * height,
        ],
        dtype=np.float32,
    )
    return np.diag(np.square(standard_deviations))


def _positive_height(measurement: np.ndarray) -> float:
    return max(float(measurement[:4].reshape(-1)[3]), 1e-6)


def xyxy_to_uvah(box: tuple[float, float, float, float]) -> np.ndarray:
    """Convert an `(x1, y1, x2, y2)` box into DeepSORT `(u, v, a, h)`.

    >>> xyxy_to_uvah((0, 0, 10, 20)).tolist()
    [5.0, 10.0, 0.5, 20.0]
    """

    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if height <= 0:
        height = 1e-6

    center_x = x1 + width / 2
    center_y = y1 + height / 2
    aspect_ratio = width / height
    return np.array([center_x, center_y, aspect_ratio, height], dtype=np.float32)


def uvah_to_xyxy(measurement: np.ndarray) -> tuple[float, float, float, float]:
    """Convert DeepSORT `(u, v, a, h)` state into `(x1, y1, x2, y2)`.

    >>> uvah_to_xyxy(np.array([5, 10, 0.5, 20]))
    (0.0, 0.0, 10.0, 20.0)
    """

    center_x, center_y, aspect_ratio, height = measurement[:4].reshape(-1)
    if height <= 0:
        height = 1e-6
    if aspect_ratio <= 0:
        aspect_ratio = 1e-6

    width = aspect_ratio * height
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return (float(x1), float(y1), float(x2), float(y2))


def mahalanobis_distance(
    projected_mean: np.ndarray,
    projected_covariance: np.ndarray,
    measurement: np.ndarray,
) -> float:
    """Return squared Mahalanobis distance from a projected Kalman state.

    >>> mahalanobis_distance(
    ...     np.array([0.0, 0.0]),
    ...     np.eye(2),
    ...     np.array([3.0, 4.0]),
    ... )
    25.0
    """

    delta = measurement.reshape(-1, 1) - projected_mean.reshape(-1, 1)
    solved = np.linalg.solve(projected_covariance, delta)
    return float((delta.T @ solved).item())


def gated_association_cost(
    *,
    mahalanobis: float,
    appearance: float,
    lamb: float,
    mahalanobis_threshold: float,
) -> float:
    """Combine motion and appearance costs, rejecting invalid motion gates.

    >>> gated_association_cost(
    ...     mahalanobis=2.0,
    ...     appearance=0.25,
    ...     lamb=0.0,
    ...     mahalanobis_threshold=9.4877,
    ... )
    0.25
    >>> np.isinf(gated_association_cost(
    ...     mahalanobis=10.0,
    ...     appearance=0.25,
    ...     lamb=0.0,
    ...     mahalanobis_threshold=9.4877,
    ... ))
    np.True_
    """

    if mahalanobis > mahalanobis_threshold:
        return float(np.inf)
    return lamb * mahalanobis + (1 - lamb) * appearance

def build_association_cost_matrix(
    tracks: list[_DeepSortTrack],
    detections: list[Detection],
    features: list[np.ndarray],
    *,
    lamb: float,
    mahalanobis_threshold: float,
) -> np.ndarray:
    """
    >>> detection = Detection(frame_index=1, x1=0, y1=0, x2=10, y2=20)
    >>> track = _DeepSortTrack(detection, np.array([1.0, 0.0]), track_id=1)
    >>> costs = build_association_cost_matrix(
    ...     [track],
    ...     [detection, detection],
    ...     [np.array([1.0, 0.0]), np.array([0.0, 0.1])],
    ...     lamb=0.0,
    ...     mahalanobis_threshold=9.4877,
    ... )
    >>> costs.shape
    (1, 2)
    >>> costs.tolist()
    [[0.0, 1.0]]
    """
    if len(detections) != len(features):
        raise ValueError("Feature count must match detection count.")

    cost_matrix = np.full((len(tracks), len(detections)), np.inf, dtype=np.float32)

    for track_index, track in enumerate(tracks):
        projected_mean, projected_covariance = track.project()

        for detection_index, detection in enumerate(detections):
            measurement = xyxy_to_uvah(detection.as_xyxy())

            motion_distance = mahalanobis_distance(
                projected_mean,
                projected_covariance,
                measurement
            )

            appearance_distance = nearest_cosine_distance(
                track.features,
                features[detection_index],
            )

            cost_matrix[track_index, detection_index] = gated_association_cost(
                mahalanobis=motion_distance,
                appearance=appearance_distance,
                lamb=lamb,
                mahalanobis_threshold=mahalanobis_threshold,
            )
    return cost_matrix


def match_by_iou(
    tracks: list[_DeepSortTrack],
    detections: list[Detection],
    *,
    track_indices: list[int],
    detection_indices: list[int],
    iou_threshold: float,
) -> list[tuple[int, int]]:
    """Match selected tracks and detections by IoU fallback.

    >>> detection = Detection(frame_index=1, x1=0, y1=0, x2=10, y2=20)
    >>> track = _DeepSortTrack(detection, np.array([1.0, 0.0]), track_id=1)
    >>> match_by_iou(
    ...     [track],
    ...     [detection],
    ...     track_indices=[0],
    ...     detection_indices=[0],
    ...     iou_threshold=0.3,
    ... )
    [(0, 0)]
    """

    if not track_indices or not detection_indices:
        return []

    cost_matrix = np.full(
        (len(track_indices), len(detection_indices)),
        np.inf,
        dtype=np.float32,
    )
    for relative_track_index, track_index in enumerate(track_indices):
        track_box = tracks[track_index].to_xyxy()
        for relative_detection_index, detection_index in enumerate(detection_indices):
            overlap = iou(track_box, detections[detection_index].as_xyxy())
            if overlap >= iou_threshold:
                cost_matrix[relative_track_index, relative_detection_index] = 1 - overlap

    relative_matches = linear_assignment_from_cost_matrix(cost_matrix)
    return [
        (track_indices[relative_track_index], detection_indices[relative_detection_index])
        for relative_track_index, relative_detection_index in relative_matches
    ]

def linear_assignment_from_cost_matrix(
    cost_matrix: np.ndarray,
) -> list[tuple[int, int]]:
    """
    >>> linear_assignment_from_cost_matrix(np.array([[0.2, 1.0], [1.0, 0.1]]))
    [(0, 0), (1, 1)]
    >>> linear_assignment_from_cost_matrix(np.array([[np.inf, 0.2]]))
    [(0, 1)]
    >>> linear_assignment_from_cost_matrix(np.array([[np.inf]]))
    []
    >>> linear_assignment_from_cost_matrix(np.array([[np.inf, 0.2], [np.inf, np.inf]]))
    [(0, 1)]
    >>> linear_assignment_from_cost_matrix(np.array([
    ...     [0.1, np.inf],
    ...     [0.2, np.inf],
    ... ]))
    [(0, 0)]
    >>> linear_assignment_from_cost_matrix(np.empty((0, 2)))
    []
    >>> linear_assignment_from_cost_matrix(np.empty((2, 0)))
    []
    """

    if not np.isfinite(cost_matrix).any():
        return []

    finite_costs = np.isfinite(cost_matrix)
    # Gating can leave whole rows or columns infeasible. Drop them before
    # Hungarian assignment so SciPy only sees a solvable subproblem.
    valid_rows = np.flatnonzero(finite_costs.any(axis=1))
    valid_cols = np.flatnonzero(finite_costs.any(axis=0))
    reduced_cost_matrix = cost_matrix[np.ix_(valid_rows, valid_cols)]

    finite_reduced_costs = np.isfinite(reduced_cost_matrix)
    unmatched_cost = float(np.max(reduced_cost_matrix[finite_reduced_costs]) + 1)
    blocked_cost = unmatched_cost + 1
    track_count, detection_count = reduced_cost_matrix.shape

    # Matching is partial: tracks and detections may remain unmatched. Dummy
    # rows/columns model that explicitly, while gated-out pairs stay blocked.
    assignment_size = track_count + detection_count
    assignment_costs = np.full(
        (assignment_size, assignment_size),
        blocked_cost,
        dtype=np.float32,
    )
    assignment_costs[:track_count, :detection_count] = np.where(
        finite_reduced_costs,
        reduced_cost_matrix,
        blocked_cost,
    )
    for track_index in range(track_count):
        assignment_costs[track_index, detection_count + track_index] = unmatched_cost
    for detection_index in range(detection_count):
        assignment_costs[track_count + detection_index, detection_index] = unmatched_cost
    assignment_costs[track_count:, detection_count:] = 0

    row_indices, col_indices = linear_sum_assignment(assignment_costs)

    matches = []
    for row, col in zip(row_indices, col_indices):
        if row >= track_count or col >= detection_count:
            continue

        original_row = int(valid_rows[row])
        original_col = int(valid_cols[col])
        if np.isinf(cost_matrix[original_row, original_col]):
            continue

        matches.append((original_row, original_col))

    return matches
