from __future__ import annotations
from dataclasses import dataclass

from filterpy.kalman import KalmanFilter

from .base import Track, Tracker
from src.methods.detection import Detection
from scipy.optimize import linear_sum_assignment
import numpy as np

@dataclass
class _KalmanTrack:
    def __init__(self, detection: Detection, track_id: int):
        self.track_id = track_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=7, dim_z=4) #TODO implement manually for better understanding

        # Motion model
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement model
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Initialize state with the first detection
        self.kf.x[:4] = self.xyxy_to_z(detection.as_xyxy())


    
    def predict_xyxy(self) -> tuple[float, float, float, float]:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.x_to_xyxy(self.kf.x)

    def update(self, detection: Detection) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self.xyxy_to_z(detection.as_xyxy()))

    def to_track(self, frame_index: int) -> Track:
        x1, y1, x2, y2 = self.x_to_xyxy(self.kf.x)
        return Track(
            track_id=self.track_id,
            frame_index=frame_index,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=None,
        )

    def x_to_xyxy(self, x: np.ndarray) -> tuple[float, float, float, float]:
        u, v, s, r = x[:4].reshape(-1)

        if s <= 0:
            s = 1e-6
        if r <= 0:
            r = 1e-6

        width = np.sqrt(s * r)
        height = s / width

        x1 = u - width / 2
        y1 = v - height / 2
        x2 = u + width / 2
        y2 = v + height / 2

        return (float(x1), float(y1), float(x2), float(y2))


    def xyxy_to_z(self, box: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            height = 1e-6

        u = x1 + width / 2
        v = y1 + height / 2
        s = width * height
        r = width / height

        return np.array([u, v, s, r]).reshape((4, 1))
    



class SORT(Tracker):
    """A simple SORT tracker implementation."""

    def __init__(self):
        super().__init__()
        self.iou_threshold = 0.3
        self.next_track_id = 1
        self.current_tracks: list[_KalmanTrack] = []
        self.max_age = 5

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
    ) -> list[Track]:

        """
        Prediction said: “I think the object is here.”
        Detection says: “I measured the object over here.”
        Kalman update combines both, weighted by uncertainty.

        :param self: Description
        :param detections: Description
        :type detections: list[Detection]
        :param frame_index: Description
        :type frame_index: int
        :return: Description
        :rtype: list[Track]
        """
        
        matches, _, unmatched_detections = self.match_detections(self.current_tracks, detections)

        # update kalman filters for matched tracks
        for track_idx, detection_idx in matches:
            self.current_tracks[track_idx].update(detections[detection_idx])

        # create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self.current_tracks.append(
                _KalmanTrack(
                    detection=detections[detection_idx],
                    track_id=self.next_track_id,
                )
            )
            self.next_track_id += 1

        # remove old tracks that haven't been matched for a while
        self.current_tracks = [
            tracker
            for tracker in self.current_tracks
            if tracker.time_since_update <= self.max_age
        ]

        return [
            tracker.to_track(frame_index)
            for tracker in self.current_tracks
        ]

    def match_detections(self, tracks: list[_KalmanTrack], detections: list[Detection]) -> list[tuple[int, int]]:

        # need to update predictions every time
        predicted_boxes = [
            track.predict_xyxy()
            for track in tracks
        ]
        
        # then decide if we should quit early
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # proceed with the matching
        cost_matrix = np.ones((len(tracks), len(detections)), dtype=np.float32)

        for i, predicted_box in enumerate(predicted_boxes):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - IOU(predicted_box, det.as_xyxy())

        row_indices, col_indices = linear_sum_assignment(cost_matrix) #TODO implement manually for better understanding

        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_detections = set(range(len(detections)))
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1 - self.iou_threshold:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_detections.discard(col)

        return matches, list(unmatched_tracks), list(unmatched_detections)

# TODO: put int common place
def IOU(boxA, boxB):
    """
    Docstring for IOU
    
    :param boxA: a tuple of (x1, y1, x2, y2) for the first box
    :param boxB: a tuple of (x1, y1, x2, y2) for the second box
    """
    interArea = Intersection(boxA, boxB)

    unionArea = Union(boxA, boxB, interArea)

    iou = interArea / unionArea if unionArea > 0 else 0

    return iou


def Union(boxA, boxB, interArea = None):
    """
    Docstring for Union
    
    :param boxA: a tuple of (x1, y1, x2, y2) for the first box
    :param boxB: a tuple of (x1, y1, x2, y2) for the second box
    :param interArea: the precomputer intersection area of the two boxes
    """
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if interArea is None:
        interArea = Intersection(boxA, boxB)

    unionArea = areaA + areaB - interArea

    return unionArea

def Intersection(boxA, boxB):
    """
    Docstring for Intersection
    
    :param boxA: a tuple of (x1, y1, x2, y2) for the first box
    :param boxB: a tuple of (x1, y1, x2, y2) for the second box
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    return interArea