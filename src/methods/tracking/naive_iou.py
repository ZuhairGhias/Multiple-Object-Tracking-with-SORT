from __future__ import annotations

from src.methods.detection import Detection

from .base import Track, Tracker


class NaiveIOUTracker(Tracker):
    """Greedy IoU tracker kept as a simple reference baseline."""

    def __init__(self):
        super().__init__()
        self.iou_threshold = 0.3
        self.max_age = 5
        self.next_track_id = 1
        self.current_tracks: list[Track] = []

    def update(
        self,
        detections: list[Detection],
        *,
        frame_index: int,
    ) -> list[Track]:
        candidates: list[tuple[int, int, float]] = []
        for track_index, track in enumerate(self.current_tracks):
            for detection_index, detection in enumerate(detections):
                iou = _iou(track.as_xyxy(), detection.as_xyxy())
                if iou >= self.iou_threshold:
                    candidates.append((track_index, detection_index, iou))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for track_index, detection_index, _ in sorted(candidates, key=lambda item: item[2], reverse=True):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            detection = detections[detection_index]
            self.current_tracks[track_index] = Track(
                track_id=self.current_tracks[track_index].track_id,
                frame_index=frame_index,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                score=detection.score,
            )

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue

            self.current_tracks.append(
                Track(
                    track_id=self.next_track_id,
                    frame_index=frame_index,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    score=detection.score,
                )
            )
            self.next_track_id += 1

        self.current_tracks = [
            track
            for track in self.current_tracks
            if track.frame_index >= frame_index - self.max_age
        ]

        return list(self.current_tracks)


def _iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    intersection_area = _intersection_area(box_a, box_b)
    union_area = _area(box_a) + _area(box_b) - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
