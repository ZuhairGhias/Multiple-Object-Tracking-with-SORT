from dataclasses import dataclass

import cv2

from src.methods.detection.base import Detection
from src.methods.detection.mot import MOT17Detector
from src.methods.tracking import SORT
from src.methods.tracking.base import Track
from src.methods.tracking.mot_ground_truth import MOTGroundTruthTracker
from src.utils.mot17 import resolve_mot17_sequence_dir
from src.utils.render import render_detections, render_tracks

#TODO: refactor to use a common registry from utils.
DETECTORS = {
    "mot17": MOT17Detector,
}

TRACKERS = {
    "sort": SORT,
    "ground_truth": MOTGroundTruthTracker,
}

@dataclass
class FrameOutputs:
    detections: list[Detection]
    tracks: list[Track]

class FrameDebugger:
    """Utility for inspecting and debugging frame-wise inputs and outputs of MOT methods."""

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    def __init__(self, mot_seq: str, detector_name: str | None, tracker_name: str | None):
        self.frame_outputs = []
        self.sequence_dir = resolve_mot17_sequence_dir(mot_seq)
        self.mot_seq = self.sequence_dir.name
        self.frame_paths = sorted((self.sequence_dir / "img1").glob("*.jpg"))
        self.frame_outputs: list[FrameOutputs] = []

        self.detector = None
        if detector_name is not None:
            detector_cls = DETECTORS[detector_name]
            self.detector = detector_cls(
                sequence_id=self.mot_seq,
                root_dir=self.sequence_dir.parent,
        )
            
        self.tracker = None
        if tracker_name is not None:
            tracker_cls = TRACKERS[tracker_name]
            self.tracker = tracker_cls(
                sequence_id=self.mot_seq,
                root_dir=self.sequence_dir.parent,
            ) if tracker_name == "ground_truth" else tracker_cls()

        # Run detection and track generation to completion to retrieve later
        self._run()
    
    def _run(self) -> None:
        """
        Run tracking and detection for all the frames in the sequence and store the outputs for later.
        Safe to run again but should not be needed ever.
        """
        self.frame_outputs = []
        for frame_index, _ in enumerate(self.frame_paths, start=1):
            detections = []
            if self.detector is not None:
                detections = self.detector.get_detections(frame_index=frame_index)

            tracks = []
            if self.tracker is not None:
                tracks = self.tracker.update(detections, frame_index=frame_index)

            self.frame_outputs.append(
                FrameOutputs(
                    detections=detections,
                    tracks=tracks,
                )
            )

    def get_annotated_frame(self, frame_index: int) -> tuple[object | None, str]:
        """
        Get the frame at the index with annotations rendered for the detector or tracker if they were provided.
        Returns a tuple of (annotated_frame, status_message).
        If the frame index is out of bounds or the frame cannot be read, returns (None, error_message).
        """
        if frame_index < 1 or frame_index > self.frame_count:
            return None, f"Frame index {frame_index} is out of bounds. Expected 1 to {self.frame_count}."

        frame_path = self.frame_paths[frame_index - 1]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return None, f"Could not read frame {frame_index}: {frame_path}"

        outputs = self.frame_outputs[frame_index - 1]
        annotated_frame = frame

        if outputs.tracks:
            annotated_frame = render_tracks(annotated_frame, outputs.tracks)
        elif outputs.detections:
            annotated_frame = render_detections(annotated_frame, outputs.detections)

        

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        status = (
            f"Frame {frame_index} / {self.frame_count} "
            f"| detections: {len(outputs.detections)} "
            f"| tracks: {len(outputs.tracks)}"
        )
        return annotated_frame, status


