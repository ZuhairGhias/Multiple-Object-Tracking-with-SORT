from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics.matcher import BoxLike, match_by_iou


@dataclass(frozen=True)
class MOTMetrics:
    mota: float # Multi-Object Tracking Accuracy. Less ID switches the better. (Tracking centric)
    motp: float # Multiple-Object Tracking Precision. More BB IoU the better. (Detector centric)
    idf1: float # F1 scores based on the assumption that the most commonly mapped ID is the TP.
    faf: float # False Alarms Per Frame
    mostly_tracked: int # Tracks that we mostly detected. (Detector centric)
    mostly_lost: int # Tracks that were mostly undetected. (Detector centric)
    false_positives: int # Unmatched predictions
    false_negatives: int # Unmatched ground truths
    id_switches: int # number of times a predicted ID assigned to a GT was different between frames.
    fragmentations: int
    matches: int
    ground_truth_count: int
    prediction_count: int
    frame_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "MOTA": self.mota,
            "MOTP": self.motp,
            "IDF1": self.idf1,
            "FAF": self.faf,
            "MT": self.mostly_tracked,
            "ML": self.mostly_lost,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "IDSW": self.id_switches,
            "Frag": self.fragmentations,
            "matches": self.matches,
            "ground_truth_count": self.ground_truth_count,
            "prediction_count": self.prediction_count,
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True)
class FrameAssociation:
    frame_index: int
    ground_truth_id: int
    prediction_id: int
    iou: float


def evaluate_mot_metrics(
    ground_truth_tracks: Iterable[BoxLike],
    predicted_tracks: Iterable[BoxLike],
    *,
    iou_threshold: float = 0.5,
    frame_count: int | None = None,
) -> MOTMetrics:
    """
    Evaluate common MOT scores from ground-truth and predicted tracks.

    Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
    Paper here:
    https://scispace.com/pdf/evaluating-multiple-object-tracking-performance-the-clear-tm8230vs80.pdf
    """


    # Group by frame for framewise comparisons
    ground_truth_by_frame = _group_by_frame(ground_truth_tracks)
    predictions_by_frame = _group_by_frame(predicted_tracks)
    frames = _frame_range(
        ground_truth_by_frame,
        predictions_by_frame,
        frame_count=frame_count,
    )

    total_ground_truth = sum(len(items) for items in ground_truth_by_frame.values())
    total_predictions = sum(len(items) for items in predictions_by_frame.values())
    total_matches = 0
    total_iou = 0.0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    previous_prediction_by_gt: dict[int, int] = {}
    associations: list[FrameAssociation] = []
    matched_frames_by_gt: dict[int, set[int]] = defaultdict(set)
    lifespan_frames_by_gt: dict[int, set[int]] = defaultdict(set)

    # Lifespans to calculate mostly tracked/lost
    for frame_index, ground_truth in ground_truth_by_frame.items():
        for gt in ground_truth:
            lifespan_frames_by_gt[gt.track_id].add(frame_index)

    for frame_index in frames:
        ground_truth = ground_truth_by_frame.get(frame_index, [])
        predictions = predictions_by_frame.get(frame_index, [])
        frame_matches = match_by_iou(
            ground_truth,
            predictions,
            iou_threshold=iou_threshold,
        )

        total_matches += len(frame_matches)
        total_iou += sum(match.iou for match in frame_matches)
        false_positives += len(predictions) - len(frame_matches)
        false_negatives += len(ground_truth) - len(frame_matches)

        # The main work of calculating ID switches
        # Essentially figuring out where the matched ID for a predicted box was different
        #   from the previous frame
        # TODO: would be nice to visualize the ID switches over time as tracks
        for match in frame_matches:
            gt = ground_truth[match.ground_truth_index]
            prediction = predictions[match.prediction_index]
            previous_prediction_id = previous_prediction_by_gt.get(gt.track_id)
            if previous_prediction_id is not None and previous_prediction_id != prediction.track_id:
                id_switches += 1
            previous_prediction_by_gt[gt.track_id] = prediction.track_id
            associations.append(
                FrameAssociation(
                    frame_index=frame_index,
                    ground_truth_id=gt.track_id,
                    prediction_id=prediction.track_id,
                    iou=match.iou,
                )
            )
            matched_frames_by_gt[gt.track_id].add(frame_index)

    mota = _safe_divide(
        total_ground_truth - false_negatives - false_positives - id_switches,
        total_ground_truth,
    )
    motp = _safe_divide(total_iou, total_matches)
    idf1 = _calculate_idf1(
        associations,
        total_ground_truth=total_ground_truth,
        total_predictions=total_predictions,
    )
    mostly_tracked, mostly_lost = _calculate_trajectory_coverage(
        lifespan_frames_by_gt,
        matched_frames_by_gt,
    )
    fragmentations = _calculate_fragmentations(
        lifespan_frames_by_gt,
        matched_frames_by_gt,
    )

    return MOTMetrics(
        mota=mota,
        motp=motp,
        idf1=idf1,
        faf=_safe_divide(false_positives, len(frames)),
        mostly_tracked=mostly_tracked,
        mostly_lost=mostly_lost,
        false_positives=false_positives,
        false_negatives=false_negatives,
        id_switches=id_switches,
        fragmentations=fragmentations,
        matches=total_matches,
        ground_truth_count=total_ground_truth,
        prediction_count=total_predictions,
        frame_count=len(frames),
    )


def compare_mot_metrics(
    ground_truth_tracks: Iterable[BoxLike],
    predictions_by_technique: Mapping[str, Iterable[BoxLike]],
    *,
    iou_threshold: float = 0.5,
    frame_count: int | None = None,
) -> dict[str, MOTMetrics]:
    """Evaluate multiple tracking techniques against one ground-truth set."""

    ground_truth = list(ground_truth_tracks)
    return {
        technique_name: evaluate_mot_metrics(
            ground_truth,
            predictions,
            iou_threshold=iou_threshold,
            frame_count=frame_count,
        )
        for technique_name, predictions in predictions_by_technique.items()
    }


def _group_by_frame(tracks: Iterable[BoxLike]) -> dict[int, list[BoxLike]]:
    tracks_by_frame: dict[int, list[BoxLike]] = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track.frame_index].append(track)
    return dict(tracks_by_frame)


def _frame_range(
    ground_truth_by_frame: dict[int, list[BoxLike]],
    predictions_by_frame: dict[int, list[BoxLike]],
    *,
    frame_count: int | None,
) -> list[int]:
    if frame_count is not None:
        return list(range(1, frame_count + 1))

    frame_indices = set(ground_truth_by_frame) | set(predictions_by_frame)
    if not frame_indices:
        return []
    return list(range(min(frame_indices), max(frame_indices) + 1))

def _calculate_idf1(
        associations: list[FrameAssociation],
        *,
        total_ground_truth: int,
        total_predictions: int,
) -> float:
    """
    Calculate identity F1 from framewise GT/prediction associations.

    >>> _calculate_idf1(
    ...     [
    ...         FrameAssociation(1, 1, 10, 1.0),
    ...         FrameAssociation(2, 1, 10, 1.0),
    ...     ],
    ...     total_ground_truth=2,
    ...     total_predictions=2,
    ... )
    1.0

    >>> _calculate_idf1(
    ...     [
    ...         FrameAssociation(1, 1, 10, 1.0),
    ...         FrameAssociation(2, 1, 11, 1.0),
    ...     ],
    ...     total_ground_truth=2,
    ...     total_predictions=2,
    ... )
    0.5

    >>> _calculate_idf1(
    ...     [
    ...         FrameAssociation(1, 1, 10, 1.0),
    ...         FrameAssociation(1, 2, 11, 1.0),
    ...         FrameAssociation(2, 1, 10, 1.0),
    ...         FrameAssociation(2, 2, 11, 1.0),
    ...         FrameAssociation(3, 1, 11, 1.0),
    ...         FrameAssociation(3, 2, 10, 1.0),
    ...     ],
    ...     total_ground_truth=6,
    ...     total_predictions=6,
    ... )
    0.6666666666666666
    """
    if not associations:
        return 0.0

    gt_ids = sorted({association.ground_truth_id for association in associations})
    prediction_ids = sorted({association.prediction_id for association in associations})
    gt_index_by_id = {track_id: index for index, track_id in enumerate(gt_ids)}
    prediction_index_by_id = {
        track_id: index
        for index, track_id in enumerate(prediction_ids)
    }

    pair_counts = np.zeros((len(gt_ids), len(prediction_ids)), dtype=np.int32)
    for association in associations:
        pair_counts[
            gt_index_by_id[association.ground_truth_id],
            prediction_index_by_id[association.prediction_id],
        ] += 1

    # In this context the TP is the ID we matched to most often across the video
    # We lock this ID mapping and count the frames that got it right
    gt_indices, prediction_indices = linear_sum_assignment(-pair_counts)
    identity_true_positives = int(
        pair_counts[gt_indices, prediction_indices].sum()
    )
    identity_false_positives = total_predictions - identity_true_positives
    identity_false_negatives = total_ground_truth - identity_true_positives

    return _safe_divide(
        2 * identity_true_positives,
        2 * identity_true_positives + identity_false_positives + identity_false_negatives,
    )


def _calculate_trajectory_coverage(
    lifespan_frames_by_gt: dict[int, set[int]],
    matched_frames_by_gt: dict[int, set[int]],
) -> tuple[int, int]:
    mostly_tracked = 0
    mostly_lost = 0
    for gt_id, lifespan_frames in lifespan_frames_by_gt.items():
        coverage = _safe_divide(
            len(matched_frames_by_gt.get(gt_id, set())),
            len(lifespan_frames),
        )
        if coverage >= 0.8:
            mostly_tracked += 1
        elif coverage < 0.2:
            mostly_lost += 1

    return mostly_tracked, mostly_lost


def _calculate_fragmentations(
    lifespan_frames_by_gt: dict[int, set[int]],
    matched_frames_by_gt: dict[int, set[int]],
) -> int:
    fragmentations = 0
    for gt_id, lifespan_frames in lifespan_frames_by_gt.items():
        matched_frames = matched_frames_by_gt.get(gt_id, set())
        was_matched = False
        matched_runs = 0
        for frame_index in sorted(lifespan_frames):
            is_matched = frame_index in matched_frames
            if is_matched and not was_matched:
                matched_runs += 1
            was_matched = is_matched
        fragmentations += max(0, matched_runs - 1)

    return fragmentations


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
