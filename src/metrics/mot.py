"""Project-level MOT metric calculations for framewise tracker outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.metrics.matcher import BoxLike, Match, iou, match_by_iou


@dataclass(frozen=True)
class MOTMetrics:
    """Summary counts and ratios produced by :func:`evaluate_mot_metrics`.

    `motp` is reported here as mean IoU over accepted matches. The data model
    also carries the raw counts needed by CSV reporting and artifact plots.
    """

    mota: float
    motp: float
    idf1: float
    faf: float
    mostly_tracked: int
    mostly_lost: int
    false_positives: int
    false_negatives: int
    id_switches: int
    fragmentations: int
    matches: int
    ground_truth_count: int
    prediction_count: int
    frame_count: int

    def as_dict(self) -> dict[str, float | int]:
        """Return stable display/report labels for this metric bundle."""

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
    """One accepted frame-level GT/prediction pairing used for IDF1."""

    frame_index: int
    ground_truth_id: int
    prediction_id: int
    iou: float


def evaluate_mot_metrics(
    ground_truth_tracks: Iterable[BoxLike],
    predicted_tracks: Iterable[BoxLike],
    *,
    ignored_tracks: Iterable[BoxLike] = (),
    iou_threshold: float = 0.5,
    frame_count: int | None = None,
) -> MOTMetrics:
    """Evaluate the project's MOT summary metrics.

    Evaluation is performed frame by frame. Accepted matches come from IoU
    assignment, then the accumulated matches and misses are used to compute:
    MOTA, mean-IoU MOTP, IDF1, FAF, MT, ML, FP, FN, ID switches, and
    fragmentations.
    """

    ground_truth_by_frame = _group_by_frame(ground_truth_tracks)
    predictions_by_frame = _group_by_frame(predicted_tracks)
    ignored_by_frame = _group_by_frame(ignored_tracks)
    frames = _frame_range(
        ground_truth_by_frame,
        predictions_by_frame,
        frame_count=frame_count,
    )

    total_ground_truth = sum(len(items) for items in ground_truth_by_frame.values())
    total_predictions = 0
    total_matches = 0
    total_iou = 0.0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    previous_prediction_by_gt: dict[int, int] = {}
    previous_frame_prediction_by_gt: dict[int, int] = {}
    associations: list[FrameAssociation] = []
    matched_frames_by_gt: dict[int, set[int]] = defaultdict(set)
    lifespan_frames_by_gt: dict[int, set[int]] = defaultdict(set)

    # Ground-truth lifespans support MT, ML, and fragmentation calculations.
    for frame_index, ground_truth in ground_truth_by_frame.items():
        for gt in ground_truth:
            lifespan_frames_by_gt[gt.track_id].add(frame_index)

    for frame_index in frames:
        ground_truth = ground_truth_by_frame.get(frame_index, [])
        predictions = predictions_by_frame.get(frame_index, [])
        frame_matches = _match_by_iou_with_continuity(
            ground_truth,
            predictions,
            previous_frame_prediction_by_gt,
            iou_threshold=iou_threshold,
        )
        matched_prediction_indices = {
            match.prediction_index
            for match in frame_matches
        }
        ignored_prediction_indices = _match_ignored_predictions(
            ignored_by_frame.get(frame_index, []),
            predictions,
            matched_prediction_indices,
            iou_threshold=iou_threshold,
        )
        evaluated_prediction_count = len(predictions) - len(ignored_prediction_indices)

        total_predictions += evaluated_prediction_count
        total_matches += len(frame_matches)
        total_iou += sum(match.iou for match in frame_matches)
        false_positives += evaluated_prediction_count - len(frame_matches)
        false_negatives += len(ground_truth) - len(frame_matches)

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
        previous_frame_prediction_by_gt = {
            ground_truth[match.ground_truth_index].track_id: predictions[match.prediction_index].track_id
            for match in frame_matches
        }

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
    ignored_tracks: Iterable[BoxLike] = (),
    iou_threshold: float = 0.5,
    frame_count: int | None = None,
) -> dict[str, MOTMetrics]:
    """Evaluate several tracker outputs against the same GT timeline."""

    ground_truth = list(ground_truth_tracks)
    ignored = list(ignored_tracks)
    return {
        technique_name: evaluate_mot_metrics(
            ground_truth,
            predictions,
            ignored_tracks=ignored,
            iou_threshold=iou_threshold,
            frame_count=frame_count,
        )
        for technique_name, predictions in predictions_by_technique.items()
    }


def _group_by_frame(tracks: Iterable[BoxLike]) -> dict[int, list[BoxLike]]:
    """Group tracker rows by frame index for framewise assignment."""

    tracks_by_frame: dict[int, list[BoxLike]] = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track.frame_index].append(track)
    return dict(tracks_by_frame)


def _match_ignored_predictions(
    ignored_regions: list[BoxLike],
    predictions: list[BoxLike],
    matched_prediction_indices: set[int],
    *,
    iou_threshold: float,
) -> set[int]:
    """Return unmatched prediction indices that overlap ignored GT regions."""

    unmatched_prediction_indices = [
        prediction_index
        for prediction_index in range(len(predictions))
        if prediction_index not in matched_prediction_indices
    ]
    if not ignored_regions or not unmatched_prediction_indices:
        return set()

    unmatched_predictions = [
        predictions[prediction_index]
        for prediction_index in unmatched_prediction_indices
    ]
    ignored_matches = match_by_iou(
        ignored_regions,
        unmatched_predictions,
        iou_threshold=iou_threshold,
    )
    return {
        unmatched_prediction_indices[match.prediction_index]
        for match in ignored_matches
    }


def _match_by_iou_with_continuity(
    ground_truth: list[BoxLike],
    predictions: list[BoxLike],
    previous_frame_prediction_by_gt: dict[int, int],
    *,
    iou_threshold: float,
) -> list[Match]:
    """Match one frame while preserving valid previous-frame assignments.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Box:
    ...     track_id: int
    ...     frame_index: int
    ...     box: tuple[float, float, float, float]
    ...     def as_xyxy(self):
    ...         return self.box
    >>> gt = [Box(1, 2, (0, 0, 10, 10)), Box(2, 2, (8, 0, 18, 10))]
    >>> pred = [Box(10, 2, (7, 0, 17, 10)), Box(11, 2, (1, 0, 11, 10))]
    >>> matches = _match_by_iou_with_continuity(
    ...     gt,
    ...     pred,
    ...     {1: 10, 2: 11},
    ...     iou_threshold=0.1,
    ... )
    >>> [(gt[m.ground_truth_index].track_id, pred[m.prediction_index].track_id) for m in matches]
    [(1, 10), (2, 11)]
    """

    if not ground_truth or not predictions:
        return []

    gt_index_by_id = {
        gt.track_id: gt_index
        for gt_index, gt in enumerate(ground_truth)
    }
    prediction_index_by_id = {}
    for prediction_index, prediction in enumerate(predictions):
        prediction_index_by_id.setdefault(prediction.track_id, prediction_index)

    matches: list[Match] = []
    matched_gt_indices: set[int] = set()
    matched_prediction_indices: set[int] = set()

    for gt_id, prediction_id in previous_frame_prediction_by_gt.items():
        gt_index = gt_index_by_id.get(gt_id)
        prediction_index = prediction_index_by_id.get(prediction_id)
        if gt_index is None or prediction_index is None:
            continue

        match_iou = iou(
            ground_truth[gt_index].as_xyxy(),
            predictions[prediction_index].as_xyxy(),
        )
        if match_iou < iou_threshold:
            continue

        matches.append(
            Match(
                ground_truth_index=gt_index,
                prediction_index=prediction_index,
                iou=match_iou,
            )
        )
        matched_gt_indices.add(gt_index)
        matched_prediction_indices.add(prediction_index)

    remaining_gt_indices = [
        gt_index
        for gt_index in range(len(ground_truth))
        if gt_index not in matched_gt_indices
    ]
    remaining_prediction_indices = [
        prediction_index
        for prediction_index in range(len(predictions))
        if prediction_index not in matched_prediction_indices
    ]
    remaining_matches = match_by_iou(
        [ground_truth[gt_index] for gt_index in remaining_gt_indices],
        [predictions[prediction_index] for prediction_index in remaining_prediction_indices],
        iou_threshold=iou_threshold,
    )
    matches.extend(
        Match(
            ground_truth_index=remaining_gt_indices[match.ground_truth_index],
            prediction_index=remaining_prediction_indices[match.prediction_index],
            iou=match.iou,
        )
        for match in remaining_matches
    )
    return matches


def _frame_range(
    ground_truth_by_frame: dict[int, list[BoxLike]],
    predictions_by_frame: dict[int, list[BoxLike]],
    *,
    frame_count: int | None,
) -> list[int]:
    """Return the evaluation timeline in frame order."""

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

    # IDTP comes from the best global one-to-one identity assignment.
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
    """Count GT trajectories tracked for >=80% or <20% of their lifespan."""

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
    """Count interrupted match coverage for each ground-truth trajectory."""

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
    """Divide while returning zero for undefined empty-denominator cases."""

    if denominator == 0:
        return 0.0
    return numerator / denominator
