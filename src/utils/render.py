from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _track_color(track_id: int) -> tuple[int, int, int]:
    # Deterministic color by track id so the same id stays visually stable.
    return (
        128 + (37 * track_id) % 128,
        128 + (17 * track_id + 85) % 128,
        128 + (29 * track_id + 170) % 128,
    )


def render_detections(frame, detections) -> Any:
    """
    Render detections on a copied frame.
    """

    rendered = np.copy(frame)
    for detection in detections:
        start_point = (int(detection.x1), int(detection.y1))
        end_point = (int(detection.x2), int(detection.y2))
        cv2.rectangle(rendered, start_point, end_point, (0, 255, 0), 2)
        label = f"{detection.score:.2f}"
        text_origin = (start_point[0], max(20, start_point[1] - 8))
        cv2.putText(
            rendered,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


def render_tracks(frame, tracks) -> Any:
    """
    Render tracks on a copied frame.
    """

    rendered = np.copy(frame)
    for track in tracks:
        color = _track_color(track.track_id)
        start_point = (int(track.x1), int(track.y1))
        end_point = (int(track.x2), int(track.y2))
        cv2.rectangle(rendered, start_point, end_point, color, 2)
        label = f"ID {track.track_id}"
        text_origin = (start_point[0], max(20, start_point[1] - 8))
        cv2.putText(
            rendered,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered
