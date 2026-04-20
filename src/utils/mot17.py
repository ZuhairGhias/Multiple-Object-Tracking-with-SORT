from __future__ import annotations

from argparse import ArgumentParser
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2

from src.methods.detection import MOT17Detector
from src.methods.tracking import MOTGroundTruthTracker, NaiveIOUTracker, SORT, Tracker
from src.utils.frames2mp4 import MP4Writer
from src.utils.render import render_detections, render_tracks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOT17_DIR = PROJECT_ROOT / "data" / "MOT17"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "temp"
TRACKING_VIDEO_FILENAMES = {
    "naive_iou": "tracking_naive_iou.mp4",
    "sort": "tracking_sort.mp4",
    "mot_ground_truth": "tracking_gt.mp4",
}


@dataclass(frozen=True)
class MOT17VideoOutputs:
    source_video_path: Path
    detections_video_path: Path
    tracking_naive_iou_video_path: Path
    tracking_sort_video_path: Path
    tracking_gt_video_path: Path | None = None


def infer_mot17_frame_rate(sequence_dir: str | Path) -> int:
    sequence_path = Path(sequence_dir)
    seqinfo_path = sequence_path / "seqinfo.ini"
    if not seqinfo_path.is_file():
        raise FileNotFoundError(f"Could not find MOT17 sequence config at '{seqinfo_path}'.")

    parser = ConfigParser()
    parser.read(seqinfo_path)
    try:
        return parser.getint("Sequence", "frameRate")
    except Exception as exc:
        raise ValueError(f"Could not read frameRate from '{seqinfo_path}'.") from exc


def _validate_sequence_dir(sequence_dir: str | Path) -> Path:
    resolved = Path(sequence_dir)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Could not find MOT17 sequence directory '{resolved}'.")
    if not (resolved / "img1").is_dir():
        raise FileNotFoundError(f"Could not find img1 directory under '{resolved}'.")
    if not (resolved / "seqinfo.ini").is_file():
        raise FileNotFoundError(f"Could not find seqinfo.ini under '{resolved}'.")
    if not (resolved / "det" / "det.txt").is_file():
        raise FileNotFoundError(f"Could not find det/det.txt under '{resolved}'.")
    return resolved


def resolve_mot17_sequence_dir(sequence: str | Path) -> Path:
    sequence_path = Path(sequence)
    if sequence_path.is_dir():
        return _validate_sequence_dir(sequence_path)

    requested_name = str(sequence).lower()
    for split in ("train", "test"):
        split_dir = DEFAULT_MOT17_DIR / split
        if not split_dir.is_dir():
            continue

        for sequence_dir in split_dir.iterdir():
            if sequence_dir.is_dir() and sequence_dir.name.lower() == requested_name:
                return _validate_sequence_dir(sequence_dir)

    raise FileNotFoundError(
        f"Could not find MOT17 sequence '{sequence}' under '{DEFAULT_MOT17_DIR}'."
    )


def generate_mot17_videos(
    sequence_dir: str | Path,
    *,
    output_dir: str | Path = DEFAULT_VIDEO_DIR,
) -> MOT17VideoOutputs:
    resolved_sequence_dir = resolve_mot17_sequence_dir(sequence_dir)
    sequence_id = resolved_sequence_dir.name
    sequence_output_dir = Path(output_dir) / sequence_id

    source_video_path = sequence_output_dir / "source.mp4"
    detections_video_path = sequence_output_dir / "detections.mp4"
    tracking_gt_video_path = sequence_output_dir / TRACKING_VIDEO_FILENAMES["mot_ground_truth"]
    tracking_naive_iou_video_path = sequence_output_dir / TRACKING_VIDEO_FILENAMES["naive_iou"]
    tracking_sort_video_path = sequence_output_dir / TRACKING_VIDEO_FILENAMES["sort"]

    fps = infer_mot17_frame_rate(resolved_sequence_dir)
    detector = MOT17Detector(sequence_id=sequence_id, root_dir=resolved_sequence_dir.parent)

    trackers: dict[str, Tracker] = {
        "naive_iou": NaiveIOUTracker(),
        "sort": SORT(),
    }
    gt_path = resolved_sequence_dir / "gt" / "gt.txt"
    if gt_path.is_file():
        trackers["mot_ground_truth"] = MOTGroundTruthTracker(
            sequence_id=sequence_id,
            root_dir=resolved_sequence_dir.parent,
        )
    else:
        warnings.warn(
            f"No gt/gt.txt found for '{sequence_id}'. tracking_gt.mp4 will not be generated.",
            stacklevel=2,
        )

    source_writer = MP4Writer(source_video_path, fps)
    detections_writer = MP4Writer(detections_video_path, fps)
    tracker_output_paths = {
        "naive_iou": tracking_naive_iou_video_path,
        "sort": tracking_sort_video_path,
        "mot_ground_truth": tracking_gt_video_path,
    }
    tracker_writers = {
        name: MP4Writer(tracker_output_paths[name], fps)
        for name in trackers
    }

    try:
        frame_paths = sorted((resolved_sequence_dir / "img1").glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No JPG frames found under '{resolved_sequence_dir / 'img1'}'.")

        for frame_index, frame_path in enumerate(frame_paths, start=1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise ValueError(f"Could not read frame '{frame_path}'.")

            source_writer.add_frame(frame.copy())

            detections = detector.get_detections(frame_index=frame_index)
            detections_writer.add_frame(render_detections(frame, detections))

            for name, tracker in trackers.items():
                tracks = tracker.update(detections, frame_index=frame_index)
                tracker_writers[name].add_frame(render_tracks(frame, tracks))
    finally:
        source_writer.close()
        detections_writer.close()
        for writer in tracker_writers.values():
            writer.close()

    return MOT17VideoOutputs(
        source_video_path=source_video_path,
        detections_video_path=detections_video_path,
        tracking_naive_iou_video_path=tracking_naive_iou_video_path,
        tracking_sort_video_path=tracking_sort_video_path,
        tracking_gt_video_path=tracking_gt_video_path if "mot_ground_truth" in trackers else None,
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Generate source, detections and ground-truth tracking MP4 files for a MOT17 sequence."
    )
    parser.add_argument(
        "sequence_dir",
        help="MOT17 sequence name or path to a sequence root containing img1/, seqinfo.ini and det/det.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_VIDEO_DIR),
        help="Base output directory for generated videos.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = generate_mot17_videos(
        args.sequence_dir,
        output_dir=args.output_dir,
    )
    print(f"source={outputs.source_video_path}")
    print(f"detections={outputs.detections_video_path}")
    print(f"tracking_naive_iou={outputs.tracking_naive_iou_video_path}")
    print(f"tracking_sort={outputs.tracking_sort_video_path}")
    if outputs.tracking_gt_video_path is not None:
        print(f"tracking_gt={outputs.tracking_gt_video_path}")


if __name__ == "__main__":
    main()
