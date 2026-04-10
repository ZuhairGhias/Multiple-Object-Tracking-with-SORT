from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Iterable

import cv2
import numpy as np

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except ImportError:  # pragma: no cover
    get_ffmpeg_exe = None


Frame = np.ndarray


def _ffmpeg_executable() -> str | None:
    if get_ffmpeg_exe is None:
        return None
    try:
        return get_ffmpeg_exe()
    except Exception:
        return None


class MP4Writer:
    """Incremental MP4 writer that prefers a final H.264 output."""

    def __init__(self, output_path: str | Path, fps: int | float) -> None:
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self._temp_output_path = self.output_path.with_name(f"{self.output_path.stem}.tmp.mp4")
        self._writer: cv2.VideoWriter | None = None
        self._frame_size: tuple[int, int] | None = None

    def add_frame(self, frame: Frame) -> None:
        if frame is None:
            raise ValueError("Cannot write an empty frame.")

        height, width = frame.shape[:2]
        frame_size = (width, height)
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._frame_size = frame_size
            self._writer = cv2.VideoWriter(
                str(self._temp_output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                frame_size,
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open output video writer for '{self._temp_output_path}'.")
        elif frame_size != self._frame_size:
            raise ValueError(
                f"All frames must share one size. Expected {self._frame_size}, got {frame_size}."
            )

        self._writer.write(frame)

    def close(self) -> Path:
        if self._writer is None:
            return self.output_path

        self._writer.release()
        self._writer = None

        ffmpeg_executable = _ffmpeg_executable()
        if ffmpeg_executable is None:
            self._temp_output_path.replace(self.output_path)
            return self.output_path

        command = [
            ffmpeg_executable,
            "-y",
            "-i",
            str(self._temp_output_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self._temp_output_path.unlink(missing_ok=True)
        return self.output_path


def frames2mp4(frames: Iterable[Frame], output_path: str | Path, fps: int | float) -> Path:
    """Write an iterable of frames to a mp4 file."""

    writer = MP4Writer(output_path=output_path, fps=fps)
    for frame in frames:
        writer.add_frame(frame)
    return writer.close()
