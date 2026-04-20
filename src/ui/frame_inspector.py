from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr

from src.debugger.frame_debugger import FrameDebugger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MAX_SAVED_IMAGE_DIMENSION = 1280


def build_frame_inspector(*, sequence_id: str) -> None:
    debugger_error = None
    detection_debugger = None
    sort_debugger = None
    try:
        detection_debugger = FrameDebugger(sequence_id, detector_name="mot17", tracker_name=None)
        sort_debugger = FrameDebugger(sequence_id, detector_name="mot17", tracker_name="sort")
    except Exception as exc:
        debugger_error = f"FrameDebugger unavailable: {exc}"

    frame_count = detection_debugger.frame_count if detection_debugger is not None else 0

    def clamp_frame_index(frame_index: int | float | None) -> int:
        if frame_count == 0:
            return 1
        if frame_index is None:
            return 1
        return max(1, min(int(frame_index), frame_count))

    def load_raw_frame(frame_index: int | float | None) -> tuple[object | None, object | None, str]:
        if frame_count == 0:
            return None, None, debugger_error or f"No frames found for {sequence_id}."

        selected_frame_index = clamp_frame_index(frame_index)
        if detection_debugger is not None and sort_debugger is not None:
            detections_frame, detections_status = detection_debugger.get_annotated_frame(selected_frame_index)
            sort_frame, sort_status = sort_debugger.get_annotated_frame(selected_frame_index)
            return detections_frame, sort_frame, f"{detections_status} | {sort_status}"

        return None, None, debugger_error or f"No debugger output available for frame {selected_frame_index}."

    def step_raw_frame(frame_index: int | float | None, delta: int) -> tuple[int, object | None, object | None, str]:
        selected_frame_index = clamp_frame_index(clamp_frame_index(frame_index) + delta)
        left_frame, right_frame, status = load_raw_frame(selected_frame_index)
        return selected_frame_index, left_frame, right_frame, status

    def downsample_image(image: object):
        height, width = image.shape[:2]
        scale = min(1.0, MAX_SAVED_IMAGE_DIMENSION / max(width, height))
        if scale == 1.0:
            return image
        return cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    def write_rgb_image(path: Path, image: object) -> None:
        resized_image = downsample_image(image)
        bgr_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr_image):
            raise RuntimeError(f"Could not write image to {path}.")

    def save_debug_frames(
        frame_index: int | float | None,
        track_id: int | float | None,
        padding: int | float | None,
    ) -> str:
        if frame_count == 0:
            return debugger_error or f"No frames found for {sequence_id}."
        if detection_debugger is None or sort_debugger is None:
            return debugger_error or "No debugger output available."

        selected_frame_index = clamp_frame_index(frame_index)
        detections_frame, detections_status = detection_debugger.get_annotated_frame(selected_frame_index)
        sort_frame, sort_status = sort_debugger.get_annotated_frame(selected_frame_index)
        if detections_frame is None or sort_frame is None:
            return f"Could not save frame {selected_frame_index}: {detections_status} | {sort_status}"

        IMAGES_DIR.mkdir(exist_ok=True)
        frame_prefix = f"{sequence_id}_frame_{selected_frame_index:06d}"
        saved_detections_frame = detections_frame
        saved_sort_frame = sort_frame
        filename_suffix = ""
        if track_id is not None:
            selected_track_id = int(track_id)
            selected_padding = max(0, int(padding or 0))
            outputs = sort_debugger.frame_outputs[selected_frame_index - 1]
            selected_track = next(
                (track for track in outputs.tracks if track.track_id == selected_track_id),
                None,
            )
            if selected_track is None:
                return f"Track ID {selected_track_id} was not found in frame {selected_frame_index}."

            image_height, image_width = sort_frame.shape[:2]
            x1 = max(0, int(selected_track.x1) - selected_padding)
            y1 = max(0, int(selected_track.y1) - selected_padding)
            x2 = min(image_width, int(selected_track.x2) + selected_padding)
            y2 = min(image_height, int(selected_track.y2) + selected_padding)
            if x1 >= x2 or y1 >= y2:
                return f"Track ID {selected_track_id} has an invalid crop box."

            saved_detections_frame = detections_frame[y1:y2, x1:x2]
            saved_sort_frame = sort_frame[y1:y2, x1:x2]
            filename_suffix = f"_track_{selected_track_id}"

        detections_path = IMAGES_DIR / f"{frame_prefix}{filename_suffix}_detections.png"
        sort_path = IMAGES_DIR / f"{frame_prefix}{filename_suffix}_sort.png"

        try:
            write_rgb_image(detections_path, saved_detections_frame)
            write_rgb_image(sort_path, saved_sort_frame)
        except RuntimeError as exc:
            return str(exc)

        saved_paths = [detections_path.relative_to(PROJECT_ROOT), sort_path.relative_to(PROJECT_ROOT)]

        return "Saved: " + ", ".join(str(path) for path in saved_paths)

    initial_left_frame, initial_right_frame, initial_status = load_raw_frame(1)

    with gr.Accordion("Frame Debugger", open=False):
        gr.Markdown(
            """
            Step through raw MOT17 frames directly from the local dataset.
            """
        )
        gr.Markdown(
            """
            **This inspector only works when the MOT17 dataset is checked out locally under `data/MOT17`.**
            """
        )
        frame_slider = gr.Slider(
            minimum=1,
            maximum=max(frame_count, 1),
            value=1,
            step=1,
            label="Frame",
            interactive=frame_count > 0,
        )
        with gr.Row():
            previous_frame = gr.Button("Previous", interactive=frame_count > 0)
            next_frame = gr.Button("Next", interactive=frame_count > 0)
        with gr.Row():
            track_id_input = gr.Number(label="Track ID", precision=0)
            crop_padding_input = gr.Number(value=32, label="Crop Padding", precision=0)
            save_frames = gr.Button("Save Frames", interactive=frame_count > 0)
        raw_frame_status = gr.Markdown(value=initial_status)
        with gr.Row():
            left_frame = gr.Image(
                value=initial_left_frame,
                label=f"{sequence_id} Detections",
                interactive=False,
            )
            right_frame = gr.Image(
                value=initial_right_frame,
                label=f"{sequence_id} SORT Tracks",
                interactive=False,
            )

        frame_slider.release(
            load_raw_frame,
            inputs=frame_slider,
            outputs=[left_frame, right_frame, raw_frame_status],
            show_progress="hidden",
        )
        previous_frame.click(
            lambda frame_index: step_raw_frame(frame_index, -1),
            inputs=frame_slider,
            outputs=[frame_slider, left_frame, right_frame, raw_frame_status],
            show_progress="hidden",
        )
        next_frame.click(
            lambda frame_index: step_raw_frame(frame_index, 1),
            inputs=frame_slider,
            outputs=[frame_slider, left_frame, right_frame, raw_frame_status],
            show_progress="hidden",
        )
        save_frames.click(
            save_debug_frames,
            inputs=[frame_slider, track_id_input, crop_padding_input],
            outputs=raw_frame_status,
            show_progress="hidden",
        )
