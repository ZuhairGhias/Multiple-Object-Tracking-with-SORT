from __future__ import annotations

import gradio as gr

from src.debugger.frame_debugger import FrameDebugger


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

    initial_left_frame, initial_right_frame, initial_status = load_raw_frame(1)

    with gr.Accordion("Frame Debugger", open=True):
        gr.Markdown(
            """
            Step through raw MOT17 frames directly from the local dataset. This is intentionally unannotated for now so we have a clean base layer for detections and tracks later.
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
