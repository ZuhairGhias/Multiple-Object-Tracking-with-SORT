from __future__ import annotations

from pathlib import Path
import sys

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEQUENCE_ID = "MOT17-09-FRCNN"
TEMP_VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "temp"
CURATED_VIDEO_DIR = PROJECT_ROOT / "data" / "videos"

from src.ui.frame_inspector import build_frame_inspector


def get_sequence_video_path(sequence_id: str, video_name: str) -> Path:
    temp_path = TEMP_VIDEO_DIR / sequence_id / video_name
    if temp_path.is_file():
        return temp_path
    return CURATED_VIDEO_DIR / sequence_id / video_name


SOURCE_DEMO_VIDEO = get_sequence_video_path(DEFAULT_SEQUENCE_ID, "source.mp4")
DETECTION_DEMO_VIDEO = get_sequence_video_path(DEFAULT_SEQUENCE_ID, "detections.mp4")
NAIVE_IOU_DEMO_VIDEO = get_sequence_video_path(DEFAULT_SEQUENCE_ID, "tracking_naive_iou.mp4")
SORT_DEMO_VIDEO = get_sequence_video_path(DEFAULT_SEQUENCE_ID, "tracking_sort.mp4")
GROUND_TRUTH_DEMO_VIDEO = get_sequence_video_path(DEFAULT_SEQUENCE_ID, "tracking_gt.mp4")


demo_video_path = SOURCE_DEMO_VIDEO


VIDEO_OPTIONS = {
    "Source": SOURCE_DEMO_VIDEO,
    "Detections": DETECTION_DEMO_VIDEO,
    "SORT": SORT_DEMO_VIDEO,
    "Naive IoU": NAIVE_IOU_DEMO_VIDEO,
    "Ground Truth": GROUND_TRUTH_DEMO_VIDEO,
}


def get_video_path(video_option: str | None) -> str:
    return str(VIDEO_OPTIONS.get(video_option or "Source", SOURCE_DEMO_VIDEO))


with gr.Blocks(title="SORT MOT17 Demo") as demo:
    gr.Markdown(
        """
        # Multiple Object Tracking with SORT
        This first milestone is a demo-only MOT17 player. The app loads one configured local video and focuses on simple playback before tracking is added.
        """
    )
    video_option = gr.Radio(
        choices=list(VIDEO_OPTIONS),
        value="Source",
        label="Video Overlay",
        interactive=True,
    )
    gr.Video(
        value=get_video_path,
        inputs=video_option,
        label="MOT17 Demo Video",
        interactive=False,
        visible=bool(demo_video_path),
        autoplay=True,
        loop=True,
        include_audio=False,
        buttons=[]
    )

    build_frame_inspector(sequence_id=DEFAULT_SEQUENCE_ID)

    gr.Markdown(
        """
        ## Kalman Filter and SORT
        SORT uses Kalman filtering to model a linear trajectory for each tracked object.

        Imagine a car is traveling at a constant velocity v = 10m/s on a 1D line starting a position x = 0m. 
        As an engineer, it is convenient to assume then it's new position would after a second has elapsed would be x = 10m.
        However, in the real world there are many sources of noise and uncertainty that make this assumption inaccurate.

        An an engineer you may model the trajectory of a car as x_(t+1) = Ax_(t) where the column vector x represents the position and speed of the car.
        
        Kalman filtering is a two step process.

        * Step 1: Prediction using ideal model. x^a_(t+1) = Ax_(t) + u where u is perterbations to the ideal model.
        * Step 3: Update using measurements. x_(t+1) = x^a_(t+1) + 

        If want to learn more here is a great video: https://www.youtube.com/watch?v=IFeCIbljreY
        """
    )

    gr.Markdown(
        """
        ## Hungarian Algorithm
        """
    )


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
