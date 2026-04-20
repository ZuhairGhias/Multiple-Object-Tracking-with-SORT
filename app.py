from __future__ import annotations

import base64
from pathlib import Path
import sys

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEQUENCE_ID = "MOT17-09-FRCNN"
TEMP_VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "temp"
CURATED_VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
EXAMPLE_IMAGE_DIR = PROJECT_ROOT / "data" / "images"

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

SORT_PREDICTION_EXAMPLES = [
    (
        "Frame 340",
        EXAMPLE_IMAGE_DIR / "MOT17-09-FRCNN_frame_000340_track_30_detections.png",
        EXAMPLE_IMAGE_DIR / "MOT17-09-FRCNN_frame_000340_track_30_sort.png",
    ),
    (
        "Frame 345",
        EXAMPLE_IMAGE_DIR / "MOT17-09-FRCNN_frame_000345_track_30_detections.png",
        EXAMPLE_IMAGE_DIR / "MOT17-09-FRCNN_frame_000345_track_30_sort.png",
    ),
]


demo_video_path = SOURCE_DEMO_VIDEO


VIDEO_OPTIONS = {
    "Source": SOURCE_DEMO_VIDEO,
    "Detections": DETECTION_DEMO_VIDEO,
    "Naive IoU": NAIVE_IOU_DEMO_VIDEO,
    "SORT": SORT_DEMO_VIDEO,
    "Ground Truth": GROUND_TRUTH_DEMO_VIDEO,
}


def get_video_path(video_option: str | None) -> str:
    return str(VIDEO_OPTIONS.get(video_option or "Source", SOURCE_DEMO_VIDEO))


def image_data_uri(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


def build_sort_prediction_examples_html() -> str:
    columns = []
    for example_label, detections_image, sort_image in SORT_PREDICTION_EXAMPLES:
        columns.append(
            f"""
            <div class="sort-example-frame">
              <div class="sort-example-title">{example_label}</div>
              <div class="sort-example-pair">
                <figure>
                  <img src="{image_data_uri(detections_image)}" alt="{example_label} detections">
                  <figcaption>Detections</figcaption>
                </figure>
                <figure>
                  <img src="{image_data_uri(sort_image)}" alt="{example_label} SORT prediction">
                  <figcaption>SORT</figcaption>
                </figure>
              </div>
            </div>
            """
        )

    return f"""
    <style>
      .sort-example-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        align-items: start;
      }}
      .sort-example-title {{
        font-weight: 600;
        margin: 0 0 6px;
      }}
      .sort-example-pair {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
      }}
      .sort-example-pair figure {{
        margin: 0;
      }}
      .sort-example-pair img {{
        display: block;
        width: 100%;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
      }}
      .sort-example-pair figcaption {{
        margin-top: 3px;
        font-size: 0.85rem;
        color: #666;
        text-align: center;
      }}
      @media (max-width: 760px) {{
        .sort-example-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    <div class="sort-example-grid">
      {''.join(columns)}
    </div>
    """


with gr.Blocks(title="SORT MOT17 Demo") as demo:
    gr.Markdown(
        """
        # Multiple Object Tracking with SORT
        **WIP - Study of the SORT family of object tracking**
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
        ## Naive IoU Tracking
        For my first attempt at tracking I used a naive approach where I assigned current tracks to the detections
        with the most IoU (greater than a threshold).
        
        You can see from the demo above that this actually works very well for cases where the person being tracked
        is visible and consistently detected.
        
        There are many flaws with this approach, but one of the primary ones we will fix with SORT is when detections 
        are inconsistent. Because we use the overlap with the previous position of the track, if an object goes a few
        frames without being detected, the IoU when it finally is detected will be off by a lot.       
        """
    )

    gr.Markdown(
        """
        ## SORT tracking with predicted trajectories (via Kalman Filters)
        SORT uses Kalman filtering to model a linear trajectory for each tracked object.
        These cropped examples show the detector pane next to the SORT pane for the same region. When the detector stops producing a box for track 30, SORT can still keep the track alive briefly by predicting where the person should be.
        """
    )
    gr.HTML(build_sort_prediction_examples_html())

    gr.Markdown(
        r"""
        ## Kalman Filters
        Imagine a car is traveling at a constant velocity v = 10m/s on a 1D line starting a position x = 0m. 
        As an engineer, it is convenient to assume then it's new position would after a second has elapsed would be x = 10m.
        However, your GPS measurement tells you that the car has only travelled 9.6m. 
        in the real world there are many sources of noise and uncertainty that make both theoretical and measured
        systems inaccurate.
        
        Kalman filtering fixes this with a two step process.

        * Step 1: Prediction using the ideal model.

        $$
        x^a_{t+1} = A x_t + u
        $$
        
        where \(u\) represents perturbations to the ideal model.
        
        * Step 2: Update using measurements.
        
        $$
        x_{t+1} = x^a_{t+1} + K(z_{t+1} - Hx^a_{t+1})
        $$

        If want to learn more here is a great video: https://www.youtube.com/watch?v=IFeCIbljreY
        """
    )

    gr.Markdown(
        """
        ## Hungarian Algorithm
        TODO
        """
    )


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
