from __future__ import annotations

import asyncio
import base64
from pathlib import Path
import sys

# Gradio video changes can make the browser cancel the previous file request.
# On Windows, the default Proactor loop logs those normal disconnects noisily.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import gradio as gr


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEQUENCE_ID = "MOT17-09-SDP"
TEMP_VIDEO_DIR = PROJECT_ROOT / "data" / "videos" / "temp"
CURATED_VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
EXAMPLE_IMAGE_DIR = PROJECT_ROOT / "data" / "images"
METRIC_IMAGE_DIR = EXAMPLE_IMAGE_DIR / "metrics"
METRICS_CSV = PROJECT_ROOT / "data" / "metrics" / "MOT17_tracking_metrics.csv"

from src.ui.frame_inspector import build_frame_inspector


def get_sequence_video_path(sequence_id: str, video_name: str) -> Path:
    temp_path = TEMP_VIDEO_DIR / sequence_id / video_name
    if temp_path.is_file():
        return temp_path
    return CURATED_VIDEO_DIR / sequence_id / video_name


VIDEO_FILENAMES = {
    "Source": "source.mp4",
    "Detections": "detections.mp4",
    "Naive IoU": "tracking_naive_iou.mp4",
    "SORT": "tracking_sort.mp4",
    "DeepSORT": "tracking_deep_sort.mp4",
    "MyDeepSORT2": "tracking_my_deep_sort2.mp4",
    "Ground Truth": "tracking_gt.mp4",
}
METRIC_GRAPH_OPTIONS = {
    "Scores": METRIC_IMAGE_DIR / "MOT17_tracking_trackers-naive_iou_sort_deepsort_mydeepsort2_detectors-all_metric_scores_by_detector.png",
    "Errors": METRIC_IMAGE_DIR / "MOT17_tracking_trackers-naive_iou_sort_deepsort_mydeepsort2_detectors-all_metric_errors_by_detector.png",
    "IDF1 / MOTA": METRIC_IMAGE_DIR / "MOT17_tracking_trackers-naive_iou_sort_deepsort_mydeepsort2_detectors-all_mota_idf1_motp_bubbles.png",
    "Performance": METRIC_IMAGE_DIR / "MOT17_tracking_trackers-naive_iou_sort_deepsort_mydeepsort2_detectors-all_performance_by_detector.png",
}

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


def discover_video_sequences() -> list[str]:
    sequence_ids = set()
    for video_dir in (CURATED_VIDEO_DIR, TEMP_VIDEO_DIR):
        if not video_dir.is_dir():
            continue
        for sequence_dir in video_dir.iterdir():
            if sequence_dir.is_dir() and any(sequence_dir.glob("*.mp4")):
                sequence_ids.add(sequence_dir.name)
    sequence_ids = {
        sequence_id
        for sequence_id in sequence_ids
        if all(
            get_sequence_video_path(sequence_id, video_name).is_file()
            for video_name in VIDEO_FILENAMES.values()
        )
    }
    if DEFAULT_SEQUENCE_ID in sequence_ids:
        return [DEFAULT_SEQUENCE_ID, *sorted(sequence_ids - {DEFAULT_SEQUENCE_ID})]
    return sorted(sequence_ids)


VIDEO_SEQUENCE_OPTIONS = discover_video_sequences()
DEFAULT_VIDEO_SEQUENCE = (
    DEFAULT_SEQUENCE_ID
    if DEFAULT_SEQUENCE_ID in VIDEO_SEQUENCE_OPTIONS
    else VIDEO_SEQUENCE_OPTIONS[0]
    if VIDEO_SEQUENCE_OPTIONS
    else DEFAULT_SEQUENCE_ID
)


def get_video_path(sequence_id: str | None, video_option: str | None) -> str | None:
    sequence = sequence_id or DEFAULT_VIDEO_SEQUENCE
    video_name = VIDEO_FILENAMES.get(video_option or "Source", VIDEO_FILENAMES["Source"])
    video_path = get_sequence_video_path(sequence, video_name)
    if video_path.is_file():
        return str(video_path)
    return None


def get_metric_image_path(graph_option: str | None) -> str | None:
    graph_path = METRIC_GRAPH_OPTIONS.get(graph_option or "Scores")
    if graph_path is not None and graph_path.is_file():
        return str(graph_path)
    return None


def build_metrics_status_text() -> str:
    missing_graphs = [
        label
        for label, image_path in METRIC_GRAPH_OPTIONS.items()
        if not image_path.is_file()
    ]
    if not missing_graphs and METRICS_CSV.is_file():
        return "Metrics CSV and plot images are available from the latest generated report."

    instructions = [
        "Metrics artifacts are missing.",
        "Run `python -m src.utils.mot17_metrics` and then `python -m src.utils.mot17_metrics_plots`.",
    ]
    if missing_graphs:
        instructions.append(f"Missing plots: {', '.join(missing_graphs)}.")
    if not METRICS_CSV.is_file():
        instructions.append("Missing CSV: `data/metrics/MOT17_tracking_metrics.csv`.")
    return " ".join(instructions)


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
        # Multiple Object Tracking on MOT17 with SORT and DeepSORT
        **Study of the SORT family of object tracking algorithms**

        **Author:** Zuhair Ghias

        **Email:** zuhairg2@illinois.edu

        **Source code:** https://github.com/ZuhairGhias/Multiple-Object-Tracking-with-SORT

        **Demo:** https://huggingface.co/spaces/zghias/Multiple-Object-Tracking-with-SORT

        Multi-object tracking (MOT) is the problem of detecting multiple objects in video and maintaining
        consistent identities for them over time.

        In a typical tracking-by-detection (TBD) system, this is split into two stages:

        1. **Detector**: classifies objects and predicts bounding boxes in each frame.
        2. **Tracker**: associates detections across frames into object trajectories.

        This report focuses on the second stage: the tracker. I use the MOT17 public detections as fixed
        detector outputs, then compare several tracking strategies on top of the same detection inputs.
        The progression starts with a naive IoU-only association baseline, then moves through SORT, DeepSORT,
        and a lightweight custom DeepSORT variant.

        The main questions are:

        1. How much do motion prediction and appearance features improve identity consistency?
        2. What runtime cost is introduced by each additional modeling component?

        """
    )

    gr.Markdown("## MOT17 Video Examples")
    video_sequence = gr.Dropdown(
        choices=VIDEO_SEQUENCE_OPTIONS,
        value=DEFAULT_VIDEO_SEQUENCE,
        label="Sequence",
        interactive=True,
    )
    video_option = gr.Radio(
        choices=list(VIDEO_FILENAMES),
        value="Source",
        label="Video",
        interactive=True,
    )
    gr.Video(
        value=get_video_path,
        inputs=[video_sequence, video_option],
        format="mp4",
        label="Video",
        interactive=False,
    )

    gr.Markdown("## Aggregate Metric Plots")
    gr.Markdown(build_metrics_status_text())
    graph_option = gr.Radio(
        choices=list(METRIC_GRAPH_OPTIONS),
        value="Scores",
        label="Graph",
        interactive=True,
    )
    gr.Image(
        value=get_metric_image_path,
        inputs=graph_option,
        label="MOT17 Metrics Graph",
        interactive=False,
    )

    build_frame_inspector(sequence_id=DEFAULT_SEQUENCE_ID)

    gr.Markdown(
        """
        ## Naive IoU-Only Association Baseline
        For my first attempt at tracking, I used a naive approach that matches current tracks to detections
        using intersection-over-union (IoU). The tracker considers all track/detection pairs above an IoU
        threshold and greedily assigns the highest-overlap pairs first.

        You can see from the demo above that this actually works very well for cases where the person being tracked
        is visible and consistently detected.

        For this baseline, I used a generous IoU threshold of 0.3 and a relatively short track lifetime of 5 frames.
        After that, an unassigned track is discarded.

        There are many flaws with this approach, but one of the primary ones we will fix with SORT is handling
        inconsistent detections. The naive tracker keeps unmatched tracks alive briefly, but it does not predict where
        the object should move. Because association still uses overlap with the last observed box, if an object goes a
        few frames without being detected, the IoU can be very small when it finally reappears.
        """
    )

    gr.Markdown(
        """
        ## MOT Evaluation Metrics

        This is a good time to talk about the key metrics we care about in tracking algorithms.
        The MOT17 training data provides both ground-truth bounding boxes and ground-truth object IDs for each frame.

        At a high level, the goal of a tracker is to:

        1. Keep the same ID assigned to the same object for as long as possible.
        2. Avoid creating false tracks for objects that are not really there.
        3. Avoid missing objects that should have been tracked.

        ### MOTA

        MOTA, or multiple object tracking accuracy, summarizes three common tracking errors:

        $$
        \\text{MOTA} = 1 - \\frac{\\text{FN} + \\text{FP} + \\text{IDSW}}{\\text{GT}}
        $$

        Where:

        - **FN** is the number of missed ground-truth objects.
        - **FP** is the number of false predicted objects.
        - **IDSW** is the number of identity switches.
        - **GT** is the total number of ground-truth object instances.

        MOTA is useful because it gives one overall error score. However, it is also heavily affected by detector quality.
        If the detector misses many objects, the tracker will inherit many false negatives.

        ### IDF1

        IDF1 focuses more directly on identity consistency. It measures how well predicted track identities match
        ground-truth identities over time.
        """
    )

    gr.HTML(
        f"""
        <figure>
          <img src="{image_data_uri(EXAMPLE_IMAGE_DIR / 'web' / 'idf1-miguel-mendez.webp')}" alt="IDF1 diagram" style="max-width: 100%; height: auto;">
          <figcaption style="font-size: 0.9rem; color: #666;">
            Source: https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics
          </figcaption>
        </figure>
        """
    )

    gr.Markdown(
        """
        A tracker can have reasonable box overlap while still performing poorly on IDF1 if it frequently changes IDs
        for the same person. For this report, IDF1 is especially important because the main difference between SORT
        and DeepSORT is how they try to preserve identity through occlusion, missed detections, and ambiguous assignments.

        <details>
        <summary>Other MOT metrics used in this report</summary>

        - **MOTP**: measures the localization precision of matched boxes.
        - **FAF**: false alarms per frame.
        - **MT**: mostly tracked trajectories, where a ground-truth target is tracked for most of its lifespan.
        - **ML**: mostly lost trajectories, where a ground-truth target is missed for most of its lifespan.
        - **FP**: false positives, or predicted boxes that do not match a ground-truth object.
        - **FN**: false negatives, or ground-truth objects that were missed.
        - **IDSW**: identity switches, where a ground-truth object changes assigned tracker ID.
        - **Frag**: fragmentations, where a ground-truth trajectory is interrupted and later recovered.

        </details>

        The common visualization you will see for tracking quality plots IDF1 against MOTA. In this report, I also
        include runtime plots because real-time performance is part of the motivation for SORT-style methods.
        """

    )

    gr.Markdown(
        """
        ### Naive IoU Quantitative Results

        The first aggregate plot shows the naive IoU tracker by detector. This gives us a baseline before adding
        motion prediction or appearance features. The horizontal axis is MOTA, the vertical axis is IDF1, and the
        bubble size scales with MOTP.

        The detector is important because the tracker can only associate the detections it is given. In these results,
        SDP is consistently the strongest detector. I keep all three detectors in the early plots to show that effect,
        but later comparisons will focus on SDP once the graphs become too crowded.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_detectors-all_mota_idf1_motp_bubbles.png"
        ),
        label="Naive IoU IDF1-MOTA Plot",
        interactive=False,
    )
    gr.Markdown(
        """
        The full set of aggregate metric values are shown below as a table.

        | Detector | Tracker | MOTA | IDF1 | MOTP | FP | FN | IDSW | Frag | ms/frame |
        |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
        | DPM | Naive IoU | 11.9% | 26.3% | 77.1% | 25063 | 72203 | **1668** | 3067 | **0.22** |
        | FRCNN | Naive IoU | 38.7% | 45.8% | **87.5%** | **15033** | 52067 | 1777 | **1476** | 0.32 |
        | SDP | Naive IoU | **49.1%** | **50.8%** | 83.0% | 19026 | **35647** | 2493 | 3064 | 0.48 |
        """
    )

    gr.Markdown(
        """
        ## Runtime Evaluation

        Tracking quality is not the only objective. SORT-style trackers are also motivated by real-time performance,
        so I track runtime alongside the MOT accuracy metrics.

        For this report, I use milliseconds per frame as the main runtime number. This is easier to compare than
        total runtime because the MOT17 sequences have different lengths. I also report mean active tracks and
        predictions per frame, since those help explain why some tracker/detector combinations take more time than
        others.

        At this stage, the naive IoU tracker is expected to be very fast because it does not run a motion model or
        appearance encoder. Its main cost is computing IoU scores between active tracks and current detections.
        """
    )

    gr.Markdown(
        """
        ## SORT: Kalman Prediction with IoU Association
        Unlike my naive method, SORT adds a motion model for each tracked object. Instead of only comparing a new
        detection to the last observed box, SORT first predicts where the object should be in the next frame using a
        Kalman filter.

        After that, it still uses IoU to compare predicted boxes against the current detections. The main difference is
        that the box being matched is now a prediction instead of just the last place the object was seen.

        The cropped examples below show the detector pane next to the SORT pane for the same region. When the detector
        stops producing a box for track 30, SORT can still keep the track alive briefly by predicting where the person
        should be.
        """
    )
    gr.HTML(build_sort_prediction_examples_html())

    gr.Markdown(
        r"""
        ## Kalman Filtering for Track State Estimation
        Imagine a car is traveling at a constant velocity \(v = 10m/s\) on a 1D line starting at position \(x = 0m\).
        If we trust the motion model completely, then after one second we would predict the car to be at \(x = 10m\).

        However, suppose the GPS measurement says the car is at \(x = 9.6m\). The model and the measurement disagree,
        but neither one is perfect. The model assumes constant velocity, which is only an approximation. The GPS reading
        also has measurement noise.

        A Kalman filter is a principled way to combine those two sources of information. It first predicts the next
        state using the motion model, then corrects that prediction using the new measurement.

        $$
        \hat{x}_t = F x_{t-1},
        \qquad
        x_t = \hat{x}_t + K(z_t - H\hat{x}_t)
        $$

        The key term is \(z_t - H\hat{x}_t\), which is the difference between what we measured and what the model
        expected to measure. The Kalman gain \(K\) controls how much we trust that measurement correction.

        For SORT, the same idea is applied to bounding boxes. Each track predicts where its box should be in the next
        frame. If a matching detection is found, the Kalman filter updates the track using that detection. If the
        detector misses the object for a few frames, SORT can still keep a predicted track alive briefly.
        """
    )
    gr.Image(
        value=str(EXAMPLE_IMAGE_DIR / "web" / "kalman-car-zoro.png"),
        label="Kalman Filter Example",
        interactive=False,
    )
    gr.Markdown(
        """
        Image source: https://www.signalpop.com/2024/06/30/using-the-unscented-kalman-filter-on-stock-prices/

        For a more detailed explanation of Kalman filtering, this video is helpful:
        https://www.youtube.com/watch?v=IFeCIbljreY
        """
    )

    gr.Markdown(
        r"""
        ## Hungarian Assignment for Detection-to-Track Matching
        The naive tracker used greedy IoU matching. It sorted all possible track/detection pairs by IoU and then kept
        taking the best remaining pair. That is simple and fast, but each decision is still local.

        SORT uses Hungarian assignment instead. We build a cost matrix where each row is a predicted track, each column
        is a detection, and each entry is the cost of assigning that track to that detection. In this implementation,
        the cost is just based on IoU:

        $$
        \text{cost} = 1 - \text{IoU}
        $$

        A lower cost means a better match. The Hungarian algorithm then finds the set of matches with the lowest total
        cost, while still making sure each track and each detection can only be used once.

        This still does not make SORT appearance-aware. It is only asking which predicted boxes overlap best with the
        current detections. The improvement over the naive baseline is that matching is done as one global assignment
        problem instead of one greedy pair at a time.
        """
    )

    gr.Markdown(
        """
        ## SORT Quantitative Results

        The improvement from using Kalman prediction is expected. SORT has a better motion prior than
        the naive IoU baseline, so it can handle short gaps in detection more gracefully.

        The gain is most visible in MOTA. SORT reduces false positives and identity switches compared with the naive
        baseline because predicted tracks are managed more consistently. IDF1 improves less dramatically, which makes
        sense because SORT still does not know anything about object appearance. If two people cross paths or overlap
        heavily, SORT is still relying on box geometry alone.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_sort_detectors-all_mota_idf1_motp_bubbles.png"
        ),
        label="Naive IoU and SORT IDF1-MOTA Plot",
        interactive=False,
    )
    gr.Markdown(
        """
        | Detector | Tracker | MOTA | IDF1 | MOTP | FP | FN | IDSW | Frag | ms/frame |
        |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
        | DPM | Naive IoU | 11.9% | 26.3% | 77.1% | 25063 | 72203 | 1668 | 3067 | **0.22** |
        | DPM | SORT | 27.6% | 28.2% | 78.9% | 2499 | 78076 | 768 | 1155 | 0.84 |
        | FRCNN | Naive IoU | 38.7% | 45.8% | 87.5% | 15033 | 52067 | 1777 | 1476 | 0.32 |
        | FRCNN | SORT | 48.8% | 50.3% | **88.0%** | 1556 | 55276 | **694** | **813** | 1.15 |
        | SDP | Naive IoU | 49.1% | 50.8% | 83.0% | 19026 | **35647** | 2493 | 3064 | 0.48 |
        | SDP | SORT | **61.5%** | **55.9%** | 85.8% | **977** | 41328 | 916 | 1342 | 1.46 |
        """
    )

    gr.Markdown(
        """
        ## SORT Runtime Evaluation

        SORT is still very fast but it is no longer as minimal as the naive IoU baseline. Each frame now includes a
        Kalman prediction for every active track, Hungarian assignment, and a Kalman update for each matched track.

        The runtime increase is visible in milliseconds per frame but it is still small in absolute terms. This is why
        SORT is such a useful baseline. It adds a meaningful motion model without adding an expensive appearance
        encoder.

        The cost also depends on the detector output. More detections usually means more possible track/detection
        pairs, which makes the assignment step larger.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_sort_detectors-all_performance_by_detector.png"
        ),
        label="Naive IoU and SORT Runtime Plot",
        interactive=False,
    )

    gr.Markdown(
        """
        ## DeepSORT: CNN Appearance-Augmented Association
        DeepSORT keeps the same basic structure as SORT but adds appearance information for re-identification or ReID.
        Instead of only asking which predicted box overlaps best with a detection, it also asks whether the cropped
        object looks like the same person seen by an existing track.

        In practice this means each detection is passed through a small CNN encoder to produce an appearance vector.
        Each track keeps a gallery of recent appearance vectors. During matching, DeepSORT compares the new detection
        embedding against that gallery. That gives the tracker a ReID signal in addition to the Kalman motion
        prediction.

        DeepSORT also uses Mahalanobis gating and a matching cascade. Mahalanobis gating checks whether a detection is
        plausible under the Kalman state uncertainty. The matching cascade gives recently updated tracks priority over
        stale tracks.

        I implemented this path in the project, but it is out of scope to go through every detail here. For this report,
        the important idea is that SORT can only reason about where an object should be. DeepSORT can also reason about
        what that object looks like.
        """
    )

    gr.Markdown(
        """
        ## DeepSORT Quantitative Results

        DeepSORT improves the main tracking scores again. This is most clear on SDP, where it gets the best MOTA and
        IDF1 in the comparison so far. That is the result we would hope to see from adding ReID.

        The improvement is not uniform across every metric. DeepSORT can reduce identity switches, but it may also keep
        or create tracks differently than SORT, so false positives and fragmentations do not always move in the same
        direction.

        The main point is that ReID gives the tracker another signal when box geometry alone is ambiguous.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_sort_deepsort_detectors-all_mota_idf1_motp_bubbles.png"
        ),
        label="Naive IoU, SORT, and DeepSORT IDF1-MOTA Plot",
        interactive=False,
    )
    gr.Markdown(
        """
        | Detector | Tracker | MOTA | IDF1 | MOTP | FP | FN | IDSW | Frag | ms/frame |
        |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
        | DPM | Naive IoU | 11.9% | 26.3% | 77.1% | 25063 | 72203 | 1668 | 3067 | **0.22** |
        | DPM | SORT | 27.6% | 28.2% | 78.9% | 2499 | 78076 | 768 | 1155 | 0.84 |
        | DPM | DeepSORT | 29.9% | 32.8% | 77.7% | 6879 | 71189 | **650** | 1576 | 18.28 |
        | FRCNN | Naive IoU | 38.7% | 45.8% | 87.5% | 15033 | 52067 | 1777 | 1476 | 0.32 |
        | FRCNN | SORT | 48.8% | 50.3% | **88.0%** | 1556 | 55276 | 694 | **813** | 1.15 |
        | FRCNN | DeepSORT | 49.4% | 49.2% | 86.4% | 4023 | 52182 | 659 | 1049 | 30.83 |
        | SDP | Naive IoU | 49.1% | 50.8% | 83.0% | 19026 | 35647 | 2493 | 3064 | 0.48 |
        | SDP | SORT | 61.5% | 55.9% | 85.8% | **977** | 41328 | 916 | 1342 | 1.46 |
        | SDP | DeepSORT | **65.2%** | **58.2%** | 84.4% | 3187 | **35068** | 834 | 1589 | 47.97 |
        """
    )

    gr.Markdown(
        """
        ## DeepSORT Runtime Evaluation

        The cost of DeepSORT is much larger than SORT. This is expected because we now run an appearance encoder on
        detections and compare those embeddings against the track galleries.

        This is the main tradeoff. DeepSORT gives the tracker a stronger identity signal, but it is no longer just a
        lightweight geometry tracker. If the goal is real time performance, the appearance model becomes one of the first
        places to optimize or replace.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_sort_deepsort_detectors-all_performance_by_detector.png"
        ),
        label="Naive IoU, SORT, and DeepSORT Runtime Plot",
        interactive=False,
    )

    gr.Markdown(
        """
        ## MyDeepSORT2: Color-Histogram Association

        The main motivation here was the slower performance of DeepSORT. I looked for a way to improve runtime even if
        it meant a slight degradation in tracking quality. The iterations I went through are listed in the appendix, but
        I do not currently have a graph to show those intermediate versions.

        MyDeepSORT2 is a small variant of DeepSORT rather than a completely new tracker. I kept the same broad tracking
        flow: Kalman prediction, matching cascade, feature gallery, track confirmation, and track deletion.

        The two changes are intentionally simple. First, I replaced the CNN appearance encoder with an HSV color
        histogram. This is a much weaker appearance feature than a DeepSORT-style appearance embedding, but it is also easier to
        compute and easier to reason about.

        Second, I replaced Mahalanobis gating in the cascade association step with an IoU gate. The cost still combines
        appearance distance and box overlap, but the motion constraint is closer to the original SORT style.

        Neither color histograms nor IoU gating are novel methods. The only potentially interesting part is how they are
        combined here inside a DeepSORT-like tracker.

        The goal was to test a cheaper approximation to DeepSORT.
        """
    )

    gr.Markdown(
        """
        ## HSV Color Histograms as Appearance Features

        The color histogram feature is intentionally simple. For each detection crop, I convert the image to HSV and
        count how many pixels fall into each color bin. This gives each detection a fixed length appearance vector.

        HSV is useful because it separates hue from brightness more cleanly than raw RGB. It is still a weak feature
        compared with a CNN-style ReID feature. Two people wearing similar colors can look very similar to this encoder,
        and lighting changes can still affect the result.

        This implementation uses OpenCV's HSV color conversion and histogram calculation. The crop is converted from BGR
        to HSV, then `cv2.calcHist` is used to count pixels across hue, saturation, and value bins.

        MyDeepSORT2 still uses the Kalman filter from DeepSORT. I am not removing motion prediction here. The change is
        that I replace the Mahalanobis gate with a simpler IoU gate during cascade matching.
        """
    )

    gr.Markdown(
        """
        ## MyDeepSORT2 Quantitative Results

        The result is better than I expected for such a simple change. MyDeepSORT2 keeps MOTA very close to DeepSORT
        and improves IDF1 across all three detectors.

        The most important result for this project is on SDP. DeepSORT gets 65.2% MOTA and 58.2% IDF1. MyDeepSORT2
        gets 65.1% MOTA and 62.8% IDF1. So in this run, the cheaper color histogram variant keeps almost the same MOTA
        while improving identity consistency.

        This does not mean color histograms are generally better than CNN-style ReID features. It does show that for
        this implementation and dataset setup, the simpler appearance signal was still useful.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_sort_deepsort_mydeepsort2_detectors-all_mota_idf1_motp_bubbles.png"
        ),
        label="Full Tracker IDF1-MOTA Plot",
        interactive=False,
    )
    gr.Markdown(
        """
        | Detector | Tracker | MOTA | IDF1 | MOTP | FP | FN | IDSW | Frag | ms/frame |
        |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
        | DPM | Naive IoU | 11.9% | 26.3% | 77.1% | 25063 | 72203 | 1668 | 3067 | **0.22** |
        | DPM | SORT | 27.6% | 28.2% | 78.9% | 2499 | 78076 | 768 | 1155 | 0.84 |
        | DPM | DeepSORT | 29.9% | 32.8% | 77.7% | 6879 | 71189 | 650 | 1576 | 18.28 |
        | DPM | MyDeepSORT2 | 30.5% | 37.2% | 77.9% | 5393 | 72231 | **422** | 1327 | 3.58 |
        | FRCNN | Naive IoU | 38.7% | 45.8% | 87.5% | 15033 | 52067 | 1777 | 1476 | 0.32 |
        | FRCNN | SORT | 48.8% | 50.3% | **88.0%** | 1556 | 55276 | 694 | **813** | 1.15 |
        | FRCNN | DeepSORT | 49.4% | 49.2% | 86.4% | 4023 | 52182 | 659 | 1049 | 30.83 |
        | FRCNN | MyDeepSORT2 | 49.3% | 54.8% | 86.6% | 3321 | 52933 | 644 | 975 | 4.56 |
        | SDP | Naive IoU | 49.1% | 50.8% | 83.0% | 19026 | 35647 | 2493 | 3064 | 0.48 |
        | SDP | SORT | 61.5% | 55.9% | 85.8% | **977** | 41328 | 916 | 1342 | 1.46 |
        | SDP | DeepSORT | **65.2%** | 58.2% | 84.4% | 3187 | **35068** | 834 | 1589 | 47.97 |
        | SDP | MyDeepSORT2 | 65.1% | **62.8%** | 84.6% | 2367 | 36159 | 696 | 1422 | 5.65 |
        """
    )

    gr.Markdown(
        """
        ## MyDeepSORT2 Runtime Evaluation

        This is where the custom variant is most useful. MyDeepSORT2 is much slower than SORT, but much faster than
        DeepSORT. On SDP, DeepSORT takes 47.97 ms/frame while MyDeepSORT2 takes 5.65 ms/frame.

        That is the main tradeoff I wanted to show. The color histogram version keeps similar MOTA and improves IDF1 in
        this run, while removing most of the CNN appearance encoding cost.

        This does not make it a better tracker in general, but it does show that cheaper appearance features can be
        useful when runtime matters.
        """
    )
    gr.Image(
        value=str(
            METRIC_IMAGE_DIR
            / "MOT17_tracking_trackers-naive_iou_sort_deepsort_mydeepsort2_detectors-all_performance_by_detector.png"
        ),
        label="Full Tracker Runtime Plot",
        interactive=False,
    )

    gr.Markdown(
        """
        ## Appendix: DeepSORT Ablation Considerations

        The main comparison focuses on Naive IoU, SORT, DeepSORT, and the final MyDeepSORT2 variant. During development,
        I also tried a few intermediate DeepSORT ablations to understand which components were costing the most runtime
        and which ones were hurting tracking quality.

        The rough path was:

        1. **DeepSORT baseline**: Kalman prediction, Mahalanobis gating, matching cascade, and CNN appearance features.
        2. **Replace CNN features with SIFT**: I tried a classical local descriptor to reduce appearance-model cost, but
        tracking quality decreased.
        3. **Look beyond the CNN encoder**: after that, I noticed that the CNN was not the only bottleneck. The
        Mahalanobis gating step was also adding cost.
        4. **Replace Mahalanobis gating with IoU gating**: this kept the Kalman prediction and cascade structure, but
        used the simpler overlap test from SORT as the motion gate.
        5. **Replace SIFT with HSV color histogram binning**: this was the final version kept as MyDeepSORT2. It uses a
        lower-cost global appearance descriptor instead of local SIFT features or CNN embeddings.
        6. **Color-invariant histogram binning**: I considered this as a next step, but did not implement it in the final
        project version.

        I did not complete a full ablation graph for these intermediate versions. The purpose of this appendix is to
        document the design rationale and the issues I saw while moving from the DeepSORT baseline to the cheaper final
        variant.
        """
    )

    gr.Markdown(
        """
        ## Future Work

        - Dense prediction masks instead of bounding boxes. This would let the tracker reason about the actual visible
          shape of each person instead of a rectangular approximation.
        - Single-stage MOT pipelines that jointly detect and track objects. This would remove the strict separation
          between detector and tracker and allow the model to optimize identity consistency directly.
        """
    )

    gr.Markdown(
        """
        ## References

        Papers:

        - Bewley et al., "Simple Online and Realtime Tracking", 2016.
        - Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric", 2017.
        - Du et al., "StrongSORT: Make DeepSORT Great Again", 2022.

        Dataset and supporting resources:

        - MOTChallenge / MOT17 benchmark and dataset.
        - MOT tracking metrics overview: https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics
        - Kalman filter tutorial video: https://www.youtube.com/watch?v=IFeCIbljreY
        - Kalman filter image credit: https://www.signalpop.com/2024/06/30/using-the-unscented-kalman-filter-on-stock-prices/
        - OpenCV color space conversions: https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html
        - OpenCV histogram calculation: https://docs.opencv.org/4.x/d6/dc7/group__imgproc__hist.html
        """
    )

    gr.Markdown(
        """
        ## AI Use Policy

        I worked on the core tracker implementations, experimentation framework, and write-up. I used AI assistance for
        brainstorming, utility code, automation, the evaluation framework, debugging, cleanup, generating repeatable
        plots and videos, and proofreading. I reviewed and integrated the changes myself.

        For the implementation, I preserved my initial versions and then used AI to iterate toward versions that aligned
        more closely with the papers. Those older versions are preserved in the code but are not included in the report.
        """
    )


def main() -> None:
    demo.launch(ssr_mode=False)


if __name__ == "__main__":
    main()
