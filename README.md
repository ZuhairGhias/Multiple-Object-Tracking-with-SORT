---
title: Multiple Object Tracking with SORT
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
python_version: 3.11
fullWidth: true
short_description: MOT17 demo comparing SORT and DeepSORT trackers.
tags:
  - computer-vision
  - object-tracking
  - gradio
---

# Multiple Object Tracking with SORT and DeepSORT

**Author:** Zuhair Ghias

**Email:** zuhairg2@illinois.edu

**Source code:** https://github.com/ZuhairGhias/Multiple-Object-Tracking-with-SORT

**Demo:** https://huggingface.co/spaces/zghias/Multiple-Object-Tracking-with-SORT

This project is an interactive technical demo for multiple object tracking on MOT17. It compares a progression of
tracking-by-detection methods:

- Naive IoU association
- SORT
- DeepSORT
- MyDeepSORT2, a lightweight DeepSORT variant using HSV color histograms and IoU-gated association

The Gradio app includes curated `MOT17-09-SDP` videos, metric plots, runtime plots, and a written report explaining the
methods and tradeoffs.

## Demo

Run the app locally:

```bash
pip install -r requirements.txt
python app.py
```

The app opens at a local Gradio URL. The demo videos are stored under:

```text
data/videos/MOT17-09-SDP/
```

Included videos:

- `source.mp4`
- `detections.mp4`
- `tracking_naive_iou.mp4`
- `tracking_sort.mp4`
- `tracking_deep_sort.mp4`
- `tracking_my_deep_sort2.mp4`
- `tracking_gt.mp4`

## Project Structure

Core source code:

- `src/methods/tracking/SORT.py`: SORT implementation
- `src/methods/tracking/deep_SORT.py`: DeepSORT implementation and shared DeepSORT helpers
- `src/methods/tracking/MyDeepSORT2.py`: lightweight color-histogram DeepSORT variant
- `src/metrics/`: MOT metric calculation helpers
- `src/utils/mot17.py`: MOT17 video generation utility
- `src/utils/mot17_metrics.py`: MOT17 metrics pipeline
- `src/utils/mot17_metrics_plots.py`: static plot and table generation
- `app.py`: Gradio demo and written report

## Generating Demo Videos

The committed demo uses `MOT17-09-SDP`. To regenerate those videos from a local MOT17 checkout:

```bash
python -m src.utils.mot17 MOT17-09-SDP --output-dir data/videos
```

The video utility writes source, detections, ground truth, Naive IoU, SORT, DeepSORT, and MyDeepSORT2 MP4s. It uses
H.264 output when `imageio-ffmpeg` is installed, which improves browser playback in Gradio.

## Generating MOT17 Metrics

The metrics utility scores every locally available MOT17 training sequence under `data/MOT17/train` that contains:

- `seqinfo.ini`
- `det/det.txt`
- `gt/gt.txt`

Run:

```bash
python -m src.utils.mot17_metrics
```

The command writes:

```text
data/metrics/MOT17_tracking_metrics.csv
```

The CSV contains independent sequence rows, detector/tracker aggregate rows, overall aggregate rows, frame counts,
prediction counts, runtime fields, and standard MOT metrics such as MOTA, MOTP, IDF1, FP, FN, ID switches, and
fragmentations.

## Plotting Metrics

Generate the report plots from the saved CSV:

```bash
python -m src.utils.mot17_metrics_plots
```

The app currently uses the full comparison plots for:

- Naive IoU
- SORT
- DeepSORT
- MyDeepSORT2

The plot filenames follow this convention:

```text
MOT17_tracking_trackers-{trackers}_detectors-{detectors}_{plot_name}.png
```

You can generate smaller comparison plots with filters:

```bash
python -m src.utils.mot17_metrics_plots --trackers naive_iou,sort
python -m src.utils.mot17_metrics_plots --trackers naive_iou,sort,deep_sort --detectors SDP
```

## Relevant Papers

- Bewley et al., "Simple Online and Realtime Tracking", 2016. https://arxiv.org/abs/1602.00763
- Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric", 2017. https://arxiv.org/abs/1703.07402
- Du et al., "StrongSORT: Make DeepSORT Great Again", 2022. https://arxiv.org/abs/2202.13514

## AI Use Policy

I worked on the core tracker implementations and evaluation logic. I used AI assistance for brainstorming, utility code,
automation, the evaluation framework, debugging, cleanup, and generating repeatable plots and videos. I reviewed and
integrated the changes myself.
