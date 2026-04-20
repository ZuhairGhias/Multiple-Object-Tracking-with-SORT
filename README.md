---
title: Multiple Object Tracking with SORT
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
python_version: 3.11
fullWidth: true
short_description: Interactive SORT demo for MOT17 tracking visualizations.
tags:
  - computer-vision
  - object-tracking
  - gradio
---

# Multiple-Object-Tracking-with-SORT

This project implements the SORT (Simple Online and Realtime Tracking) algorithm and provides an interactive demo for visualizing object tracking behavior.

## Features
- SORT tracker implementation
- Video-based object tracking
- Interactive demo (Gradio)
- Visualization of track IDs and trajectories

## Demo
Launch the Gradio demo to play the sample MOT17 video.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Generating Demo Videos From MOT17

The repository keeps curated demo MP4s in `data/videos/`, but it does not need the extracted MOT17 frame folders checked in. Regenerated local videos are written to `data/videos/temp/`, which is ignored by git.

To build source and detections MP4s from an original MOT17 sequence that you have locally:

1. Download and extract the MOT17 dataset outside the repository.
2. Pick a sequence root that contains `img1/`, `seqinfo.ini`, and `det/det.txt`, for example `~/datasets/MOT17/train/MOT17-01-DPM`.
3. Run the MOT17 utility against that sequence root:

```bash
python -m src.utils.mot17 ~/datasets/MOT17/train/MOT17-01-DPM
```

By default, the script:
- reads the frame rate from `seqinfo.ini`
- writes `data/videos/temp/MOT17-01-DPM/source.mp4`
- writes `data/videos/temp/MOT17-01-DPM/detections.mp4`
- uses ffmpeg/libx264 for MP4 output when available

For train sequences that include `gt/gt.txt`, the utility also writes `tracking_gt.mp4`. For test sequences, it prints a warning and skips that output.

## Relevant Papers

### SORT (Baseline)
- **Simple Online and Realtime Tracking (SORT)**  
  Bewley et al., 2016  
  https://arxiv.org/abs/1602.00763  
  - Introduces a lightweight tracking-by-detection framework  
  - Uses Kalman filtering + Hungarian algorithm + IoU matching  
  - Focuses on real-time performance

---

### DeepSORT (Appearance Modeling)
- **Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)**  
  Wojke et al., 2017  
  https://arxiv.org/abs/1703.07402  
  - Extends SORT with deep appearance embeddings  
  - Improves robustness to occlusions and re-identification  
  - Combines motion + appearance for association

---

### StrongSORT (Modern Improvements)
- **StrongSORT: Make DeepSORT Great Again**  
  Du et al., 2022  
  https://arxiv.org/abs/2202.13514  
  - Improves DeepSORT with multiple plug-and-play upgrades  
  - Better motion modeling, appearance features, and matching  
  - Strong performance on modern MOT benchmarks
