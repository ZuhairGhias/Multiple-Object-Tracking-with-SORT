# Data Layout

This repository keeps only small demo assets under `data/` in git. The full MOT17 dataset is expected to live here locally, but it is ignored by git and will not be synced to GitHub.

## Expected Layout

```text
data/
  MOT17/
    train/
      MOT17-02-FRCNN/
      ...
    test/
      MOT17-01-FRCNN/
      ...
  videos/
    temp/
      MOT17-01-DPM/
        source.mp4
        detections.mp4
    MOT17-01-DPM/
      source.mp4
      detections.mp4
```

- `data/MOT17/` is where a full local MOT17 checkout or extract normally lives.
- `data/videos/temp/` holds regenerated local scratch videos and is ignored by git.
- `data/videos/` holds curated per-sequence MP4 files that are ready to keep.

## Git Behavior

The following paths are intentionally ignored:

```text
data/MOT17
data/videos/temp/
```

That means you can keep the dataset locally without accidentally syncing it.

## Generating MP4 Files

If you have an extracted MOT17 sequence locally, you can generate `source.mp4`, `detections.mp4`, and, when ground truth is available, `tracking_gt.mp4` by pointing the utility at the sequence root:

```bash
python -m src.utils.mot17 data/MOT17/test/MOT17-01-FRCNN
```

That command expects the input directory to contain `img1/`, `seqinfo.ini`, and `det/det.txt`, and by default it will create scratch outputs:

```text
data/videos/temp/MOT17-01-FRCNN/source.mp4
data/videos/temp/MOT17-01-FRCNN/detections.mp4
```

The Gradio app checks `data/videos/temp/<sequence>/` first and falls back to curated videos in `data/videos/<sequence>/`. Detection score filtering is configured inside the MOT17 detector.
