# MOT17 SORT Debugging Plan

## Summary

The project has moved past the demo-only video player. We now have a MOT17 frame pipeline that can generate source, detection, ground-truth tracking, and simple SORT tracking videos. The next priority is a debugging workflow that makes it easy to inspect individual frames and understand why tracks fail before investing in formal metrics.

## Completed

- Added MOT17 detection parsing from `det/det.txt`.
- Added a tracking interface plus a ground-truth tracker backed by `gt/gt.txt`.
- Added a simple IoU-based `SORT` tracker prototype.
- Refactored utilities around a one-pass frame pipeline:
  - `src/utils/mot17.py` is the MOT17 CLI entry point.
  - `src/utils/frames2mp4.py` handles MP4 writing.
  - `src/utils/render.py` handles drawing detections and tracks.
- Generated per-sequence videos under `data/videos/<sequence>/`:
  - `source.mp4`
  - `detections.mp4`
  - `tracking_sort.mp4`
  - `tracking_gt.mp4` when ground truth is available
- Kept the full MOT17 dataset local and ignored by git.

## Current Problem

The simple SORT tracker runs, but it performs poorly. Before adding standard MOT metrics, we need better visibility into what happens frame by frame:

- Which detections are available on a given frame?
- Which detections does SORT match to existing tracks?
- Which tracks are stale or newly created?
- How do SORT tracks compare with ground truth on the same frame?
- Does a specific ID disappear, switch, or drift?

## Next Milestone: Frame Debugger

Build a frame-level debugging tool alongside the MP4 outputs.

Desired behavior:

- Select a MOT17 sequence.
- Select a frame number.
- Replay tracker state from frame 1 through the selected frame, because SORT is stateful.
- Render the selected frame in several views:
  - source frame
  - detections
  - SORT tracks
  - ground-truth tracks when available
- Optionally focus on a specific ID.
- When an ID is focused:
  - highlight that ID
  - dim or de-emphasize other tracks
  - show whether the ID is missing on the selected frame

Initial implementation can be CLI or utility-first. A Gradio panel with a frame slider can come after the core frame-inspection function works.

## Debug Data To Surface

For each inspected frame, report:

- frame index
- detection count
- active SORT track IDs
- active ground-truth IDs when available
- focused ID status when provided
- boxes for focused SORT and ground-truth IDs when available

Later, if helpful, include matching diagnostics:

- IoU candidates
- accepted detection-to-track matches
- unmatched detections
- unmatched tracks
- newly created tracks
- expired tracks

## Evaluation Milestone

After the frame debugger is useful, add an evaluation layer.

Start with simple interpretable metrics:

- detection coverage against ground truth
- false positives and false negatives using IoU matching
- approximate ID switches
- track fragmentation
- per-frame match summaries

Defer full MOTChallenge-style metrics until the debugging workflow is strong enough to explain failures.

## Open Questions

- Should the frame debugger live first as a CLI utility, a Gradio panel, or both?
- What default IoU threshold should evaluation use for matching tracks to ground truth?
- Should focused IDs refer to SORT IDs, ground-truth IDs, or allow selecting either namespace?
- How much match-debug state should `SORT.update()` expose versus keeping the tracker interface minimal?

## Near-Term Recommendation

Build the frame debugger first. Metrics will be more useful once we can inspect and explain individual tracking failures.
