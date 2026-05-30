# Phase 0 Baseline Summary

Date: 2026-05-27
Model: LingBot-MAP public checkpoint
Purpose: establish baseline failure boundaries before any RoAM3R method changes

## Scope

These results are not ground-truth trajectory benchmarks.
The local sequences used here do not provide trusted pose GT.

Therefore, this phase uses repeatable proxy stability metrics:

- frame-to-frame rotation magnitude
- frame-to-frame translation magnitude
- variability of step motion
- variability of depth confidence across frames
- inference speed and memory

The point is not to claim final geometric accuracy.
The point is to establish whether the baseline becomes less stable under stress.

## 1. Temporal Sparsity: `1 fps` vs `2 fps`

Source:

- video: `/home/slam/datasets/zhuyevideos/video_20260418_190605(1).mp4`
- first `24` frames after extraction
- mode: `streaming`

Reports:

- `research_eval/temporal_subsampling/fps1_vs_fps2_small/fps_1.json`
- `research_eval/temporal_subsampling/fps1_vs_fps2_small/fps_2.json`

### Observed proxy metrics

| Setting | Step rot mean | Step rot max | Step trans mean | Step trans max | Step trans std | Depth-conf frame std |
|---|---:|---:|---:|---:|---:|---:|
| `1 fps` | `1.1482` | `3.1572` | `0.0381` | `0.0583` | `0.00529` | `0.4459` |
| `2 fps` | `0.5922` | `1.8041` | `0.0216` | `0.0273` | `0.00352` | `0.1974` |

### Interpretation

Compared with `2 fps`, the `1 fps` baseline shows:

- about `1.94x` larger mean step rotation
- about `1.76x` larger mean step translation
- about `2.26x` larger framewise depth-confidence variability

This is direct evidence that temporal sparsity makes the baseline substantially less stable.

## 2. Projection Shift: perspective video vs 360-derived perspective sequence

Sources:

- perspective: `tmp_frames/video_20260418_1906051_ffmpeg_frames_fps1`
- wide-FoV / 360-derived: `tmp_frames/outdoor_perspective_az090`
- first `24` frames
- mode: `windowed`

Reports:

- `research_eval/projection_shift/perspective_vs_360_small/perspective.json`
- `research_eval/projection_shift/perspective_vs_360_small/widefov.json`

### Important caveat

This is not a same-scene matched benchmark.
The two sequences are different captures.

Therefore, this result should be treated as boundary evidence, not as a final paper table.

### Observed proxy metrics

| Setting | Step rot mean | Step rot max | Step trans mean | Step trans max | Step trans std | Depth-conf frame std |
|---|---:|---:|---:|---:|---:|---:|
| perspective | `1.1078` | `3.1572` | `0.0382` | `0.0577` | `0.00520` | `0.5761` |
| 360-derived | `37.8099` | `126.0191` | `3.3665` | `10.2935` | `3.1474` | `1.0871` |

### Interpretation

The 360-derived sequence is dramatically less stable than the normal perspective sequence:

- mean step rotation is about `34.1x` larger
- mean step translation is about `88.1x` larger
- step translation variability is about `605x` larger
- depth-confidence frame variability is about `1.89x` larger

Even with the caveat that the scenes differ, this is strong practical evidence that the public LingBot-MAP pipeline is badly mismatched to wide-FoV / 360-derived input.

## 3. Viewpoint Break: control vs dropped middle segment

Source:

- source frames: `tmp_frames/video_20260418_1906051_ffmpeg_frames_fps2`
- control: first `24` frames intact
- break condition: first `24` frames with frames `[8, 15]` removed
- mode: `streaming`

Reports:

- `research_eval/viewpoint_break/control_vs_drop_24/control_24.json`
- `research_eval/viewpoint_break/control_vs_drop_24/drop_mid_24.json`

### Observed proxy metrics

| Setting | Step rot mean | Step rot max | Step trans mean | Step trans max | Step trans std | Depth-conf frame std |
|---|---:|---:|---:|---:|---:|---:|
| control | `0.5922` | `1.8041` | `0.0216` | `0.0273` | `0.00352` | `0.1974` |
| drop-mid | `0.6445` | `3.2119` | `0.0324` | `0.1663` | `0.0360` | `0.2067` |

### Interpretation

The dropped-middle sequence does not blow up in mean values alone, but it clearly spikes in instability:

- max step rotation increases from `1.80` to `3.21`
- max step translation increases from `0.0273` to `0.1663`, about `6.1x`
- step translation std increases from `0.00352` to `0.0360`, about `10.2x`

This matters because viewpoint discontinuity appears mainly as burst instability, not just as a smooth increase in average motion.

## 4. Phase 0 Conclusion

Phase 0 succeeded.

The baseline shows three useful boundaries:

1. Temporal sparsity weakens stability.
2. 360-derived / wide-FoV input is strongly mismatched to the current pipeline.
3. Viewpoint discontinuity introduces burst instability even when average metrics do not fully expose it.

That is enough evidence to justify the RoAM3R paper direction:

- projection-aware geometry
- adaptive memory for sparse temporal support
- explicit recovery behavior

## 5. What This Means for RoAM3R

These results support the following module-to-problem mapping:

1. `Unified Ray Camera Interface`
   Justified by the severe instability under 360-derived input.

2. `Overlap-Aware Adaptive Memory`
   Justified by the degradation from `2 fps` to `1 fps`.

3. `Uncertainty-Gated Relocalization`
   Justified by burst instability under dropped-frame viewpoint discontinuity.

## 6. Immediate Next Step

The next highest-value move is not more paper writing.
It is to convert these Phase 0 proxy results into a cleaner Phase 1 benchmark harness:

- add matched projection-shift pairs if possible
- add CSV aggregation across multiple runs
- add a small plotting utility for degradation curves

## 7. Preprocess Ablation Addendum

An extra ablation was run to test whether the wide-FoV instability is mainly a preprocessing artifact.

### 7.1 360-derived perspective cuts: `crop` vs `pad`

Source:

- `tmp_frames/outdoor_perspective_az090`
- first `24` frames
- mode: `windowed`

Observed:

- `crop` and `pad` produce effectively identical proxy metrics
- mean step rotation stays at `37.81`
- mean step translation stays at `3.3665`

Interpretation:

- for these already square perspective-cut frames, preprocessing mode is not the bottleneck

### 7.2 Raw fisheye center crops: `crop` vs `pad`

Source:

- `tmp_frames/outdoor_right_center_crops`
- first `24` frames
- mode: `windowed`

Observed:

- `crop`: step translation mean `2.8838`, depth-conf frame std `1.1354`
- `pad`: step translation mean `2.6990`, depth-conf frame std `1.0071`

Interpretation:

- `pad` changes the behavior slightly, but does not fundamentally resolve instability
- the instability remains orders of magnitude larger than normal perspective input

### 7.3 Conclusion

This ablation weakens a simple "just fix preprocessing" explanation.
Preprocessing rigidity is part of the problem, but it is not the central failure source.

That strengthens the RoAM3R framing:

- the main issue is still camera geometry mismatch and temporally fragile streaming assumptions
- not only the current crop/pad policy
