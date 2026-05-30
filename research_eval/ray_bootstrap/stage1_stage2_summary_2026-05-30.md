# Stage 1 / Stage 2 Bootstrap Summary

Date: 2026-05-30

## Scope

This note summarizes three checkpoints on the same 24-frame perspective-video
 sequence:

1. baseline public checkpoint
2. Stage 1 bootstrap distillation
3. Stage 2 intrinsics-jitter bootstrap

## Shared Evaluation Sequence

- folder: `tmp_frames/video_20260418_1906051_ffmpeg_frames_fps2`
- frames: first 24
- mode: streaming
- scale frames: 8
- backend: SDPA

## Checkpoints

- baseline:
  [video_fps2_24_baseline_2026-05-30.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_baseline_2026-05-30.json)
- Stage 1 bootstrap:
  [summary.md](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step50_2026-05-30/summary.md)
- Stage 2 jitter:
  [summary.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step30_jitter_2026-05-30/summary.json)

## Training Recipes

### Stage 1

- student intrinsics:
  `research_eval/camera_inputs/video_20260418_pseudo_intrinsics_1920x1080.json`
- no intrinsics jitter
- 50 steps

### Stage 2

- same student intrinsics file
- `--student_focal_jitter_pct 0.05`
- `--student_principal_point_jitter_px 8`
- 30 steps

## Stability Comparison

| metric | baseline | Stage 1 | Stage 2 |
|---|---:|---:|---:|
| step_rotation_deg_mean | 0.577273 | 0.577014 | 0.575994 |
| step_translation_mean | 0.019921 | 0.019890 | 0.019946 |
| depth_conf_mean | 3.149373 | 3.158855 | 3.175366 |
| translation_norm_mean | 0.223785 | 0.223626 | 0.224086 |

## Stage 1 vs Baseline

- mean rotation delta: `-0.000259`
- mean translation delta: `-0.000031`
- depth confidence delta: `+0.009482`
- interpretation:
  - essentially no baseline regression

## Stage 2 vs Baseline

- mean rotation delta: `-0.001280`
- mean translation delta: `+0.000026`
- depth confidence delta: `+0.025992`
- max rotation delta: `-0.004536`
- max translation delta: `-0.000316`
- interpretation:
  - still no meaningful baseline regression at this scale
  - the jittered recipe remains conservative

## Optimization Notes

### Stage 1

- best loss: `0.00013240`
- best checkpoint gate mean: about `0.000740`
- elapsed: `116.83 s`

### Stage 2

- best loss: `0.00018623`
- best checkpoint gate mean: about `0.001022`
- elapsed: see
  [summary.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step30_jitter_2026-05-30/summary.json)

## Conclusion

Two conclusions are now supported by actual runs:

1. the ray-conditioning branch is trainable under a stable teacher-student
   bootstrap recipe
2. adding moderate intrinsics jitter still preserves baseline perspective-video
   stability

That is enough to justify a stronger next phase:

- larger jitter ranges
- chunk-level camera perturbation schedules
- eventually wide-FoV / projection-shift adaptation
