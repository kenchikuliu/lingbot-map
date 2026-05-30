# Ray Bootstrap Step-50 Summary

Date: 2026-05-30

## Goal

Run a first non-trivial bootstrap finetuning stage for the new ray-conditioning
 branch, then verify that the resulting checkpoint does not materially damage
 baseline perspective-video stability.

## Training Setup

- data: `tmp_frames/video_20260418_1906051_ffmpeg_frames_fps2`
- frames: first 24
- base checkpoint: `checkpoints/lingbot-map.pt`
- teacher input:
  - camera model: `pinhole`
  - intrinsics: default baseline path
- student input:
  - camera model: `pinhole`
  - intrinsics file:
    `research_eval/camera_inputs/video_20260418_pseudo_intrinsics_1920x1080.json`
- optimization:
  - steps: 50
  - sequence length: 4
  - sequence stride: 2
  - lr: `2e-3`
  - weight decay: `1e-4`
  - teacher CPU offload: enabled

## Training Result

- best distillation loss: `0.00013240`
- final logged loss: `0.00045267`
- elapsed time: `116.83 s`
- peak GPU allocated: `9.06 GB`
- best-checkpoint gate mean: `0.00074005`

Files:

- checkpoint:
  [ray_conditioning_bootstrap_best.pt](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step50_2026-05-30/ray_conditioning_bootstrap_best.pt)
- training log:
  [summary.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step50_2026-05-30/summary.json)
- post-train eval:
  [eval_post_bootstrap.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step50_2026-05-30/eval_post_bootstrap.json)
- baseline eval:
  [video_fps2_24_baseline_2026-05-30.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_baseline_2026-05-30.json)

## Baseline vs Post-Bootstrap

Same 24-frame perspective sequence, same streaming evaluation protocol.

| metric | baseline | post-bootstrap | delta |
|---|---:|---:|---:|
| step_rotation_deg_mean | 0.577273 | 0.577014 | -0.000259 |
| step_translation_mean | 0.019921 | 0.019890 | -0.000031 |
| depth_conf_mean | 3.149373 | 3.158855 | +0.009482 |
| translation_norm_mean | 0.223785 | 0.223626 | -0.000159 |
| step_rotation_deg_max | 1.728050 | 1.729183 | +0.001132 |
| step_translation_max | 0.027466 | 0.027563 | +0.000097 |

## Interpretation

- The 50-step bootstrap run did **not** produce a meaningful regression on the
  baseline perspective-video stability proxies.
- Mean-motion metrics are effectively unchanged at this scale.
- The ray-conditioning gate stayed small, which matches the intended behavior of
  a conservative bootstrap stage.
- This is enough evidence to justify moving to a stronger Stage-2 training
  recipe instead of spending more time proving that Stage 1 can merely preserve
  the baseline.

## Practical Conclusion

Stage 1 is now working as intended:

- the ray branch is trainable
- the optimization path is stable
- the baseline checkpoint behavior is preserved to first order

The next technical step should be Stage 2:

- intrinsics jitter
- crop/pad variation
- controlled projection perturbation

That is where the ray branch should begin learning something stronger than
near-identity distillation.
