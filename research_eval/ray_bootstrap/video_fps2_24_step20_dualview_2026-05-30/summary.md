# Ray Bootstrap Mixed Dual-View Summary

Date: 2026-05-30

## Goal

Stress the ray-conditioning bootstrap beyond single-view crop distillation by
mixing:

- stronger intrinsics jitter
- framewise intrinsics drift inside each training chunk
- alternating student preprocess views (`crop` and `pad`)

This is still a bootstrap experiment, not a final robustness claim.

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
  - steps: 20
  - sequence length: 4
  - sequence stride: 2
  - lr: `2e-3`
  - weight decay: `1e-4`
  - teacher CPU offload: enabled
- perturbations:
  - `--student_focal_jitter_pct 0.10`
  - `--student_principal_point_jitter_px 16`
  - `--student_framewise_focal_drift_pct 0.05`
  - `--student_framewise_principal_point_drift_px 8`
  - `--jitter_schedule cosine_ramp`
  - `--jitter_ramp_steps 20`
  - `--student_dual_view_preprocess_mode pad`
  - `--dual_view_distill_prob 0.5`

## Training Result

- best distillation loss: `0.00020754`
- final logged loss: `0.00073126`
- elapsed time: `104.42 s`
- peak GPU allocated: `13.08 GB`
- final gate mean in training log: `0.00094223`
- saved-checkpoint gate stats:
  - mean: `0.00099945`
  - min: `-0.00102234`
  - max: `0.00302124`

Files:

- training log:
  [summary.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step20_dualview_2026-05-30/summary.json)
- post-train eval:
  [eval_post_dualview.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_step20_dualview_2026-05-30/eval_post_dualview.json)
- baseline eval:
  [video_fps2_24_baseline_2026-05-30.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/video_fps2_24_baseline_2026-05-30.json)

## What Actually Happened During Training

- total steps: `20`
- crop-view steps: `11`
- pad-view steps: `9`
- dual-view student path used: `9/20` steps
- depth loss active: `11/20` steps
- depth-confidence loss active: `11/20` steps

Important detail:

- when the student used the `pad` view, the spatial tensor shape no longer
  matched the crop-view teacher target
- those steps therefore trained only pose and FoV distillation
- depth and depth-confidence losses were skipped by design on `pad` batches

## Baseline vs Post-Dual-View Eval

Same 24-frame perspective sequence, same streaming evaluation protocol.

| metric | baseline | post-dual-view | delta |
|---|---:|---:|---:|
| step_rotation_deg_mean | 0.577273 | 0.575498 | -0.001776 |
| step_translation_mean | 0.019921 | 0.019924 | +0.000003 |
| depth_conf_mean | 3.149373 | 3.170730 | +0.021357 |
| translation_norm_mean | 0.223785 | 0.224022 | +0.000237 |
| step_rotation_deg_max | 1.728050 | 1.717828 | -0.010223 |
| step_translation_max | 0.027466 | 0.027155 | -0.000311 |

## Comparison to Earlier Bootstrap Stages

| metric | Stage 1 | Stage 2 jitter | mixed dual-view |
|---|---:|---:|---:|
| step_rotation_deg_mean | 0.577014 | 0.575994 | 0.575498 |
| step_translation_mean | 0.019890 | 0.019946 | 0.019924 |
| depth_conf_mean | 3.158855 | 3.175366 | 3.170730 |
| step_rotation_deg_max | 1.729183 | 1.723515 | 1.717828 |

## Interpretation

- The stronger mixed recipe still does not show a meaningful regression on the
  baseline perspective-video stability proxies.
- Mean and max step rotation improved slightly relative to both the baseline
  checkpoint and the earlier Stage 2 jitter run.
- Translation stability stayed effectively flat.
- The ray-conditioning gate remains small, which is still consistent with a
  conservative bootstrap phase instead of aggressive adaptation.

## Caveat

- The best checkpoint was selected by total distillation loss.
- In this run the best loss happened at step 1 on a crop-view batch.
- That means `ray_conditioning_bootstrap_best.pt` is still biased toward
  conservative crop-view preservation, not necessarily the strongest `pad`-view
  adaptation point.

## Practical Conclusion

This run justifies the next step technically:

1. keep the same stable bootstrap scaffold
2. add checkpoint selection that explicitly tracks dual-view batches
3. move from perspective-only student perturbation toward actual
   projection-shift finetuning

What it does **not** justify yet:

- any claim of ground-truth trajectory improvement
- any claim that wide-FoV robustness is solved
- any claim that the current `best.pt` is the right final adaptation checkpoint
