# Projection-Shift Pad Distillation Smoke Summary

Date: 2026-05-30

## Goal

Run a lighter cross-folder projection-shift bootstrap where the student uses
`pad` preprocessing so teacher and student tensors share the same spatial shape
and full geometric distillation can activate.

## Training Setup

- teacher folder:
  `tmp_frames/outdoor_perspectives_8_az135_el000`
- student folder:
  `tmp_frames/outdoor_right_center_crops`
- pairing mode: `ordered`
- frames: first 12
- teacher preprocess: `crop`
- student preprocess: `pad`
- steps: 6
- sequence length: 2
- sequence stride: 1
- scale frames: 1
- camera iterations: 1
- teacher CPU offload: enabled

## Training Result

- best loss: `0.140527`
- final loss: `0.140527`
- final gate mean: `0.000893`
- peak GPU allocated: `8.44 GB`

Files:

- training log:
  [summary.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_pad_smoke12_2026-05-30/summary.json)
- baseline eval:
  [eval_baseline_student_pad12.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_pad_smoke12_2026-05-30/eval_baseline_student_pad12.json)
- post-train eval:
  [eval_post_student_pad12.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_pad_smoke12_2026-05-30/eval_post_student_pad12.json)

## What Changed Technically

Unlike the earlier crop-to-crop smoke run, this recipe activated all four
distillation terms:

- pose
- FoV
- depth
- depth-confidence

Training evidence from the log:

- step 1 total loss: `0.302254`
- step 6 total loss: `0.140527`
- step 1 depth loss: `0.222166`
- step 6 depth loss: `0.112613`
- step 1 depth-conf loss: `1.333202`
- step 6 depth-conf loss: `0.357308`

## Student-Side Eval Delta

Same 12-frame student folder, same `pad` evaluation settings.

| metric | baseline | post-train | delta |
|---|---:|---:|---:|
| step_rotation_deg_mean | 46.333229 | 46.334934 | +0.001705 |
| step_rotation_deg_max | 136.537842 | 136.598846 | +0.061005 |
| step_translation_mean | 5.264674 | 5.261162 | -0.003511 |
| step_translation_max | 13.633723 | 13.597383 | -0.036341 |
| depth_conf_mean | 1.551369 | 1.513321 | -0.038048 |
| depth_conf_frame_mean_std | 0.217389 | 0.209631 | -0.007758 |
| depth_mean | 1.366789 | 1.349256 | -0.017533 |
| translation_norm_mean | 5.123201 | 5.120943 | -0.002259 |

## Interpretation

- This run is the first actual proof that cross-folder projection-shift
  distillation can train with full geometric supervision on the current GPU.
- The student-side proxy changes are still small.
- Translation and depth-confidence stability moved slightly in the intended
  direction.
- Rotation metrics stayed effectively flat.

## Practical Conclusion

The code path is now strong enough for the next real optimization pass:

1. longer pad-based training on a larger subset
2. stronger teacher/student viewpoint pairing
3. mixed crop/pad student schedule once the memory budget is under control
