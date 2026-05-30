# Projection-Shift Student-Selection Summary

Date: 2026-05-30

## Goal

Add student-side checkpoint selection and longer training to the cross-folder
projection-shift bootstrap recipe:

- teacher folder: `tmp_frames/outdoor_perspectives_8_az135_el000`
- student folder: `tmp_frames/outdoor_right_center_crops`
- selection target: first 24 student frames with `pad` preprocessing

The intent was to stop selecting checkpoints by distillation loss alone and
switch to explicit student-side proxy metrics.

## Training Setup

- teacher preprocess: `crop`
- student preprocess: `pad`
- paired frames: `138`
- pairing mode: `ordered`
- steps: `40`
- sequence length: `2`
- sequence stride: `1`
- scale frames: `1`
- camera iterations: `1`
- lr: `3e-4`
- teacher CPU offload: enabled
- selection eval every: `5` steps
- selection metric: `balanced_projection_shift`
- selection goal: `min`
- early-stop patience: `3`

Important implementation detail:

- selection eval was requested in `windowed` mode
- the current training model is the streaming variant, so selection eval
  correctly fell back to `streaming`

## Training Result

- best distillation loss: `0.061456`
- final loss: `0.239271`
- best student-eval score: `0.991940`
- best student-eval step: `40`
- final gate mean: `0.001160`
- early stopping triggered: `no`

Selection trajectory:

- step `0`: `1.000000`
- step `5`: `0.999049`
- step `10`: `0.997706`
- step `15`: `0.997007`
- step `20`: `0.995227`
- step `25`: `0.994746`
- step `30`: `0.993740`
- step `35`: `0.993221`
- step `40`: `0.991940`

This run never plateaued on the chosen student-side metric within 40 steps.

## Student Eval Comparison

All values below use the same 24-frame student-side `pad` evaluation target.

| run | rot_mean | rot_max | trans_mean | trans_max | depth_conf_mean | depth_conf_frame_std | depth_mean | trans_norm_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline24 | 38.429668 | 136.537842 | 4.314075 | 13.633723 | 2.047986 | 0.616918 | 1.250021 | 5.259009 |
| pad138_lr3e4_best | 38.434792 | 136.440384 | 4.301320 | 13.584756 | 2.033958 | 0.611169 | 1.246335 | 5.246917 |
| pad138_select40_best | 38.441055 | 136.588623 | 4.300098 | 13.578547 | 2.008447 | 0.602395 | 1.239871 | 5.244617 |
| pad138_select40_final | 38.442013 | 136.661545 | 4.304957 | 13.570285 | 2.002061 | 0.599829 | 1.240162 | 5.248873 |

Notes:

- `pad138_select40_best_student_eval` matched `pad138_select40_final` on this
  subset because the student-side score kept improving through step `40`
- `pad138_select40_best` is the best training-loss checkpoint
- `pad138_select40_final` is also the best student-eval checkpoint

## Interpretation

Compared with the earlier 20-step `pad138_lr3e4_best` run, the new
selection-capable 40-step run produced a clearer tradeoff:

- better `step_translation_mean`
- better `step_translation_max`
- better `depth_conf_mean`
- better `depth_conf_frame_mean_std`
- better `depth_mean`
- slightly worse rotation metrics

In other words, longer training with explicit student-side monitoring continued
to push the student toward better translation and depth-confidence stability on
the held-out student subset, but not yet toward the best rotation max.

## Practical Conclusion

This run validates two changes:

1. student-side proxy-metric checkpoint selection works end-to-end
2. the 138-frame recipe was still improving at step `40`, so the previous
   20-step run was stopping too early

The next optimization pass should extend this recipe rather than revert it:

1. run `60-80` steps with the same student-side selection
2. test a rotation-heavier selection weighting if rotation max matters more
3. compare streaming selection with a true windowed eval model once available
