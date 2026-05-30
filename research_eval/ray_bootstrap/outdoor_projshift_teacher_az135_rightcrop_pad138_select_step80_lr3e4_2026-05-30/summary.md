# Projection-Shift Student-Selection Summary

Date: 2026-05-30

## Goal

Extend the student-side checkpoint-selection recipe from `40` to `80` steps on
the same cross-folder projection-shift bootstrap setup:

- teacher folder: `tmp_frames/outdoor_perspectives_8_az135_el000`
- student folder: `tmp_frames/outdoor_right_center_crops`
- selection target: first `24` student frames with `pad` preprocessing

The objective was to check whether the student-side proxy metric had actually
plateaued at `40` steps or whether the longer run still had room to improve.

## Training Setup

- teacher preprocess: `crop`
- student preprocess: `pad`
- paired frames: `138`
- pairing mode: `ordered`
- steps: `80`
- sequence length: `2`
- sequence stride: `1`
- scale frames: `1`
- camera iterations: `1`
- lr: `3e-4`
- teacher CPU offload: enabled
- selection eval every: `5` steps
- selection metric: `balanced_projection_shift`
- selection goal: `min`
- early-stop patience: `4`

Important implementation detail:

- selection eval was requested in `windowed` mode
- the current training model is the streaming variant, so selection eval
  correctly fell back to `streaming`

## Training Result

- best distillation loss: `0.061435`
- final loss: `0.148369`
- best student-eval score: `0.984866`
- best student-eval step: `80`
- final gate mean: `0.001274`
- early stopping triggered: `no`

Selection trajectory:

- step `0`: `1.000000`
- step `5`: `0.999498`
- step `10`: `0.998128`
- step `15`: `0.996963`
- step `20`: `0.995879`
- step `25`: `0.994622`
- step `30`: `0.993301`
- step `35`: `0.992637`
- step `40`: `0.991901`
- step `45`: `0.991032`
- step `50`: `0.989652`
- step `55`: `0.989118`
- step `60`: `0.988539`
- step `65`: `0.987154`
- step `70`: `0.986474`
- step `75`: `0.985605`
- step `80`: `0.984866`

This run also never plateaued on the chosen student-side metric within `80`
steps.

## Student Eval Comparison

All values below use the same 24-frame student-side `pad` evaluation target.

| run | rot_mean | rot_max | trans_mean | trans_max | depth_conf_mean | depth_conf_frame_std | depth_mean | trans_norm_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline24 | 38.429668 | 136.537842 | 4.314075 | 13.633723 | 2.047986 | 0.616918 | 1.250021 | 5.259009 |
| pad138_select40_best | 38.441055 | 136.588623 | 4.300098 | 13.578547 | 2.008447 | 0.602395 | 1.239871 | 5.244617 |
| pad138_select40_final | 38.442013 | 136.661545 | 4.304957 | 13.570285 | 2.002061 | 0.599829 | 1.240162 | 5.248873 |
| pad138_select80_best | 38.450600 | 136.414703 | 4.304505 | 13.581787 | 2.008552 | 0.602869 | 1.240605 | 5.248595 |
| pad138_select80_final | 38.482208 | 136.437500 | 4.299042 | 13.533513 | 1.966763 | 0.586222 | 1.230179 | 5.241807 |

Notes:

- `pad138_select80_best_student_eval` matched `pad138_select80_final` on this
  subset because the student-side score kept improving through step `80`
- `pad138_select80_best` is the best training-loss checkpoint
- `pad138_select80_final` is also the best student-eval checkpoint

## Interpretation

Compared with the earlier `40`-step selection run, the `80`-step run showed a
split outcome:

- the best-loss checkpoint stayed nearly flat on the held-out student subset
- the final or student-selected checkpoint improved clearly

Relative to `pad138_select40_final`, `pad138_select80_final` achieved:

- better `step_translation_mean`
- better `step_translation_max`
- better `depth_conf_mean`
- better `depth_conf_frame_mean_std`
- better `depth_mean`
- better `step_rotation_deg_max`
- slightly worse `step_rotation_deg_mean`

So the longer run did not just keep lowering the internal selection score. It
also translated into a stronger held-out student-side checkpoint once the full
`80` steps completed.

## Practical Conclusion

This run changes the current recommendation:

1. the `40`-step recipe was still too short
2. student-side selection continues to improve through at least `80` steps
3. the current best long-run student-selected checkpoint is
   `pad138_select80_final`

The next optimization pass should no longer ask whether selection is useful.
That part is established. The next question is whether to:

1. extend again to `120` steps to find the actual plateau
2. retune the selection metric to penalize rotation mean drift more directly
3. replace the streaming fallback with a true windowed student-side selector
