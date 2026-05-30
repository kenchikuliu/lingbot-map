# Projection-Shift Pad-24 Comparison

Date: 2026-05-30

## Scope

This note compares several cross-folder projection-shift finetuning runs on the
same student-side evaluation target:

- student eval folder: `tmp_frames/outdoor_right_center_crops`
- eval frames: first 24
- eval preprocess: `pad`
- eval mode: windowed
- scale frames: 1
- camera iterations: 1

All training runs use:

- teacher folder: `tmp_frames/outdoor_perspectives_8_az135_el000`
- student folder: `tmp_frames/outdoor_right_center_crops`
- pairing mode: `ordered`

## Compared Runs

1. baseline public checkpoint
2. `pad24_best`
   - 24-frame training subset
   - 20 steps
3. `pad24_final`
   - same run, final checkpoint
4. `pad138_lr1e3_best`
   - 138-frame training pool
   - 30 steps
   - lr `1e-3`
5. `pad138_lr3e4_best`
   - 138-frame training pool
   - 20 steps
   - lr `3e-4`
6. `pad138_select40_best`
   - 138-frame training pool
   - 40 steps
   - lr `3e-4`
   - student-side selection enabled
7. `pad138_select40_final`
   - same run, final checkpoint
   - also matched the best student-eval checkpoint on this subset
8. `pad138_select80_best`
   - 138-frame training pool
   - 80 steps
   - lr `3e-4`
   - student-side selection enabled
9. `pad138_select80_final`
   - same run, final checkpoint
   - also matched the best student-eval checkpoint on this subset

## Metric Table

| run | rot_mean | rot_max | trans_mean | trans_max | depth_conf_mean | depth_conf_frame_std | depth_mean | trans_norm_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline24 | 38.429668 | 136.537842 | 4.314075 | 13.633723 | 2.047986 | 0.616918 | 1.250021 | 5.259009 |
| pad24_best | 38.472996 | 136.743317 | 4.333354 | 13.610612 | 1.900653 | 0.551470 | 1.227539 | 5.282252 |
| pad24_final | 38.458229 | 136.748734 | 4.334539 | 13.629212 | 1.881228 | 0.543703 | 1.222741 | 5.284243 |
| pad138_lr1e3_best | 38.464390 | 136.527573 | 4.301548 | 13.574484 | 2.005896 | 0.601128 | 1.240824 | 5.245010 |
| pad138_lr3e4_best | 38.434792 | 136.440384 | 4.301320 | 13.584756 | 2.033958 | 0.611169 | 1.246335 | 5.246917 |
| pad138_select40_best | 38.441055 | 136.588623 | 4.300098 | 13.578547 | 2.008447 | 0.602395 | 1.239871 | 5.244617 |
| pad138_select40_final | 38.442013 | 136.661545 | 4.304957 | 13.570285 | 2.002061 | 0.599829 | 1.240162 | 5.248873 |
| pad138_select80_best | 38.450600 | 136.414703 | 4.304505 | 13.581787 | 2.008552 | 0.602869 | 1.240605 | 5.248595 |
| pad138_select80_final | 38.482208 | 136.437500 | 4.299042 | 13.533513 | 1.966763 | 0.586222 | 1.230179 | 5.241807 |

## Delta vs Baseline

### pad24_best

- `step_rotation_deg_mean`: `+0.043327`
- `step_rotation_deg_max`: `+0.205475`
- `step_translation_mean`: `+0.019279`
- `step_translation_max`: `-0.023111`
- `depth_conf_mean`: `-0.147333`
- `depth_conf_frame_mean_std`: `-0.065448`
- `depth_mean`: `-0.022482`
- `translation_norm_mean`: `+0.023242`

### pad24_final

- `step_rotation_deg_mean`: `+0.028561`
- `step_rotation_deg_max`: `+0.210892`
- `step_translation_mean`: `+0.020463`
- `step_translation_max`: `-0.004511`
- `depth_conf_mean`: `-0.166758`
- `depth_conf_frame_mean_std`: `-0.073215`
- `depth_mean`: `-0.027280`
- `translation_norm_mean`: `+0.025234`

### pad138_lr1e3_best

- `step_rotation_deg_mean`: `+0.034721`
- `step_rotation_deg_max`: `-0.010269`
- `step_translation_mean`: `-0.012527`
- `step_translation_max`: `-0.059239`
- `depth_conf_mean`: `-0.042090`
- `depth_conf_frame_mean_std`: `-0.015790`
- `depth_mean`: `-0.009197`
- `translation_norm_mean`: `-0.013999`

### pad138_lr3e4_best

- `step_rotation_deg_mean`: `+0.005123`
- `step_rotation_deg_max`: `-0.097458`
- `step_translation_mean`: `-0.012756`
- `step_translation_max`: `-0.048967`
- `depth_conf_mean`: `-0.014028`
- `depth_conf_frame_mean_std`: `-0.005749`
- `depth_mean`: `-0.003686`
- `translation_norm_mean`: `-0.012092`

### pad138_select40_best

- `step_rotation_deg_mean`: `+0.011387`
- `step_rotation_deg_max`: `+0.050781`
- `step_translation_mean`: `-0.013977`
- `step_translation_max`: `-0.055176`
- `depth_conf_mean`: `-0.039539`
- `depth_conf_frame_mean_std`: `-0.014523`
- `depth_mean`: `-0.010150`
- `translation_norm_mean`: `-0.014392`

### pad138_select40_final

- `step_rotation_deg_mean`: `+0.012344`
- `step_rotation_deg_max`: `+0.123703`
- `step_translation_mean`: `-0.009118`
- `step_translation_max`: `-0.063438`
- `depth_conf_mean`: `-0.045925`
- `depth_conf_frame_mean_std`: `-0.017089`
- `depth_mean`: `-0.009859`
- `translation_norm_mean`: `-0.010137`

### pad138_select80_best

- `step_rotation_deg_mean`: `+0.020932`
- `step_rotation_deg_max`: `-0.123139`
- `step_translation_mean`: `-0.009570`
- `step_translation_max`: `-0.051936`
- `depth_conf_mean`: `-0.039434`
- `depth_conf_frame_mean_std`: `-0.014049`
- `depth_mean`: `-0.009416`
- `translation_norm_mean`: `-0.010414`

### pad138_select80_final

- `step_rotation_deg_mean`: `+0.052540`
- `step_rotation_deg_max`: `-0.100342`
- `step_translation_mean`: `-0.015033`
- `step_translation_max`: `-0.100210`
- `depth_conf_mean`: `-0.081223`
- `depth_conf_frame_mean_std`: `-0.030696`
- `depth_mean`: `-0.019842`
- `translation_norm_mean`: `-0.017202`

## Interpretation

Two distinct behaviors showed up.

### 24-frame finetune

- strongest reduction in `depth_conf_mean`
- strongest reduction in `depth_conf_frame_mean_std`
- but translation mean and rotation metrics drifted slightly worse

This looks like a more aggressive local adaptation to the short evaluation
subset.

### 138-frame finetune

- smaller depth-confidence gains
- better translation mean
- better translation max
- better or flat rotation max

This looks more conservative and more balanced.

### 138-frame finetune with student-side selection

- best student-side score improved monotonically from `1.000000` to `0.991940`
- translation mean improved slightly beyond the earlier `pad138_lr3e4_best`
- translation max improved further
- depth-confidence metrics improved further
- rotation drift became slightly worse than the earlier 20-step balanced run

This looks like a useful next-stage recipe if translation/depth stability is
the primary objective.

### Extending selection from 40 to 80 steps

- the student-side score kept improving from `0.991940` at step `40` to
  `0.984866` at step `80`
- the best-loss checkpoint at `80` steps did not improve much on this subset
- the final or student-selected checkpoint at `80` steps improved clearly over
  `pad138_select40_final`
- translation max, translation mean, depth confidence, and depth mean all
  improved further
- rotation mean drift became the main remaining tradeoff

This makes the longer selection run the strongest current recipe when the goal
is student-side translation and depth stability rather than the smallest
possible rotation-mean drift.

## Current Best Balanced Checkpoint

If the priority is a balanced proxy improvement rather than the single largest
depth-confidence gain, the current best candidate is:

- `pad138_select80_final` for the strongest overall student-side proxy result,
  or
- `pad138_lr3e4_best` if rotation-mean conservatism matters more than the extra
  translation or depth gain

If the priority is strongest depth-confidence improvement on this exact subset,
the current best candidate is:

- `pad24_final` or `pad138_select80_final`

If the priority is strongest translation-max improvement among the longer
138-frame runs, the current best candidate is:

- `pad138_select80_final`

## Practical Conclusion

The current training system is now capable of three things that were not
available before:

1. teacher/student cross-folder projection-shift distillation
2. full geometric supervision with `pad` student preprocessing
3. student-side validation-based checkpoint selection on top of that training
   path

The next optimization step should be to extend the selection-based 138-frame
run beyond `80` steps or retune the selection weights for rotation-sensitive
selection, since the student-side score was still improving at the end.
