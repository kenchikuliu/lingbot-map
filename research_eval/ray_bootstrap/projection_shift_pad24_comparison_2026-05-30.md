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

## Metric Table

| run | rot_mean | rot_max | trans_mean | trans_max | depth_conf_mean | depth_conf_frame_std | depth_mean | trans_norm_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline24 | 38.429668 | 136.537842 | 4.314075 | 13.633723 | 2.047986 | 0.616918 | 1.250021 | 5.259009 |
| pad24_best | 38.472996 | 136.743317 | 4.333354 | 13.610612 | 1.900653 | 0.551470 | 1.227539 | 5.282252 |
| pad24_final | 38.458229 | 136.748734 | 4.334539 | 13.629212 | 1.881228 | 0.543703 | 1.222741 | 5.284243 |
| pad138_lr1e3_best | 38.464390 | 136.527573 | 4.301548 | 13.574484 | 2.005896 | 0.601128 | 1.240824 | 5.245010 |
| pad138_lr3e4_best | 38.434792 | 136.440384 | 4.301320 | 13.584756 | 2.033958 | 0.611169 | 1.246335 | 5.246917 |

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

## Current Best Balanced Checkpoint

If the priority is a balanced proxy improvement rather than the single largest
depth-confidence gain, the current best candidate is:

- `pad138_lr3e4_best` for the smallest rotation drift with improved translation
  stability, or
- `pad138_lr1e3_best` if translation max matters more than rotation mean drift

If the priority is strongest depth-confidence improvement on this exact subset,
the current best candidate is:

- `pad24_final`

## Practical Conclusion

The current training system is now capable of three things that were not
available before:

1. teacher/student cross-folder projection-shift distillation
2. full geometric supervision with `pad` student preprocessing
3. meaningful tradeoff exploration between local adaptation and broader-scene
   regularization

The next optimization step should be checkpoint selection based on explicit
student-side validation metrics rather than distillation loss alone.
