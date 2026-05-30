# Aggregated Research Eval Summary

This file is generated from JSON reports under `research_eval/`.

## Temporal Sparsity

| label | num_frames_after_sampling | step_rotation_deg_mean | step_translation_mean | step_translation_std | depth_conf_frame_mean_std | frames_per_sec_inference |
|---|---|---|---|---|---|---|
| temporal_fps_1 | 24 | 1.1482 | 0.0381 | 0.0053 | 0.4459 | 6.0539 |
| temporal_fps_2 | 24 | 0.5922 | 0.0216 | 0.0035 | 0.1974 | 6.0683 |
| temporal_fps_2 | 12 | 0.4310 | 0.0210 | 0.0039 | 0.1120 | 6.7905 |

## Projection Shift

| label | num_frames_after_sampling | step_rotation_deg_mean | step_translation_mean | step_translation_std | depth_conf_frame_mean_std | frames_per_sec_inference |
|---|---|---|---|---|---|---|
| perspective | 24 | 1.1078 | 0.0382 | 0.0052 | 0.5761 | 2.2373 |
| widefov | 24 | 37.8099 | 3.3665 | 3.1474 | 1.0871 | 3.3633 |

## Viewpoint Break

| label | num_frames_after_sampling | step_rotation_deg_mean | step_translation_mean | step_translation_std | depth_conf_frame_mean_std | frames_per_sec_inference |
|---|---|---|---|---|---|---|
| control_24 | 24 | 0.5922 | 0.0216 | 0.0035 | 0.1974 | 6.0662 |
| drop_mid_24 | 16 | 0.6445 | 0.0324 | 0.0360 | 0.2067 | 6.6036 |

## Preprocess Ablation

| label | num_frames_after_sampling | step_rotation_deg_mean | step_translation_mean | step_translation_std | depth_conf_frame_mean_std | frames_per_sec_inference |
|---|---|---|---|---|---|---|
| raw_fisheye_right_crop | 24 | 37.6355 | 2.8838 | 2.7243 | 1.1354 | 5.0434 |
| raw_fisheye_right_pad | 24 | 38.7414 | 2.6990 | 2.6388 | 1.0071 | 3.5235 |
| widefov_crop | 24 | 37.8099 | 3.3665 | 3.1474 | 1.0871 | 3.4546 |
| widefov_pad | 24 | 37.8099 | 3.3665 | 3.1474 | 1.0871 | 3.4894 |
