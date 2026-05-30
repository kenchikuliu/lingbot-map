# Projection-Shift Cross-Folder Smoke Summary

Date: 2026-05-30

## Goal

Verify that the bootstrap trainer can distill across two different input
folders:

- teacher: perspective subviews from the same 360 capture set
- student: raw right-fisheye center crops

This run tests the new cross-folder training path, not final robustness.

## Training Setup

- teacher folder:
  `tmp_frames/outdoor_perspectives_8_az135_el000`
- student folder:
  `tmp_frames/outdoor_right_center_crops`
- pairing mode: `ordered`
- frames: first 24
- teacher preprocess: `crop`
- student preprocess: `crop`
- steps: 10
- sequence length: 4
- sequence stride: 2
- teacher CPU offload: enabled

## Training Result

- best loss: `0.029066`
- final loss: `0.098853`
- final gate mean: `0.001175`
- peak GPU allocated: `10.93 GB`

Files:

- training log:
  [summary.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_smoke_2026-05-30/summary.json)
- post-train eval:
  [eval_post_student.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_smoke_2026-05-30/eval_post_student.json)

## Important Behavior

- the new teacher/student dual-source path worked
- frame pairing worked on same-named files across the two folders
- depth and depth-confidence losses were skipped on every step

Why they were skipped:

- teacher crop view became `518x518`
- student crop view became `518x364`
- that shape mismatch disables depth/depth-confidence distillation by design

So this run only verified:

- pose distillation
- FoV distillation
- cross-folder data plumbing

## Student-Side Proxy Change

Compared against the existing student-folder baseline report:

- baseline:
  [widefov.json](/media/slam/My%20Passport/lingbot-map/research_eval/preprocess_ablation/raw_fisheye_right_crop/widefov.json)
- post-train:
  [eval_post_student.json](/media/slam/My%20Passport/lingbot-map/research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_smoke_2026-05-30/eval_post_student.json)

Observed deltas:

- `step_rotation_deg_mean`: `-0.006035`
- `step_rotation_deg_max`: `-0.129074`
- `step_translation_mean`: `-0.003942`
- `step_translation_max`: `-0.007267`
- `depth_conf_mean`: `+0.007980`
- `translation_norm_mean`: `-0.005613`

## Conclusion

This run is useful because it proves the code path for true projection-shift
distillation is now available.

It is **not** enough yet to claim strong optimization because the geometric
supervision was still incomplete.
