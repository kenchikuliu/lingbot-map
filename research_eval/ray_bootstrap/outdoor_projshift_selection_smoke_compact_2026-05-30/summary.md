# Student-Selection Compact Smoke Summary

Date: 2026-05-30

## Goal

Validate two engineering changes before launching longer runs:

1. student-side selection metrics run correctly during training
2. bootstrap checkpoints save in compact ray-conditioning-only format instead
   of multi-GB full-model snapshots

## Training Setup

- teacher folder: `tmp_frames/outdoor_perspectives_8_az135_el000`
- student folder: `tmp_frames/outdoor_right_center_crops`
- teacher preprocess: `crop`
- student preprocess: `pad`
- frames: first `24`
- steps: `6`
- sequence length: `2`
- sequence stride: `1`
- selection eval every: `2`
- selection metric: `balanced_projection_shift`

## Result

- initial student-eval score: `1.000000`
- step `2`: `0.999905`
- step `4`: `0.999516`
- step `6`: `0.999150`

Compact checkpoint sizes:

- `ray_conditioning_bootstrap_best.pt`: `12,264` bytes
- `ray_conditioning_bootstrap_final.pt`: `12,270` bytes
- `ray_conditioning_bootstrap_best_student_eval.pt`: `12,406` bytes

This smoke run confirmed that:

- `postprocess + compute_proxy_metrics` works inside the trainer loop
- selection history is written to `summary.json`
- compact checkpoints can be loaded directly by `scripts/eval_sequence.py`
- student-side metric selection updates the best checkpoint as expected

## Practical Conclusion

This was the gating smoke test for the longer 138-frame selection run, and it
passed cleanly.
