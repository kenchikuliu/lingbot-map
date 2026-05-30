# RoAM3R Ray Bootstrap Plan v1

Date: 2026-05-27

## Goal

Train the newly added ray-conditioning branch without destabilizing the public
LingBot-MAP checkpoint.

This is a **bootstrap stage**, not the final robustness training stage.

## Principle

- teacher: original LingBot-MAP checkpoint behavior
- student: same checkpoint + ray-conditioning branch enabled
- trainable parameters:
  - `aggregator.ray_conditioning_proj.*`
  - `aggregator.ray_conditioning_gate`
- frozen:
  - backbone
  - camera head
  - depth head

## Why this stage exists

The new ray branch is zero-initialized for checkpoint safety. That preserves
baseline behavior, but it also means the new pathway has no learned utility.

A first bootstrap stage is needed to:

1. open the gate from zero safely
2. make the ray pathway numerically trainable
3. preserve baseline predictions on standard perspective video

## Script

- [train_ray_conditioning_bootstrap.py](/media/slam/My%20Passport/lingbot-map/scripts/train_ray_conditioning_bootstrap.py)

## Stage 1

Identity distillation on perspective sequences:

- teacher input: `pinhole`
- student input: `pinhole`
- optional student intrinsics:
  - default pseudo intrinsics, or
  - explicit calibrated intrinsics file

Losses:

- pose center + quaternion distillation
- FoV distillation
- depth distillation
- depth-confidence distillation

Expected outcome:

- the gate becomes non-zero
- the ray projection layer learns a numerically stable initial mapping
- the student remains close to baseline on ordinary perspective video

## Stage 2

Intrinsics-jitter distillation:

- teacher input: canonical perspective preprocessing
- student input: perspective preprocessing with controlled intrinsics variation
- optional alternate student view: `pad` preprocessing mixed against the
  teacher's canonical `crop` view

Current script support:

- `--student_focal_jitter_pct`
- `--student_principal_point_jitter_px`
- `--jitter_schedule constant|linear_ramp|cosine_ramp`
- `--jitter_ramp_steps`
- `--student_framewise_focal_drift_pct`
- `--student_framewise_principal_point_drift_px`
- `--student_dual_view_preprocess_mode none|crop|pad`
- `--dual_view_distill_prob`

Purpose:

- teach the student that camera geometry changes should be explained by ray
  metadata rather than by drifting the latent geometry estimate
- make the student tolerate preprocessing/view changes without immediately
  collapsing into a crop-only assumption

Current implementation note:

- when the alternate student view changes spatial tensor shape, depth and
  depth-confidence distillation are skipped automatically for that batch
- pose and FoV distillation still apply

## Stage 3

Projection-shift adaptation:

- teacher: perspective baseline or stronger offline reconstructor
- student: 360-derived / wide-FoV / projection-shifted input

Current script support:

- role-specific sequence sources:
  - `--teacher_image_folder`
  - `--teacher_video_path`
  - `--student_image_folder`
  - `--student_video_path`
- role-specific preprocessing:
  - `--teacher_preprocess_mode`
  - `--student_preprocess_mode`
- frame alignment:
  - `--pairing_mode ordered|basename`

This stage is where the paper claim begins, but it should come only after
Stage 1 and Stage 2 are stable.

## Recommended first run

```bash
/home/slam/anaconda3/envs/paper_gen/bin/python scripts/train_ray_conditioning_bootstrap.py \
  --image_folder "tmp_frames/video_20260418_1906051_ffmpeg_frames_fps2" \
  --model_path "checkpoints/lingbot-map.pt" \
  --first_k 24 \
  --use_sdpa \
  --sequence_length 4 \
  --sequence_stride 2 \
  --max_steps 30 \
  --offload_teacher_to_cpu \
  --student_input_camera_model pinhole \
  --student_input_intrinsics_file "research_eval/camera_inputs/video_20260418_pseudo_intrinsics_1920x1080.json" \
  --output_dir "research_eval/ray_bootstrap/video_fps2_24_smoke"
```

## Recommended stronger Stage 2 run

```bash
/home/slam/anaconda3/envs/paper_gen/bin/python scripts/train_ray_conditioning_bootstrap.py \
  --image_folder "tmp_frames/video_20260418_1906051_ffmpeg_frames_fps2" \
  --model_path "checkpoints/lingbot-map.pt" \
  --first_k 24 \
  --use_sdpa \
  --sequence_length 4 \
  --sequence_stride 2 \
  --max_steps 20 \
  --offload_teacher_to_cpu \
  --student_input_camera_model pinhole \
  --student_input_intrinsics_file "research_eval/camera_inputs/video_20260418_pseudo_intrinsics_1920x1080.json" \
  --student_focal_jitter_pct 0.10 \
  --student_principal_point_jitter_px 16 \
  --student_framewise_focal_drift_pct 0.05 \
  --student_framewise_principal_point_drift_px 8 \
  --jitter_schedule cosine_ramp \
  --jitter_ramp_steps 20 \
  --student_dual_view_preprocess_mode pad \
  --dual_view_distill_prob 0.5 \
  --output_dir "research_eval/ray_bootstrap/video_fps2_24_step20_dualview"
```

## Recommended Stage 3 smoke run

Use paired teacher/student image folders from the same scene:

- teacher:
  `tmp_frames/outdoor_perspectives_8_az135_el000`
- student:
  `tmp_frames/outdoor_right_center_crops`

First verify the cross-folder path with conservative crop preprocessing:

```bash
/home/slam/anaconda3/envs/paper_gen/bin/python scripts/train_ray_conditioning_bootstrap.py \
  --teacher_image_folder "tmp_frames/outdoor_perspectives_8_az135_el000" \
  --student_image_folder "tmp_frames/outdoor_right_center_crops" \
  --model_path "checkpoints/lingbot-map.pt" \
  --first_k 24 \
  --use_sdpa \
  --sequence_length 4 \
  --sequence_stride 2 \
  --max_steps 10 \
  --offload_teacher_to_cpu \
  --teacher_input_camera_model pinhole \
  --student_input_camera_model pinhole \
  --output_dir "research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_smoke"
```

Then run a lighter square-pad recipe so depth and depth-confidence losses can
participate:

```bash
/home/slam/anaconda3/envs/paper_gen/bin/python scripts/train_ray_conditioning_bootstrap.py \
  --teacher_image_folder "tmp_frames/outdoor_perspectives_8_az135_el000" \
  --student_image_folder "tmp_frames/outdoor_right_center_crops" \
  --student_preprocess_mode pad \
  --model_path "checkpoints/lingbot-map.pt" \
  --first_k 12 \
  --use_sdpa \
  --sequence_length 2 \
  --sequence_stride 1 \
  --num_scale_frames 1 \
  --camera_num_iterations 1 \
  --max_steps 6 \
  --offload_teacher_to_cpu \
  --teacher_input_camera_model pinhole \
  --student_input_camera_model pinhole \
  --output_dir "research_eval/ray_bootstrap/outdoor_projshift_teacher_az135_rightcrop_pad_smoke12"
```

Observed practical constraint on this 16 GB GPU:

- full 24-frame `pad` projection-shift distillation can OOM once depth-head
  supervision is active
- a smaller `first_k/sequence_length/num_scale_frames` recipe is currently the
  reliable path

## Success criteria

- training loss decreases over the first 20-50 steps
- gate magnitude becomes stably non-zero
- re-evaluating the saved checkpoint on baseline perspective video does not
  produce a large stability regression

## Warning

This bootstrap recipe does **not** prove camera-model robustness by itself.
It only creates a safe optimization path for the new projection-aware module.
