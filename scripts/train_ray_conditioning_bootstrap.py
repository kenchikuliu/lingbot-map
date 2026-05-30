#!/usr/bin/env python3
"""Minimal bootstrap trainer for LingBot-MAP ray conditioning.

This script is intentionally small and conservative:

- teacher: original checkpoint behavior
- student: same checkpoint + ray-conditioning branch enabled
- trainable params: only ``aggregator.ray_conditioning_*``
- objective: distill teacher pose/depth/confidence on short frame chunks

The goal is not to claim camera robustness yet. The goal is to make the new
projection-aware side channel trainable with a reproducible first-stage recipe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

legacy_alloc_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", legacy_alloc_conf or "expandable_segments:True")

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo import load_images, load_model
from lingbot_map.utils import load_input_intrinsics
from lingbot_map.utils.load_fn import load_and_preprocess_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap trainer for LingBot-MAP ray conditioning")

    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--preprocess_mode", type=str, default="crop", choices=["crop", "pad"])
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true", default=False)

    parser.add_argument("--teacher_input_camera_model", type=str, default="pinhole")
    parser.add_argument("--teacher_input_intrinsics_file", type=str, default=None)
    parser.add_argument("--student_input_camera_model", type=str, default="pinhole")
    parser.add_argument("--student_input_intrinsics_file", type=str, default=None)

    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--sequence_stride", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ray_gate_init", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--offload_teacher_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the frozen teacher on CPU between training steps to reduce GPU memory pressure.",
    )

    parser.add_argument("--loss_w_pose", type=float, default=1.0)
    parser.add_argument("--loss_w_fov", type=float, default=0.1)
    parser.add_argument("--loss_w_depth", type=float, default=1.0)
    parser.add_argument("--loss_w_depth_conf", type=float, default=0.05)
    parser.add_argument(
        "--student_focal_jitter_pct",
        type=float,
        default=0.0,
        help="Uniform multiplicative jitter range for student focal lengths. "
             "For example 0.05 samples scale factors in [0.95, 1.05].",
    )
    parser.add_argument(
        "--student_principal_point_jitter_px",
        type=float,
        default=0.0,
        help="Uniform additive jitter range in model-input pixels for student principal point.",
    )
    parser.add_argument(
        "--jitter_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear_ramp", "cosine_ramp"],
        help="Schedule for scaling student camera perturbations over training steps.",
    )
    parser.add_argument(
        "--jitter_ramp_steps",
        type=int,
        default=0,
        help="Ramp length for non-constant jitter schedules. 0 means use max_steps.",
    )
    parser.add_argument(
        "--student_framewise_focal_drift_pct",
        type=float,
        default=0.0,
        help="Per-sequence focal drift magnitude across time inside a chunk.",
    )
    parser.add_argument(
        "--student_framewise_principal_point_drift_px",
        type=float,
        default=0.0,
        help="Per-sequence principal-point drift magnitude across time inside a chunk.",
    )
    parser.add_argument(
        "--student_dual_view_preprocess_mode",
        type=str,
        default="none",
        choices=["none", "crop", "pad"],
        help="Optional alternate preprocess mode for student-view distillation.",
    )
    parser.add_argument(
        "--dual_view_distill_prob",
        type=float,
        default=0.0,
        help="Probability of using the alternate student preprocess view on a training step.",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model_args(base_args: argparse.Namespace, *, enable_ray_conditioning: bool, gate_value: float | None):
    args = SimpleNamespace(**vars(base_args))
    args.mode = "streaming"
    args.enable_ray_conditioning = enable_ray_conditioning
    args.ray_use_default_intrinsics = True
    args.ray_conditioning_gate_value = gate_value
    return args


def _maybe_cast_aggregator(model: torch.nn.Module) -> torch.dtype:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
    return dtype


def _freeze_non_ray_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for param in model.parameters():
        param.requires_grad_(False)

    trainable = []
    aggregator = model.aggregator
    if getattr(aggregator, "ray_conditioning_proj", None) is not None:
        for param in aggregator.ray_conditioning_proj.parameters():
            param.requires_grad_(True)
            trainable.append(param)
    if getattr(aggregator, "ray_conditioning_gate", None) is not None:
        aggregator.ray_conditioning_gate.requires_grad_(True)
        trainable.append(aggregator.ray_conditioning_gate)

    if not trainable:
        raise ValueError("Student model has no trainable ray-conditioning parameters")
    return trainable


def _build_all_input_kwargs(
    image_paths: list[str],
    *,
    image_size: int,
    patch_size: int,
    preprocess_mode: str,
    camera_model: str,
    intrinsics_file: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"input_camera_model": camera_model}
    if intrinsics_file:
        kwargs["input_intrinsics"] = load_input_intrinsics(
            image_paths,
            intrinsics_file,
            image_size=image_size,
            patch_size=patch_size,
            preprocess_mode=preprocess_mode,
        )
    return kwargs


def _maybe_load_dual_view_images(
    image_paths: list[str],
    *,
    image_size: int,
    patch_size: int,
    preprocess_mode: str,
) -> torch.Tensor | None:
    """Load an alternate student preprocess view when requested."""
    if preprocess_mode == "none":
        return None
    images = load_and_preprocess_images(
        image_paths,
        mode=preprocess_mode,
        image_size=image_size,
        patch_size=patch_size,
    )
    return images


def _slice_model_kwargs(model: torch.nn.Module, kwargs: dict[str, Any], start_idx: int, end_idx: int) -> dict[str, Any]:
    return model._slice_temporal_model_inputs(kwargs, start_idx, end_idx, batch_size=1)


def _build_sequence_starts(num_frames: int, sequence_length: int, sequence_stride: int) -> list[int]:
    if num_frames <= sequence_length:
        return [0]
    starts = list(range(0, num_frames - sequence_length + 1, max(sequence_stride, 1)))
    if starts[-1] != num_frames - sequence_length:
        starts.append(num_frames - sequence_length)
    return starts


def _compute_distill_loss(
    student_pred: dict[str, torch.Tensor],
    teacher_pred: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: dict[str, torch.Tensor] = {}

    losses["pose_xyz_quat"] = F.smooth_l1_loss(student_pred["pose_enc"][..., :7], teacher_pred["pose_enc"][..., :7])
    losses["pose_fov"] = F.smooth_l1_loss(student_pred["pose_enc"][..., 7:], teacher_pred["pose_enc"][..., 7:])

    depth_shape_match = (
        "depth" in student_pred and "depth" in teacher_pred
        and student_pred["depth"].shape == teacher_pred["depth"].shape
    )
    if depth_shape_match:
        losses["depth"] = F.smooth_l1_loss(student_pred["depth"], teacher_pred["depth"])
    else:
        losses["depth"] = torch.zeros((), device=student_pred["pose_enc"].device)

    depth_conf_shape_match = (
        "depth_conf" in student_pred and "depth_conf" in teacher_pred
        and student_pred["depth_conf"].shape == teacher_pred["depth_conf"].shape
    )
    if depth_conf_shape_match:
        losses["depth_conf"] = F.l1_loss(student_pred["depth_conf"], teacher_pred["depth_conf"])
    else:
        losses["depth_conf"] = torch.zeros((), device=student_pred["pose_enc"].device)

    total = (
        args.loss_w_pose * losses["pose_xyz_quat"]
        + args.loss_w_fov * losses["pose_fov"]
        + args.loss_w_depth * losses["depth"]
        + args.loss_w_depth_conf * losses["depth_conf"]
    )
    scalar_losses = {key: float(value.detach().item()) for key, value in losses.items()}
    scalar_losses["total"] = float(total.detach().item())
    scalar_losses["used_depth_loss"] = float(depth_shape_match)
    scalar_losses["used_depth_conf_loss"] = float(depth_conf_shape_match)
    return total, scalar_losses


def _jitter_schedule_scale(step: int, args: argparse.Namespace) -> float:
    """Return the current perturbation-strength multiplier."""
    if args.jitter_schedule == "constant":
        return 1.0

    ramp_steps = args.jitter_ramp_steps if args.jitter_ramp_steps > 0 else args.max_steps
    progress = min(max(step / max(ramp_steps, 1), 0.0), 1.0)

    if args.jitter_schedule == "linear_ramp":
        return progress
    if args.jitter_schedule == "cosine_ramp":
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    raise ValueError(f"Unsupported jitter schedule: {args.jitter_schedule}")


def _sample_student_intrinsics(
    intrinsics: torch.Tensor,
    *,
    focal_jitter_pct: float,
    principal_point_jitter_px: float,
    framewise_focal_drift_pct: float,
    framewise_principal_point_drift_px: float,
) -> torch.Tensor:
    """Apply small randomized calibration perturbations to student intrinsics."""
    if (
        focal_jitter_pct <= 0
        and principal_point_jitter_px <= 0
        and framewise_focal_drift_pct <= 0
        and framewise_principal_point_drift_px <= 0
    ):
        return intrinsics

    intrinsics_aug = intrinsics.clone()
    device = intrinsics_aug.device
    dtype = intrinsics_aug.dtype
    leading_shape = tuple(intrinsics_aug.shape[:-3])
    sequence_length = intrinsics_aug.shape[-3]
    alpha = None
    if sequence_length > 1:
        alpha = torch.linspace(-1.0, 1.0, steps=sequence_length, device=device, dtype=dtype)

    def rand_uniform(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape if shape else (), device=device, dtype=dtype).uniform_(-1.0, 1.0)

    if focal_jitter_pct > 0:
        fx_scale = (1.0 + rand_uniform(leading_shape) * focal_jitter_pct).unsqueeze(-1).expand(*leading_shape, sequence_length)
        fy_scale = (1.0 + rand_uniform(leading_shape) * focal_jitter_pct).unsqueeze(-1).expand(*leading_shape, sequence_length)
        if alpha is not None and framewise_focal_drift_pct > 0:
            fx_scale = fx_scale + rand_uniform(leading_shape).unsqueeze(-1) * framewise_focal_drift_pct * alpha
            fy_scale = fy_scale + rand_uniform(leading_shape).unsqueeze(-1) * framewise_focal_drift_pct * alpha
        intrinsics_aug[..., 0, 0] *= fx_scale
        intrinsics_aug[..., 1, 1] *= fy_scale

    elif alpha is not None and framewise_focal_drift_pct > 0:
        fx_scale = 1.0 + rand_uniform(leading_shape).unsqueeze(-1) * framewise_focal_drift_pct * alpha
        fy_scale = 1.0 + rand_uniform(leading_shape).unsqueeze(-1) * framewise_focal_drift_pct * alpha
        intrinsics_aug[..., 0, 0] *= fx_scale
        intrinsics_aug[..., 1, 1] *= fy_scale

    if principal_point_jitter_px > 0:
        cx_offset = (rand_uniform(leading_shape) * principal_point_jitter_px).unsqueeze(-1).expand(*leading_shape, sequence_length)
        cy_offset = (rand_uniform(leading_shape) * principal_point_jitter_px).unsqueeze(-1).expand(*leading_shape, sequence_length)
        if alpha is not None and framewise_principal_point_drift_px > 0:
            cx_offset = cx_offset + rand_uniform(leading_shape).unsqueeze(-1) * framewise_principal_point_drift_px * alpha
            cy_offset = cy_offset + rand_uniform(leading_shape).unsqueeze(-1) * framewise_principal_point_drift_px * alpha
        intrinsics_aug[..., 0, 2] += cx_offset
        intrinsics_aug[..., 1, 2] += cy_offset

    elif alpha is not None and framewise_principal_point_drift_px > 0:
        intrinsics_aug[..., 0, 2] += rand_uniform(leading_shape).unsqueeze(-1) * framewise_principal_point_drift_px * alpha
        intrinsics_aug[..., 1, 2] += rand_uniform(leading_shape).unsqueeze(-1) * framewise_principal_point_drift_px * alpha

    return intrinsics_aug


def _run_forward(
    model: torch.nn.Module,
    images_chunk: torch.Tensor,
    *,
    scale_frames: int,
    kwargs: dict[str, Any],
    dtype: torch.dtype,
    use_grad: bool,
) -> dict[str, torch.Tensor]:
    context = torch.enable_grad if use_grad else torch.no_grad
    autocast_enabled = images_chunk.device.type == "cuda"
    with context():
        with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=dtype):
            return model.forward(
                images_chunk,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
                **kwargs,
            )


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detach a model state dict onto CPU for checkpointing."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    if not args.image_folder and not args.video_path:
        raise SystemExit("Provide --image_folder or --video_path")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder,
        video_path=args.video_path,
        fps=args.fps,
        first_k=args.first_k,
        stride=args.stride,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
    )
    num_frames = int(images.shape[0])
    if num_frames < 2:
        raise SystemExit("Need at least 2 frames for bootstrap training")

    teacher_args = _build_model_args(args, enable_ray_conditioning=False, gate_value=None)
    student_args = _build_model_args(args, enable_ray_conditioning=True, gate_value=args.ray_gate_init)

    teacher = load_model(teacher_args, device)
    student = load_model(student_args, device)
    teacher_dtype = _maybe_cast_aggregator(teacher)
    student_dtype = _maybe_cast_aggregator(student)
    trainable_params = _freeze_non_ray_params(student)
    teacher.eval()
    student.train()

    teacher_storage_device = torch.device("cpu") if args.offload_teacher_to_cpu and device.type == "cuda" else device
    if teacher_storage_device.type == "cpu":
        teacher = teacher.to(teacher_storage_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    teacher_kwargs_all = _build_all_input_kwargs(
        paths,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        camera_model=args.teacher_input_camera_model,
        intrinsics_file=args.teacher_input_intrinsics_file,
    )
    student_kwargs_all = _build_all_input_kwargs(
        paths,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        camera_model=args.student_input_camera_model,
        intrinsics_file=args.student_input_intrinsics_file,
    )
    student_dual_view_images = _maybe_load_dual_view_images(
        paths,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.student_dual_view_preprocess_mode,
    )
    student_dual_view_kwargs_all = None
    if student_dual_view_images is not None:
        student_dual_view_kwargs_all = _build_all_input_kwargs(
            paths,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.student_dual_view_preprocess_mode,
            camera_model=args.student_input_camera_model,
            intrinsics_file=args.student_input_intrinsics_file,
        )

    starts = _build_sequence_starts(num_frames, min(args.sequence_length, num_frames), args.sequence_stride)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    images_cpu = images
    scale_frames = min(args.num_scale_frames, args.sequence_length, num_frames)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_state_dict = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Loaded {num_frames} frames from {resolved_image_folder}")
    print(f"Training on {len(starts)} temporal chunks, sequence_length={min(args.sequence_length, num_frames)}")
    print(f"Teacher camera model: {args.teacher_input_camera_model}")
    print(f"Student camera model: {args.student_input_camera_model}")
    print(f"Ray gate init: {args.ray_gate_init}")
    if student_dual_view_images is not None:
        print(
            f"Student dual view: {args.student_dual_view_preprocess_mode} "
            f"with shape {tuple(student_dual_view_images.shape)}"
        )

    for step in range(1, args.max_steps + 1):
        start_idx = random.choice(starts)
        end_idx = min(start_idx + args.sequence_length, num_frames)
        teacher_chunk = images_cpu[start_idx:end_idx].unsqueeze(0).to(device)
        use_dual_view = (
            student_dual_view_images is not None
            and args.dual_view_distill_prob > 0
            and random.random() < args.dual_view_distill_prob
        )
        student_images_source = student_dual_view_images if use_dual_view else images_cpu
        student_kwargs_source = student_dual_view_kwargs_all if use_dual_view else student_kwargs_all
        student_view_mode = args.student_dual_view_preprocess_mode if use_dual_view else args.preprocess_mode
        student_chunk = student_images_source[start_idx:end_idx].unsqueeze(0).to(device)
        teacher_scale = min(scale_frames, teacher_chunk.shape[1])
        student_scale = min(scale_frames, student_chunk.shape[1])
        schedule_scale = _jitter_schedule_scale(step, args)

        teacher_kwargs = _slice_model_kwargs(teacher, teacher_kwargs_all, start_idx, end_idx)
        student_kwargs = _slice_model_kwargs(student, student_kwargs_source, start_idx, end_idx)
        if "input_intrinsics" in student_kwargs:
            student_kwargs["input_intrinsics"] = _sample_student_intrinsics(
                student_kwargs["input_intrinsics"],
                focal_jitter_pct=args.student_focal_jitter_pct * schedule_scale,
                principal_point_jitter_px=args.student_principal_point_jitter_px * schedule_scale,
                framewise_focal_drift_pct=args.student_framewise_focal_drift_pct * schedule_scale,
                framewise_principal_point_drift_px=args.student_framewise_principal_point_drift_px * schedule_scale,
            )

        teacher.clean_kv_cache()
        student.clean_kv_cache()
        if teacher_storage_device.type == "cpu":
            teacher = teacher.to(device)
        teacher_pred = _run_forward(
            teacher,
            teacher_chunk,
            scale_frames=teacher_scale,
            kwargs=teacher_kwargs,
            dtype=teacher_dtype,
            use_grad=False,
        )
        if teacher_storage_device.type == "cpu":
            teacher = teacher.to(teacher_storage_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        student_pred = _run_forward(
            student,
            student_chunk,
            scale_frames=student_scale,
            kwargs=student_kwargs,
            dtype=student_dtype,
            use_grad=True,
        )

        total_loss, scalar_losses = _compute_distill_loss(student_pred, teacher_pred, args)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
        optimizer.step()

        gate_mean = float(student.aggregator.ray_conditioning_gate.detach().mean().item())
        log_entry = {
            "step": step,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "gate_mean": gate_mean,
            "dual_view_used": float(use_dual_view),
            "student_view_mode": student_view_mode,
            "jitter_schedule_scale": schedule_scale,
            **scalar_losses,
        }
        history.append(log_entry)

        if scalar_losses["total"] < best_loss:
            best_loss = scalar_losses["total"]
            best_state_dict = _cpu_state_dict(student)

        if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
            print(
                f"step={step:04d} loss={scalar_losses['total']:.6f} "
                f"pose={scalar_losses['pose_xyz_quat']:.6f} depth={scalar_losses['depth']:.6f} "
                f"conf={scalar_losses['depth_conf']:.6f} gate_mean={gate_mean:.6f}"
            )

        teacher.clean_kv_cache()
        student.clean_kv_cache()
        del teacher_pred, student_pred, teacher_chunk, student_chunk, total_loss

    checkpoint_path = output_dir / "ray_conditioning_bootstrap_best.pt"
    if best_state_dict is None:
        raise RuntimeError("No checkpoint state captured during training")
    torch.save({"model": best_state_dict, "config": vars(args), "best_loss": best_loss}, checkpoint_path)

    summary = {
        "input": {
            "resolved_image_folder": resolved_image_folder,
            "num_frames": num_frames,
            "num_chunks": len(starts),
        },
        "settings": {
            **vars(args),
            "teacher_dtype": str(teacher_dtype),
            "student_dtype": str(student_dtype),
        },
        "results": {
            "best_loss": best_loss,
            "final_loss": history[-1]["total"] if history else None,
            "final_gate_mean": history[-1]["gate_mean"] if history else None,
            "peak_gpu_allocated_gb": (
                float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None
            ),
            "elapsed_sec": time.time() - t0,
            "checkpoint_path": str(checkpoint_path),
        },
        "history": history,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
