#!/usr/bin/env python3
"""Headless LingBot-MAP sequence evaluator.

Runs the same model pipeline as demo.py but skips visualization and writes a
compact JSON report with proxy robustness metrics. This is intended for
baseline stress testing on local data that may not have ground-truth poses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

legacy_alloc_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("PYTORCH_ALLOC_CONF", legacy_alloc_conf or "expandable_segments:True")

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo import load_images, load_model, postprocess
from lingbot_map.utils import load_input_intrinsics


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_mean(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _safe_std(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    return float(np.std(arr))


def _rotation_angle_deg(R_rel: np.ndarray) -> float:
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_proxy_metrics(predictions: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    extrinsic = predictions.get("extrinsic")
    if extrinsic is not None:
        extrinsic = np.asarray(extrinsic)
        trans = extrinsic[:, :3, 3]
        metrics["translation_norm_mean"] = _safe_mean(np.linalg.norm(trans, axis=1))
        metrics["translation_norm_std"] = _safe_std(np.linalg.norm(trans, axis=1))

        if extrinsic.shape[0] >= 2:
            step_trans = np.linalg.norm(np.diff(trans, axis=0), axis=1)
            step_rot = []
            for i in range(extrinsic.shape[0] - 1):
                R_prev = extrinsic[i, :3, :3]
                R_next = extrinsic[i + 1, :3, :3]
                step_rot.append(_rotation_angle_deg(R_next @ R_prev.T))
            step_rot = np.asarray(step_rot, dtype=np.float32)
            metrics["step_translation_mean"] = _safe_mean(step_trans)
            metrics["step_translation_std"] = _safe_std(step_trans)
            metrics["step_translation_max"] = _float_or_none(step_trans.max() if step_trans.size else None)
            metrics["step_rotation_deg_mean"] = _safe_mean(step_rot)
            metrics["step_rotation_deg_std"] = _safe_std(step_rot)
            metrics["step_rotation_deg_max"] = _float_or_none(step_rot.max() if step_rot.size else None)

    depth = predictions.get("depth")
    if depth is not None:
        depth = np.asarray(depth)[..., 0]
        valid = depth > 1e-8
        metrics["valid_depth_ratio_mean"] = _safe_mean(valid.mean(axis=(1, 2)))
        metrics["valid_depth_ratio_std"] = _safe_std(valid.mean(axis=(1, 2)))
        if valid.any():
            valid_depth = depth[valid]
            metrics["depth_mean"] = _safe_mean(valid_depth)
            metrics["depth_std"] = _safe_std(valid_depth)
            metrics["depth_p95"] = _float_or_none(np.percentile(valid_depth, 95))

    depth_conf = predictions.get("depth_conf")
    if depth_conf is not None:
        depth_conf = np.asarray(depth_conf)
        metrics["depth_conf_mean"] = _safe_mean(depth_conf)
        metrics["depth_conf_std"] = _safe_std(depth_conf)
        per_frame = depth_conf.mean(axis=(1, 2))
        metrics["depth_conf_frame_mean_std"] = _safe_std(per_frame)

    world_points_conf = predictions.get("world_points_conf")
    if world_points_conf is not None:
        world_points_conf = np.asarray(world_points_conf)
        metrics["world_points_conf_mean"] = _safe_mean(world_points_conf)
        metrics["world_points_conf_std"] = _safe_std(world_points_conf)

    frame_type = predictions.get("frame_type")
    if frame_type is not None:
        frame_type = np.asarray(frame_type)
        metrics["num_scale_frames_observed"] = int((frame_type == 0).sum())
        metrics["num_keyframes_observed"] = int((frame_type == 1).sum())
        metrics["num_non_keyframes_observed"] = int((frame_type == 2).sum())

    is_keyframe = predictions.get("is_keyframe")
    if is_keyframe is not None:
        is_keyframe = np.asarray(is_keyframe).astype(bool)
        metrics["keyframe_ratio"] = _safe_mean(is_keyframe.astype(np.float32))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless evaluator for LingBot-MAP sequences")

    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="Image preprocessing mode before model input.",
    )

    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"])
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=None)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument("--enable_ray_conditioning", action="store_true", default=False)
    parser.add_argument(
        "--input_camera_model",
        type=str,
        default="pinhole",
        choices=["pinhole", "perspective", "equirectangular"],
    )
    parser.add_argument("--input_intrinsics_file", type=str, default=None)
    parser.add_argument(
        "--ray_use_default_intrinsics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ray_conditioning_gate_value", type=float, default=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--overlap_size", type=int, default=16)

    parser.add_argument("--json_out", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_folder and not args.video_path:
        raise SystemExit("Provide --image_folder or --video_path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_load0 = time.time()
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
    model = load_model(args, device)
    load_time_sec = time.time() - t_load0

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = int(images.shape[0])

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = math.ceil(num_frames / 320)
        else:
            args.keyframe_interval = 1

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    forward_kwargs = {
        "input_camera_model": args.input_camera_model,
    }
    if args.input_intrinsics_file:
        forward_kwargs["input_intrinsics"] = load_input_intrinsics(
            paths,
            args.input_intrinsics_file,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.preprocess_mode,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_inf0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
                **forward_kwargs,
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                output_device=output_device,
                **forward_kwargs,
            )
    inference_time_sec = time.time() - t_inf0

    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]
    else:
        images_for_post = images

    t_post0 = time.time()
    predictions, images_cpu = postprocess(predictions, images_for_post)
    postprocess_time_sec = time.time() - t_post0

    proxy_metrics = compute_proxy_metrics(predictions)

    report = {
        "tag": args.tag,
        "input": {
            "image_folder": args.image_folder,
            "video_path": args.video_path,
            "resolved_image_folder": resolved_image_folder,
            "num_input_paths": len(paths),
            "num_frames_after_sampling": num_frames,
            "fps": args.fps,
            "first_k": args.first_k,
            "stride": args.stride,
        },
        "settings": {
            "mode": args.mode,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "preprocess_mode": args.preprocess_mode,
            "num_scale_frames": args.num_scale_frames,
            "keyframe_interval": args.keyframe_interval,
            "window_size": args.window_size,
            "overlap_size": args.overlap_size,
            "camera_num_iterations": args.camera_num_iterations,
            "use_sdpa": bool(args.use_sdpa),
            "offload_to_cpu": bool(args.offload_to_cpu),
            "enable_ray_conditioning": bool(args.enable_ray_conditioning),
            "input_camera_model": args.input_camera_model,
            "input_intrinsics_file": args.input_intrinsics_file,
            "ray_use_default_intrinsics": bool(args.ray_use_default_intrinsics),
            "ray_conditioning_gate_value": args.ray_conditioning_gate_value,
            "dtype": str(dtype),
        },
        "timing": {
            "load_time_sec": load_time_sec,
            "inference_time_sec": inference_time_sec,
            "postprocess_time_sec": postprocess_time_sec,
            "frames_per_sec_inference": float(num_frames / inference_time_sec) if inference_time_sec > 0 else None,
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "max_memory_allocated_gb": (
                float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None
            ),
            "max_memory_reserved_gb": (
                float(torch.cuda.max_memory_reserved() / 1e9) if torch.cuda.is_available() else None
            ),
        },
        "prediction_keys": sorted(predictions.keys()),
        "proxy_metrics": proxy_metrics,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))

    del images_cpu


if __name__ == "__main__":
    main()
