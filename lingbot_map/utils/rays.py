"""Camera-ray utilities for projection-aware geometry experiments.

These helpers are intentionally standalone: they do not alter the current
LingBot-MAP inference path, but they provide the geometry interface needed by a
future camera-agnostic streaming model.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def make_pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    pixel_center: float = 0.5,
) -> torch.Tensor:
    """Return an ``HxWx2`` pixel-center grid with ``(x, y)`` coordinates."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + pixel_center,
        torch.arange(width, device=device, dtype=dtype) + pixel_center,
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1)


def compute_preprocess_geometry(
    orig_height: int,
    orig_width: int,
    *,
    image_size: int,
    patch_size: int,
    mode: str = "crop",
) -> dict[str, int | float]:
    """Mirror the resize/crop/pad logic in ``load_fn.py``.

    Returns a small metadata dictionary describing how the original image is
    transformed before it reaches the model.
    """
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be 'crop' or 'pad'")

    if mode == "pad":
        if orig_width >= orig_height:
            resized_width = image_size
            resized_height = round(orig_height * (resized_width / orig_width) / patch_size) * patch_size
        else:
            resized_height = image_size
            resized_width = round(orig_width * (resized_height / orig_height) / patch_size) * patch_size
    else:
        resized_width = image_size
        resized_height = round(orig_height * (resized_width / orig_width) / patch_size) * patch_size

    crop_top = 0
    crop_left = 0
    output_height = resized_height
    output_width = resized_width
    pad_top = pad_bottom = pad_left = pad_right = 0

    if mode == "crop" and resized_height > image_size:
        crop_top = (resized_height - image_size) // 2
        output_height = image_size

    if mode == "pad":
        pad_h = image_size - resized_height
        pad_w = image_size - resized_width
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        output_height = image_size
        output_width = image_size

    return {
        "orig_height": orig_height,
        "orig_width": orig_width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "output_height": output_height,
        "output_width": output_width,
        "scale_x": resized_width / orig_width,
        "scale_y": resized_height / orig_height,
        "crop_top": crop_top,
        "crop_left": crop_left,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }


def transform_pinhole_intrinsics(
    intrinsics: Any,
    *,
    orig_height: int,
    orig_width: int,
    image_size: int,
    patch_size: int,
    mode: str = "crop",
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Transform pinhole intrinsics through the current image preprocessing."""
    intrinsics_t = torch.as_tensor(intrinsics)
    if intrinsics_t.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must end with shape (3, 3), got {tuple(intrinsics_t.shape)}")

    meta = compute_preprocess_geometry(
        orig_height,
        orig_width,
        image_size=image_size,
        patch_size=patch_size,
        mode=mode,
    )

    dtype = intrinsics_t.dtype if intrinsics_t.is_floating_point() else torch.float32
    intrinsics_t = intrinsics_t.to(dtype=dtype).clone()

    intrinsics_t[..., 0, 0] *= float(meta["scale_x"])
    intrinsics_t[..., 1, 1] *= float(meta["scale_y"])
    intrinsics_t[..., 0, 2] *= float(meta["scale_x"])
    intrinsics_t[..., 1, 2] *= float(meta["scale_y"])

    intrinsics_t[..., 0, 2] -= float(meta["crop_left"])
    intrinsics_t[..., 1, 2] -= float(meta["crop_top"])
    intrinsics_t[..., 0, 2] += float(meta["pad_left"])
    intrinsics_t[..., 1, 2] += float(meta["pad_top"])

    return intrinsics_t, meta


def pinhole_rays(
    height: int,
    width: int,
    intrinsics: Any,
    *,
    normalize: bool = True,
    pixel_center: float = 0.5,
) -> torch.Tensor:
    """Build pinhole camera rays in OpenCV coordinates.

    Output shape is ``(..., H, W, 3)`` where the leading dimensions follow the
    input intrinsics shape.
    """
    intrinsics_t = torch.as_tensor(intrinsics)
    if intrinsics_t.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must end with shape (3, 3), got {tuple(intrinsics_t.shape)}")

    dtype = intrinsics_t.dtype if intrinsics_t.is_floating_point() else torch.float32
    intrinsics_t = intrinsics_t.to(dtype=dtype)
    grid = make_pixel_grid(height, width, device=intrinsics_t.device, dtype=dtype, pixel_center=pixel_center)

    fx = intrinsics_t[..., 0, 0][..., None, None]
    fy = intrinsics_t[..., 1, 1][..., None, None]
    cx = intrinsics_t[..., 0, 2][..., None, None]
    cy = intrinsics_t[..., 1, 2][..., None, None]

    x = (grid[..., 0] - cx) / fx
    y = (grid[..., 1] - cy) / fy
    z = torch.ones_like(x)
    rays = torch.stack([x, y, z], dim=-1)

    if normalize:
        rays = torch.nn.functional.normalize(rays, dim=-1)
    return rays


def equirectangular_rays(
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
    pixel_center: float = 0.5,
) -> torch.Tensor:
    """Build unit rays for an equirectangular panorama in OpenCV coordinates.

    The convention is:

    - ``x`` points right
    - ``y`` points down
    - ``z`` points forward

    The center pixel looks forward, matching the common OpenCV camera frame.
    """
    grid = make_pixel_grid(height, width, device=device, dtype=dtype, pixel_center=pixel_center)

    azimuth = (grid[..., 0] / width) * (2.0 * math.pi) - math.pi
    elevation = 0.5 * math.pi - (grid[..., 1] / height) * math.pi

    cos_el = torch.cos(elevation)
    x = torch.sin(azimuth) * cos_el
    y = -torch.sin(elevation)
    z = torch.cos(azimuth) * cos_el
    rays = torch.stack([x, y, z], dim=-1)

    if normalize:
        rays = torch.nn.functional.normalize(rays, dim=-1)
    return rays


def patchify_rays(rays: Any, patch_size: int, *, normalize: bool = True) -> torch.Tensor:
    """Average ray directions within non-overlapping patch cells."""
    rays_t = torch.as_tensor(rays)
    if rays_t.shape[-1] != 3:
        raise ValueError(f"rays must end with shape (..., 3), got {tuple(rays_t.shape)}")

    height, width = rays_t.shape[-3], rays_t.shape[-2]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"ray grid spatial shape {(height, width)} is not divisible by patch_size={patch_size}"
        )

    batch_shape = rays_t.shape[:-3]
    rays_flat = rays_t.reshape(-1, height, width, 3)
    rays_flat = rays_flat.reshape(
        -1,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
        3,
    )
    patch_rays = rays_flat.mean(dim=(2, 4))
    patch_rays = patch_rays.reshape(*batch_shape, height // patch_size, width // patch_size, 3)

    if normalize:
        patch_rays = torch.nn.functional.normalize(patch_rays, dim=-1)
    return patch_rays


def build_camera_rays(
    camera_model: str,
    *,
    height: int,
    width: int,
    intrinsics: Any | None = None,
    normalize: bool = True,
    pixel_center: float = 0.5,
) -> torch.Tensor:
    """Dispatch helper for common camera models."""
    model = camera_model.lower()
    if model in {"pinhole", "perspective"}:
        if intrinsics is None:
            raise ValueError("intrinsics are required for pinhole ray generation")
        return pinhole_rays(
            height,
            width,
            intrinsics,
            normalize=normalize,
            pixel_center=pixel_center,
        )
    if model in {"equirectangular", "erp", "panorama"}:
        return equirectangular_rays(
            height,
            width,
            device=torch.as_tensor(intrinsics).device if intrinsics is not None else None,
            dtype=(torch.as_tensor(intrinsics).dtype if intrinsics is not None and torch.as_tensor(intrinsics).is_floating_point() else torch.float32),
            normalize=normalize,
            pixel_center=pixel_center,
        )
    raise ValueError(f"Unsupported camera_model: {camera_model}")
