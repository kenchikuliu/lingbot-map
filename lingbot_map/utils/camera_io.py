"""Utilities for loading per-frame camera metadata for experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .rays import transform_pinhole_intrinsics

_INTRINSICS_KEYS = ("intrinsics", "K", "matrix")
_NAME_KEYS = ("image_path", "path", "file_name", "filename", "name", "frame_name", "frame")


def _load_serialized_camera_spec(path: str | os.PathLike[str]) -> Any:
    """Load a camera-spec object from ``.json``, ``.npy``, ``.npz``, or ``.pt``."""
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
            return data.item()
        return data

    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)

    raise ValueError(f"Unsupported intrinsics file type: {path}")


def _coerce_intrinsics_tensor(value: Any) -> torch.Tensor:
    """Convert a raw intrinsics payload to a float tensor."""
    tensor = torch.as_tensor(value)
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must end with shape (3, 3), got {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


def _path_tokens(path_like: Any) -> list[str]:
    """Return progressively weaker lookup tokens for an image path."""
    raw = str(path_like).replace("\\", "/")
    path = Path(raw)
    tokens = []
    for token in (raw, path.name, path.stem):
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _resolve_intrinsics_from_named_entries(entries: list[dict[str, Any]], image_paths: list[str]) -> torch.Tensor:
    """Resolve per-frame intrinsics from a list of named entries."""
    token_to_idx: dict[str, int] = {}
    matrices: list[torch.Tensor] = []

    for idx, entry in enumerate(entries):
        intrinsics_value = next((entry[key] for key in _INTRINSICS_KEYS if key in entry), None)
        if intrinsics_value is None:
            raise ValueError(f"Entry {idx} is missing an intrinsics matrix")
        matrices.append(_coerce_intrinsics_tensor(intrinsics_value))

        name_value = next((entry[key] for key in _NAME_KEYS if key in entry), None)
        if name_value is None:
            continue
        for token in _path_tokens(name_value):
            token_to_idx[token] = idx

    if not token_to_idx:
        if len(entries) != len(image_paths):
            raise ValueError(
                "Intrinsics entry list has no per-frame names, so its length must match the input frame count"
            )
        return torch.stack(matrices, dim=0)

    ordered = []
    for path in image_paths:
        matched_idx = next((token_to_idx[token] for token in _path_tokens(path) if token in token_to_idx), None)
        if matched_idx is None:
            raise ValueError(f"Could not find intrinsics for frame: {path}")
        ordered.append(matrices[matched_idx])
    return torch.stack(ordered, dim=0)


def _resolve_intrinsics_sequence(spec: Any, image_paths: list[str]) -> torch.Tensor:
    """Normalize a camera-spec object to ``[S, 3, 3]`` intrinsics."""
    num_frames = len(image_paths)

    if isinstance(spec, dict):
        intrinsics_value = next((spec[key] for key in _INTRINSICS_KEYS if key in spec), None)
        if intrinsics_value is not None:
            tensor = _coerce_intrinsics_tensor(intrinsics_value)
            name_list = next(
                (spec[key] for key in ("image_paths", "paths", "frame_names", "names") if key in spec),
                None,
            )
            if name_list is None:
                if tensor.ndim == 2:
                    return tensor.unsqueeze(0).expand(num_frames, -1, -1).clone()
                if tensor.ndim == 3 and tensor.shape[0] == num_frames:
                    return tensor
                raise ValueError(
                    "Intrinsics tensor must have shape (3, 3) or (S, 3, 3) when no path list is provided"
                )

            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).expand(len(name_list), -1, -1).clone()
            if tensor.ndim != 3 or tensor.shape[0] != len(name_list):
                raise ValueError("Path list length must match the number of intrinsics matrices")

            entries = [
                {"name": name, "intrinsics": tensor[idx]}
                for idx, name in enumerate(name_list)
            ]
            return _resolve_intrinsics_from_named_entries(entries, image_paths)

        if all(isinstance(value, dict) for value in spec.values()):
            entries = []
            for name, entry in spec.items():
                if not any(key in entry for key in _NAME_KEYS):
                    entry = dict(entry)
                    entry["name"] = name
                entries.append(entry)
            return _resolve_intrinsics_from_named_entries(entries, image_paths)

        entries = [{"name": name, "intrinsics": value} for name, value in spec.items()]
        return _resolve_intrinsics_from_named_entries(entries, image_paths)

    if isinstance(spec, (list, tuple)):
        if not spec:
            raise ValueError("Intrinsics list is empty")
        if all(isinstance(item, dict) for item in spec):
            return _resolve_intrinsics_from_named_entries(list(spec), image_paths)

        tensor = _coerce_intrinsics_tensor(spec)
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).expand(num_frames, -1, -1).clone()
        if tensor.ndim == 3 and tensor.shape[0] == num_frames:
            return tensor
        raise ValueError("Intrinsics list must have shape (3, 3) or (S, 3, 3)")

    tensor = _coerce_intrinsics_tensor(spec)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).expand(num_frames, -1, -1).clone()
    if tensor.ndim == 3 and tensor.shape[0] == num_frames:
        return tensor
    raise ValueError("Intrinsics array must have shape (3, 3) or (S, 3, 3)")


def _read_image_size(path: str) -> tuple[int, int]:
    """Return ``(height, width)`` for a source image without decoding all pixels."""
    with Image.open(path) as img:
        width, height = img.size
    return height, width


def load_input_intrinsics(
    image_paths: list[str],
    intrinsics_file: str | os.PathLike[str],
    *,
    image_size: int,
    patch_size: int,
    preprocess_mode: str,
) -> torch.Tensor:
    """Load and transform intrinsics into model-input pixel coordinates.

    The file is interpreted in the original image coordinate system, then each
    matrix is transformed through the repo's resize/crop/pad preprocessing.
    """
    if len(image_paths) == 0:
        raise ValueError("At least one image path is required to load intrinsics")

    raw_spec = _load_serialized_camera_spec(intrinsics_file)
    intrinsics = _resolve_intrinsics_sequence(raw_spec, image_paths)

    transformed = []
    for idx, path in enumerate(image_paths):
        orig_height, orig_width = _read_image_size(path)
        intrinsics_i, _ = transform_pinhole_intrinsics(
            intrinsics[idx],
            orig_height=orig_height,
            orig_width=orig_width,
            image_size=image_size,
            patch_size=patch_size,
            mode=preprocess_mode,
        )
        transformed.append(intrinsics_i)

    return torch.stack(transformed, dim=0)
