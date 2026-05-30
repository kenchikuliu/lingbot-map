#!/usr/bin/env python3
"""Aggregate headless evaluation JSON reports into tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SELECTED_METRICS = [
    "step_rotation_deg_mean",
    "step_rotation_deg_max",
    "step_translation_mean",
    "step_translation_max",
    "step_translation_std",
    "depth_conf_frame_mean_std",
    "depth_conf_mean",
    "translation_norm_mean",
]

TIMING_METRICS = [
    "frames_per_sec_inference",
    "inference_time_sec",
    "load_time_sec",
    "postprocess_time_sec",
]

GROUP_ORDER = [
    "temporal_subsampling",
    "projection_shift",
    "viewpoint_break",
    "preprocess_ablation",
    "other",
]

GROUP_TITLES = {
    "temporal_subsampling": "Temporal Sparsity",
    "projection_shift": "Projection Shift",
    "viewpoint_break": "Viewpoint Break",
    "preprocess_ablation": "Preprocess Ablation",
    "other": "Other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate research_eval JSON reports")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research_eval"),
        help="Root directory containing JSON evaluation reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_eval/aggregated"),
        help="Directory for aggregated CSV/Markdown/plot outputs",
    )
    return parser.parse_args()


def infer_group(path: Path) -> str:
    parts = set(path.parts)
    if "temporal_subsampling" in parts:
        return "temporal_subsampling"
    if "projection_shift" in parts:
        return "projection_shift"
    if "viewpoint_break" in parts:
        return "viewpoint_break"
    if "preprocess_ablation" in parts:
        return "preprocess_ablation"
    return "other"


def infer_sort_key(label: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if label.startswith("temporal_fps_"):
        try:
            fps = int(label.split("_")[-1])
            return (0, fps)
        except ValueError:
            pass
    if label.startswith("fps_"):
        try:
            fps = int(label.split("_")[-1])
            return (0, fps)
        except ValueError:
            pass
    return (1, label)


def read_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proxy = data.get("proxy_metrics", {})
    timing = data.get("timing", {})
    gpu = data.get("gpu", {})
    input_info = data.get("input", {})
    settings = data.get("settings", {})
    group = infer_group(path)
    label = data.get("tag") or path.stem
    if group == "preprocess_ablation":
        label = path.parent.name

    row: dict[str, Any] = {
        "group": group,
        "path": str(path),
        "label": label,
        "num_frames_after_sampling": input_info.get("num_frames_after_sampling"),
        "image_folder": input_info.get("image_folder"),
        "mode": settings.get("mode"),
        "camera_num_iterations": settings.get("camera_num_iterations"),
        "window_size": settings.get("window_size"),
        "overlap_size": settings.get("overlap_size"),
        "max_memory_allocated_gb": gpu.get("max_memory_allocated_gb"),
        "max_memory_reserved_gb": gpu.get("max_memory_reserved_gb"),
    }
    for key in SELECTED_METRICS:
        row[key] = proxy.get(key)
    for key in TIMING_METRICS:
        row[key] = timing.get(key)
    return row


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Aggregated Research Eval Summary")
    lines.append("")
    lines.append("This file is generated from JSON reports under `research_eval/`.")
    lines.append("")

    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUP_ORDER}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    for group in GROUP_ORDER:
        items = grouped.get(group, [])
        if not items:
            continue
        items = sorted(items, key=lambda row: infer_sort_key(row["label"], row))
        lines.append(f"## {GROUP_TITLES.get(group, group)}")
        lines.append("")
        headers = [
            "label",
            "num_frames_after_sampling",
            "step_rotation_deg_mean",
            "step_translation_mean",
            "step_translation_std",
            "depth_conf_frame_mean_std",
            "frames_per_sec_inference",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in items:
            lines.append(
                "| " + " | ".join(format_value(row.get(header)) for header in headers) + " |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_group(rows: list[dict[str, Any]], group: str, out_dir: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: infer_sort_key(row["label"], row))
    labels = [row["label"] for row in rows]

    metrics = [
        ("step_rotation_deg_mean", "Mean Step Rotation (deg)"),
        ("step_translation_mean", "Mean Step Translation"),
        ("step_translation_std", "Step Translation Std"),
        ("depth_conf_frame_mean_std", "Depth-Conf Frame Std"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    x = list(range(len(labels)))

    for ax, (key, title) in zip(axes, metrics):
        values = [row.get(key) for row in rows]
        ax.bar(x, values, color="#4C78A8")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(GROUP_TITLES.get(group, group))
    fig.tight_layout()
    fig.savefig(out_dir / f"{group}_bars.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_curve(rows: list[dict[str, Any]], out_dir: Path) -> None:
    temporal_rows = [row for row in rows if row["group"] == "temporal_subsampling"]
    parsed = []
    for row in temporal_rows:
        label = str(row["label"])
        fps = None
        if label.startswith("temporal_fps_"):
            try:
                fps = int(label.split("_")[-1])
            except ValueError:
                pass
        elif label.startswith("fps_"):
            try:
                fps = int(label.split("_")[-1])
            except ValueError:
                pass
        if fps is not None:
            parsed.append((fps, row))

    if len(parsed) < 2:
        return

    parsed.sort(key=lambda item: item[0], reverse=True)
    fps_values = [item[0] for item in parsed]
    rot_values = [item[1].get("step_rotation_deg_mean") for item in parsed]
    trans_values = [item[1].get("step_translation_mean") for item in parsed]
    conf_values = [item[1].get("depth_conf_frame_mean_std") for item in parsed]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(fps_values, rot_values, marker="o", color="#F58518")
    axes[0].set_title("Mean Step Rotation")
    axes[1].plot(fps_values, trans_values, marker="o", color="#54A24B")
    axes[1].set_title("Mean Step Translation")
    axes[2].plot(fps_values, conf_values, marker="o", color="#E45756")
    axes[2].set_title("Depth-Conf Frame Std")

    for ax in axes:
        ax.set_xlabel("FPS")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()

    fig.suptitle("Temporal Sparsity Degradation")
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_degradation_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(root.rglob("*.json")):
        rows.append(read_report(path))

    rows.sort(key=lambda row: (GROUP_ORDER.index(row["group"]) if row["group"] in GROUP_ORDER else 999, row["label"]))

    write_csv(rows, output_dir / "all_reports.csv")
    write_markdown(rows, output_dir / "summary.md")

    for group in GROUP_ORDER:
        group_rows = [row for row in rows if row["group"] == group]
        if group_rows:
            plot_group(group_rows, group, output_dir)
    plot_temporal_curve(rows, output_dir)

    print(f"Wrote {len(rows)} reports to {output_dir}")
    print(f"CSV: {output_dir / 'all_reports.csv'}")
    print(f"Markdown: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
