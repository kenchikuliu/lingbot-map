#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_viewpoint_break.sh --image-folder PATH [options]

Options:
  --image-folder PATH           Source image folder (required)
  --model-path PATH             Checkpoint path
  --python PATH                 Python executable
  --break-start N               Start index of dropped segment (default: 0, disabled if break-len=0)
  --break-len N                 Number of frames to drop from the middle (default: 0)
  --keep-prefix N               Keep only first N frames before appending suffix (optional)
  --mode MODE                   streaming or windowed (default: streaming)
  --window-size N               Window size for windowed mode (default: 64)
  --num-scale-frames N          Passed to evaluator (default: 2)
  --camera-num-iterations N     Passed to evaluator (default: 1)
  --preprocess-mode MODE        crop or pad (default: crop)
  --output-dir PATH             Output directory
  --tag NAME                    Tag for the generated report (default: viewpoint_break)
  -h, --help                    Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
MODEL_PATH="${ROOT_DIR}/checkpoints/lingbot-map.pt"
IMAGE_FOLDER=""
BREAK_START=0
BREAK_LEN=0
KEEP_PREFIX=""
MODE="streaming"
WINDOW_SIZE=64
NUM_SCALE_FRAMES=2
CAMERA_NUM_ITERATIONS=1
PREPROCESS_MODE="crop"
OUTPUT_DIR="${ROOT_DIR}/research_eval/viewpoint_break/default"
TAG="viewpoint_break"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-folder) IMAGE_FOLDER="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --break-start) BREAK_START="$2"; shift 2 ;;
    --break-len) BREAK_LEN="$2"; shift 2 ;;
    --keep-prefix) KEEP_PREFIX="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --num-scale-frames) NUM_SCALE_FRAMES="$2"; shift 2 ;;
    --camera-num-iterations) CAMERA_NUM_ITERATIONS="$2"; shift 2 ;;
    --preprocess-mode) PREPROCESS_MODE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${IMAGE_FOLDER}" ]]; then
  echo "--image-folder is required" >&2
  exit 1
fi

if [[ ! -d "${IMAGE_FOLDER}" ]]; then
  echo "Image folder not found: ${IMAGE_FOLDER}" >&2
  exit 1
fi

prepared_dir="${OUTPUT_DIR}/prepared_${TAG}"
json_out="${OUTPUT_DIR}/${TAG}.json"
mkdir -p "${prepared_dir}" "${OUTPUT_DIR}"
rm -rf "${prepared_dir}"
mkdir -p "${prepared_dir}"

mapfile -t all_images < <(find "${IMAGE_FOLDER}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort)
if [[ "${#all_images[@]}" -eq 0 ]]; then
  echo "No images found in ${IMAGE_FOLDER}" >&2
  exit 1
fi

out_idx=0
for i in "${!all_images[@]}"; do
  if [[ -n "${KEEP_PREFIX}" && "${i}" -ge "${KEEP_PREFIX}" ]]; then
    break
  fi
  if [[ "${BREAK_LEN}" -gt 0 && "${i}" -ge "${BREAK_START}" && "${i}" -lt $((BREAK_START + BREAK_LEN)) ]]; then
    continue
  fi
  ext="${all_images[$i]##*.}"
  ln -sfn "${all_images[$i]}" "$(printf '%s/%06d.%s' "${prepared_dir}" "${out_idx}" "${ext}")"
  out_idx=$((out_idx + 1))
done

if [[ "${out_idx}" -eq 0 ]]; then
  echo "Prepared sequence is empty" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_sequence.py"
  --model_path "${MODEL_PATH}"
  --image_folder "${prepared_dir}"
  --mode "${MODE}"
  --use_sdpa
  --num_scale_frames "${NUM_SCALE_FRAMES}"
  --camera_num_iterations "${CAMERA_NUM_ITERATIONS}"
  --preprocess_mode "${PREPROCESS_MODE}"
  --json_out "${json_out}"
  --tag "${TAG}"
)
if [[ "${MODE}" == "windowed" ]]; then
  cmd+=(--window_size "${WINDOW_SIZE}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "Wrote viewpoint-break report to ${json_out}"
