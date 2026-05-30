#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_projection_shift.sh [options]

Options:
  --model-path PATH             Checkpoint path
  --python PATH                 Python executable
  --perspective-dir PATH        Perspective image folder to evaluate
  --widefov-dir PATH            Wide-FoV / 360-derived image folder to evaluate
  --mode MODE                   streaming or windowed (default: windowed)
  --window-size N               Window size for windowed mode (default: 16)
  --overlap-size N              Overlap size for windowed mode (default: 4)
  --num-scale-frames N          Passed to evaluator (default: 2)
  --camera-num-iterations N     Passed to evaluator (default: 1)
  --first-k N                   Limit frames passed to evaluator
  --preprocess-mode MODE        crop or pad (default: crop)
  --output-dir PATH             Output directory (default: research_eval/projection_shift/default)
  -h, --help                    Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
MODEL_PATH="${ROOT_DIR}/checkpoints/lingbot-map.pt"
PERSPECTIVE_DIR=""
WIDEFOV_DIR=""
MODE="windowed"
WINDOW_SIZE=16
OVERLAP_SIZE=4
NUM_SCALE_FRAMES=2
CAMERA_NUM_ITERATIONS=1
FIRST_K=""
PREPROCESS_MODE="crop"
OUTPUT_DIR="${ROOT_DIR}/research_eval/projection_shift/default"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --perspective-dir) PERSPECTIVE_DIR="$2"; shift 2 ;;
    --widefov-dir) WIDEFOV_DIR="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --overlap-size) OVERLAP_SIZE="$2"; shift 2 ;;
    --num-scale-frames) NUM_SCALE_FRAMES="$2"; shift 2 ;;
    --camera-num-iterations) CAMERA_NUM_ITERATIONS="$2"; shift 2 ;;
    --first-k) FIRST_K="$2"; shift 2 ;;
    --preprocess-mode) PREPROCESS_MODE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

run_eval() {
  local tag="$1"
  local image_dir="$2"
  local json_out="${OUTPUT_DIR}/${tag}.json"
  local -a cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_sequence.py"
    --model_path "${MODEL_PATH}"
    --image_folder "${image_dir}"
    --mode "${MODE}"
    --use_sdpa
    --num_scale_frames "${NUM_SCALE_FRAMES}"
    --camera_num_iterations "${CAMERA_NUM_ITERATIONS}"
    --preprocess_mode "${PREPROCESS_MODE}"
    --window_size "${WINDOW_SIZE}"
    --overlap_size "${OVERLAP_SIZE}"
    --json_out "${json_out}"
    --tag "${tag}"
  )
  if [[ -n "${FIRST_K}" ]]; then
    cmd+=(--first_k "${FIRST_K}")
  fi
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

if [[ -n "${PERSPECTIVE_DIR}" ]]; then
  run_eval "perspective" "${PERSPECTIVE_DIR}"
fi

if [[ -n "${WIDEFOV_DIR}" ]]; then
  run_eval "widefov" "${WIDEFOV_DIR}"
fi

echo "Wrote projection shift reports to ${OUTPUT_DIR}"
