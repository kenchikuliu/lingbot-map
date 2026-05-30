#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/eval_temporal_subsampling.sh --video /path/to/video.mp4 [options]

Options:
  --video PATH                  Input video path (required)
  --model-path PATH             Checkpoint path
  --python PATH                 Python executable
  --fps-list LIST               Comma-separated FPS list (default: 10,5,2,1)
  --mode MODE                   streaming or windowed (default: streaming)
  --window-size N               Window size for windowed mode (default: 128)
  --num-scale-frames N          Passed to evaluator (default: 2)
  --camera-num-iterations N     Passed to evaluator (default: 1)
  --output-dir PATH             Output directory (default: research_eval/temporal_subsampling/<video-stem>)
  --first-k N                   Limit frames passed to evaluator
  --preprocess-mode MODE        crop or pad (default: crop)
  --reextract                   Force frame re-extraction for each fps
  -h, --help                    Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
MODEL_PATH="${ROOT_DIR}/checkpoints/lingbot-map.pt"
FPS_LIST="10,5,2,1"
MODE="streaming"
WINDOW_SIZE=128
NUM_SCALE_FRAMES=2
CAMERA_NUM_ITERATIONS=1
OUTPUT_DIR=""
VIDEO_PATH=""
FIRST_K=""
PREPROCESS_MODE="crop"
REEXTRACT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO_PATH="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --fps-list) FPS_LIST="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --num-scale-frames) NUM_SCALE_FRAMES="$2"; shift 2 ;;
    --camera-num-iterations) CAMERA_NUM_ITERATIONS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --first-k) FIRST_K="$2"; shift 2 ;;
    --preprocess-mode) PREPROCESS_MODE="$2"; shift 2 ;;
    --reextract) REEXTRACT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "--video is required" >&2
  exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Video not found: ${VIDEO_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

video_name="$(basename "${VIDEO_PATH}")"
video_stem="${video_name%.*}"
safe_stem="$(printf '%s' "${video_stem}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${ROOT_DIR}/research_eval/temporal_subsampling/${safe_stem}"
fi
mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a FPS_VALUES <<< "${FPS_LIST}"

for fps in "${FPS_VALUES[@]}"; do
  frames_dir="${ROOT_DIR}/tmp_frames/${safe_stem}_ffmpeg_frames_fps${fps}"
  if [[ "${REEXTRACT}" -eq 1 ]]; then
    rm -rf "${frames_dir}"
  fi
  if [[ ! -d "${frames_dir}" ]] || [[ -z "$(find "${frames_dir}" -maxdepth 1 -type f -name '*.jpg' -print -quit 2>/dev/null)" ]]; then
    mkdir -p "${frames_dir}"
    echo "Extracting ${VIDEO_PATH} at ${fps} fps -> ${frames_dir}"
    ffmpeg -y -i "${VIDEO_PATH}" -vf "fps=${fps}" "${frames_dir}/%06d.jpg"
  else
    echo "Reusing extracted frames from ${frames_dir}"
  fi

  json_out="${OUTPUT_DIR}/fps_${fps}.json"
  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_sequence.py"
    --model_path "${MODEL_PATH}"
    --image_folder "${frames_dir}"
    --mode "${MODE}"
    --use_sdpa
    --num_scale_frames "${NUM_SCALE_FRAMES}"
    --camera_num_iterations "${CAMERA_NUM_ITERATIONS}"
    --preprocess_mode "${PREPROCESS_MODE}"
    --json_out "${json_out}"
    --tag "temporal_fps_${fps}"
  )
  if [[ -n "${FIRST_K}" ]]; then
    cmd+=(--first_k "${FIRST_K}")
  fi
  if [[ "${MODE}" == "windowed" ]]; then
    cmd+=(--window_size "${WINDOW_SIZE}")
  fi

  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
done

echo "Wrote temporal subsampling reports to ${OUTPUT_DIR}"
