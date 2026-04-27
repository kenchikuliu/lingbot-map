#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_video_demo.sh --video /path/to/video.mp4 [options] [-- <extra demo.py args>]

Options:
  --video PATH                  Input video path (required)
  --model-path PATH             Checkpoint path
  --python PATH                 Python executable to run demo.py
  --fps N                       Extraction FPS for ffmpeg (default: 1)
  --frames-dir PATH             Reuse or write extracted frames here
  --reextract                   Force re-extraction even if frames already exist
  --port N                      Viewer port (default: 8080)
  --mode MODE                   demo.py mode: streaming or windowed (default: streaming)
  --window-size N               Window size for windowed mode (default: 128)
  --first-k N                   Limit frames passed to demo.py
  --num-scale-frames N          demo.py --num_scale_frames (default: 2)
  --camera-num-iterations N     demo.py --camera_num_iterations (default: 1)
  --mask-sky                    Enable sky masking
  -h, --help                    Show this help

Examples:
  scripts/run_video_demo.sh \
    --video /home/slam/datasets/zhuyevideos/video_20260418_190605\(1\).mp4 \
    --fps 1 --port 18093

  scripts/run_video_demo.sh \
    --video /home/slam/datasets/zhuyevideos/video_20260418_190605\(1\).mp4 \
    --fps 2 --port 18094 -- --downsample_factor 6
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${ROOT_DIR}/tmp_frames"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
MODEL_PATH="${ROOT_DIR}/checkpoints/lingbot-map.pt"
FPS=1
FRAMES_DIR=""
REEXTRACT=0
PORT=8080
MODE="streaming"
WINDOW_SIZE=128
FIRST_K=""
NUM_SCALE_FRAMES=2
CAMERA_NUM_ITERATIONS=1
MASK_SKY=0
VIDEO_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO_PATH="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --frames-dir)
      FRAMES_DIR="$2"
      shift 2
      ;;
    --reextract)
      REEXTRACT=1
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --window-size)
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --first-k)
      FIRST_K="$2"
      shift 2
      ;;
    --num-scale-frames)
      NUM_SCALE_FRAMES="$2"
      shift 2
      ;;
    --camera-num-iterations)
      CAMERA_NUM_ITERATIONS="$2"
      shift 2
      ;;
    --mask-sky)
      MASK_SKY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${VIDEO_PATH}" ]]; then
  echo "--video is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${VIDEO_PATH}" ]]; then
  echo "Video not found: ${VIDEO_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but was not found in PATH" >&2
  exit 1
fi

if [[ -z "${FRAMES_DIR}" ]]; then
  video_name="$(basename "${VIDEO_PATH}")"
  video_stem="${video_name%.*}"
  safe_stem="$(printf '%s' "${video_stem}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"
  FRAMES_DIR="${TMP_ROOT}/${safe_stem}_ffmpeg_frames_fps${FPS}"
fi

extract_frames() {
  mkdir -p "${FRAMES_DIR}"
  echo "Extracting frames to ${FRAMES_DIR} at ${FPS} fps"
  ffmpeg -y -i "${VIDEO_PATH}" -vf "fps=${FPS}" "${FRAMES_DIR}/%06d.jpg"
}

frame_count=0
if [[ -d "${FRAMES_DIR}" ]]; then
  frame_count="$(find "${FRAMES_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
fi

if [[ "${REEXTRACT}" -eq 1 ]]; then
  rm -rf "${FRAMES_DIR}"
  frame_count=0
fi

if [[ "${frame_count}" -eq 0 ]]; then
  extract_frames
  frame_count="$(find "${FRAMES_DIR}" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
else
  echo "Reusing ${frame_count} extracted frames from ${FRAMES_DIR}"
fi

if [[ "${frame_count}" -eq 0 ]]; then
  echo "No frames were extracted to ${FRAMES_DIR}" >&2
  exit 1
fi

echo "Running demo.py with ${frame_count} frames"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "${PYTHON_BIN}" demo.py
  --model_path "${MODEL_PATH}"
  --image_folder "${FRAMES_DIR}"
  --mode "${MODE}"
  --use_sdpa
  --num_scale_frames "${NUM_SCALE_FRAMES}"
  --camera_num_iterations "${CAMERA_NUM_ITERATIONS}"
  --port "${PORT}"
)

if [[ -n "${FIRST_K}" ]]; then
  cmd+=(--first_k "${FIRST_K}")
fi

if [[ "${MODE}" == "windowed" ]]; then
  cmd+=(--window_size "${WINDOW_SIZE}")
fi

if [[ "${MASK_SKY}" -eq 1 ]]; then
  cmd+=(--mask_sky)
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "Command: ${cmd[*]}"
exec "${cmd[@]}"
