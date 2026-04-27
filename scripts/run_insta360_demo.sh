#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_insta360_demo.sh --source-dir PATH [options] [-- <extra demo.py args>]

Modes:
  1) Pre-rendered perspective frames (for example outdoor_perspectives_8)
     scripts/run_insta360_demo.sh \
       --source-dir /path/to/outdoor_perspectives_8 \
       --input-mode perspective \
       --azimuth 090 \
       --port 18097

  2) Raw Insta360 .insp stills using a conservative center crop from one lens
     scripts/run_insta360_demo.sh \
       --source-dir /path/to/outdoor \
       --input-mode crop \
       --lens right \
       --port 18098

Options:
  --source-dir PATH             Input directory (required)
  --input-mode MODE             perspective or crop (default: perspective)
  --azimuth DEG                 Perspective azimuth like 000, 045, 090... (default: 090)
  --elevation DEG               Perspective elevation token in filenames (default: 000)
  --lens SIDE                   crop mode only: left or right (default: right)
  --crop-width PX               crop mode only: center crop width (default: 2800)
  --crop-height PX              crop mode only: center crop height (default: 2000)
  --prepared-dir PATH           Reuse or write prepared frames here
  --rebuild                     Force rebuilding the prepared frame directory
  --model-path PATH             Checkpoint path
  --python PATH                 Python executable to run demo.py
  --port N                      Viewer port (default: 8080)
  --mode MODE                   demo.py mode: streaming or windowed (default: windowed)
  --window-size N               windowed mode window size (default: 16)
  --overlap-size N              windowed mode overlap size (default: 4)
  --first-k N                   Limit frames passed to demo.py
  --num-scale-frames N          demo.py --num_scale_frames (default: 2)
  --camera-num-iterations N     demo.py --camera_num_iterations (default: 1)
  --mask-sky                    Enable sky masking
  -h, --help                    Show this help

Examples:
  scripts/run_insta360_demo.sh \
    --source-dir '/media/slam/My Passport/360imgs/DCIM/Camera01/outdoor_perspectives_8' \
    --input-mode perspective \
    --azimuth 135 \
    --port 18099

  scripts/run_insta360_demo.sh \
    --source-dir '/media/slam/My Passport/360imgs/DCIM/Camera01/outdoor' \
    --input-mode crop \
    --lens left \
    --port 18100
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${ROOT_DIR}/tmp_frames"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
MODEL_PATH="${ROOT_DIR}/checkpoints/lingbot-map.pt"

SOURCE_DIR=""
INPUT_MODE="perspective"
AZIMUTH="090"
ELEVATION="000"
LENS="right"
CROP_WIDTH=2800
CROP_HEIGHT=2000
PREPARED_DIR=""
REBUILD=0
PORT=8080
MODE="windowed"
WINDOW_SIZE=16
OVERLAP_SIZE=4
FIRST_K=""
NUM_SCALE_FRAMES=2
CAMERA_NUM_ITERATIONS=1
MASK_SKY=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --input-mode)
      INPUT_MODE="$2"
      shift 2
      ;;
    --azimuth)
      AZIMUTH="$2"
      shift 2
      ;;
    --elevation)
      ELEVATION="$2"
      shift 2
      ;;
    --lens)
      LENS="$2"
      shift 2
      ;;
    --crop-width)
      CROP_WIDTH="$2"
      shift 2
      ;;
    --crop-height)
      CROP_HEIGHT="$2"
      shift 2
      ;;
    --prepared-dir)
      PREPARED_DIR="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
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
    --overlap-size)
      OVERLAP_SIZE="$2"
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

if [[ -z "${SOURCE_DIR}" ]]; then
  echo "--source-dir is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory not found: ${SOURCE_DIR}" >&2
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

if [[ "${INPUT_MODE}" != "perspective" && "${INPUT_MODE}" != "crop" ]]; then
  echo "--input-mode must be perspective or crop" >&2
  exit 1
fi

if [[ "${LENS}" != "left" && "${LENS}" != "right" ]]; then
  echo "--lens must be left or right" >&2
  exit 1
fi

source_name="$(basename "${SOURCE_DIR}")"
safe_source="$(printf '%s' "${source_name}" | tr ' ' '_' | tr -cd '[:alnum:]_.-')"

if [[ -z "${PREPARED_DIR}" ]]; then
  if [[ "${INPUT_MODE}" == "perspective" ]]; then
    PREPARED_DIR="${TMP_ROOT}/${safe_source}_az${AZIMUTH}_el${ELEVATION}"
  else
    PREPARED_DIR="${TMP_ROOT}/${safe_source}_${LENS}_center_crops"
  fi
fi

if [[ "${REBUILD}" -eq 1 ]]; then
  rm -rf "${PREPARED_DIR}"
fi

prepare_perspective_frames() {
  local count
  mkdir -p "${PREPARED_DIR}"
  count=0
  while IFS= read -r frame; do
    local target
    target="$(printf '%s/%06d.jpg' "${PREPARED_DIR}" "${count}")"
    ln -sfn "${frame}" "${target}"
    count=$((count + 1))
  done < <(find "${SOURCE_DIR}" -maxdepth 1 -type f -name "frame*_az${AZIMUTH}_el${ELEVATION}.jpg" | sort)

  if [[ "${count}" -eq 0 ]]; then
    echo "No frames found for azimuth ${AZIMUTH} elevation ${ELEVATION} in ${SOURCE_DIR}" >&2
    exit 1
  fi
}

prepare_crop_frames() {
  mkdir -p "${PREPARED_DIR}"
  "${PYTHON_BIN}" - <<'PY' "${SOURCE_DIR}" "${PREPARED_DIR}" "${LENS}" "${CROP_WIDTH}" "${CROP_HEIGHT}"
from pathlib import Path
import sys
from PIL import Image

source_dir = Path(sys.argv[1])
prepared_dir = Path(sys.argv[2])
lens = sys.argv[3]
crop_width = int(sys.argv[4])
crop_height = int(sys.argv[5])

files = sorted(source_dir.glob("*.insp"))
if not files:
    raise SystemExit(f"No .insp files found in {source_dir}")

first_img = Image.open(files[0])
img_w, img_h = first_img.size
center_y = img_h // 2
center_x = img_w // 4 if lens == "left" else (img_w * 3) // 4

left = center_x - crop_width // 2
top = center_y - crop_height // 2
right = left + crop_width
bottom = top + crop_height

if left < 0 or top < 0 or right > img_w or bottom > img_h:
    raise SystemExit(
        f"Crop box {(left, top, right, bottom)} exceeds image size {(img_w, img_h)}"
    )

for idx, path in enumerate(files):
    img = Image.open(path).convert("RGB")
    crop = img.crop((left, top, right, bottom))
    crop.save(prepared_dir / f"{idx:06d}.jpg", quality=95)

print(f"Prepared {len(files)} crop frames in {prepared_dir}")
PY
}

count_prepared_frames() {
  find "${PREPARED_DIR}" -maxdepth 1 \( -type f -o -type l \) -name '*.jpg' | wc -l
}

frame_count=0
if [[ -d "${PREPARED_DIR}" ]]; then
  frame_count="$(count_prepared_frames)"
fi

if [[ "${frame_count}" -eq 0 ]]; then
  echo "Preparing frames in ${PREPARED_DIR}"
  if [[ "${INPUT_MODE}" == "perspective" ]]; then
    prepare_perspective_frames
  else
    prepare_crop_frames
  fi
  frame_count="$(count_prepared_frames)"
else
  echo "Reusing ${frame_count} prepared frames from ${PREPARED_DIR}"
fi

if [[ "${frame_count}" -eq 0 ]]; then
  echo "No frames were prepared in ${PREPARED_DIR}" >&2
  exit 1
fi

echo "Running demo.py with ${frame_count} frames"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  "${PYTHON_BIN}" demo.py
  --model_path "${MODEL_PATH}"
  --image_folder "${PREPARED_DIR}"
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
  cmd+=(--window_size "${WINDOW_SIZE}" --overlap_size "${OVERLAP_SIZE}")
fi

if [[ "${MASK_SKY}" -eq 1 ]]; then
  cmd+=(--mask_sky)
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "Command: ${cmd[*]}"
exec "${cmd[@]}"
