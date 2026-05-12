#!/usr/bin/env bash
# Export preprocessed input bins under each example's model/{int8,fp32}/input_bins/
# by running demos with --skip-runtime and --output-dir set to that model directory.
# Requires: built demos (cmake --build build), .dla present, Neuron runtime loadable.
# Run from anywhere; changes cwd to repo root.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

shopt -s nullglob

BUILD="${BUILD:-build}"
if [[ ! -d "$BUILD" ]]; then
  echo "error: missing build dir '$BUILD' (configure and build first)" >&2
  exit 1
fi

# demo_executable:examples_subdir (paths under examples/)
JOBS=(
  "yolov5s_demo:yolov5s"
  "yolov8n_demo:yolov8n"
  "yolo11n_demo:yolo11n"
  "yolov10n_demo:yolov10n"
  "yolo26n_demo:yolo26n"
  "mobilenet_v1_demo:mobilenet_v1"
  "mobilenet_v2_demo:mobilenet_v2"
  "ssd_mobilenet_v2_demo:ssd_mobilenet_v2"
  "efficientnet_b0_classification_demo:efficientnet_b0_classification"
  "resnet50_demo:resnet50"
  "mobilenet_v3_large_demo:mobilenet_v3_large"
  "mobilenet_v3_small_demo:mobilenet_v3_small"
  "yolo12n_demo:yolo12n"
  "yolov8n-seg_demo:yolov8n-seg"
  "yolo11n-seg_demo:yolo11n-seg"
  "yolo26n-seg_demo:yolo26n-seg"
  "yolov8n-pose_demo:yolov8n-pose"
  "yolov8n-obb_demo:yolov8n-obb"
  "yolo11n-obb_demo:yolo11n-obb"
  "yolo26n-obb_demo:yolo26n-obb"
  "yolo11n-pose_demo:yolo11n-pose"
  "yolo26n-pose_demo:yolo26n-pose"
)

skipped_int8=0
skipped_fp32=0
failed=0

for job in "${JOBS[@]}"; do
  exe="${job%%:*}"
  dir="${job##*:}"
  bin="${BUILD}/${exe}"
  if [[ ! -e "$bin" ]]; then
    echo "error: missing executable: $bin" >&2
    exit 1
  fi

  int8_dir="examples/${dir}/model/int8"
  fp32_dir="examples/${dir}/model/fp32"
  int8_dlas=( "${int8_dir}"/*.dla )
  fp32_dlas=( "${fp32_dir}"/*.dla )

  if ((${#int8_dlas[@]} > 0)); then
    echo "==> INT8  ${dir}  ->  ${int8_dir}/input_bins/"
    if ! "${bin}" --skip-runtime --output-dir "${int8_dir}"; then
      echo "FAILED: INT8 export for ${dir}" >&2
      ((failed++)) || true
    fi
  else
    echo "SKIP INT8 ${dir} (no .dla under ${int8_dir})"
    ((skipped_int8++)) || true
  fi

  if ((${#fp32_dlas[@]} > 0)); then
    echo "==> FP32  ${dir}  ->  ${fp32_dir}/input_bins/"
    if ! "${bin}" --skip-runtime --fp32 --output-dir "${fp32_dir}"; then
      echo "FAILED: FP32 export for ${dir}" >&2
      ((failed++)) || true
    fi
  else
    echo "SKIP FP32 ${dir} (no .dla under ${fp32_dir})"
    ((skipped_fp32++)) || true
  fi
done

echo "---"
echo "done. skipped int8: ${skipped_int8}, skipped fp32: ${skipped_fp32}, failed runs: ${failed}"
if ((failed > 0)); then
  exit 1
fi
exit 0
