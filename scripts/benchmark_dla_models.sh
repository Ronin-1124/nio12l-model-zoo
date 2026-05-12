#!/usr/bin/env bash
# Benchmark all DLA models found under examples/*/model/{int8,fp32}/.
# Uses one pre-exported input bin from each model directory's input_bins/.
#
# Limit to specific examples (optional; default = all under examples/):
#   ./scripts/benchmark_dla_models.sh yolov8n
#   ./scripts/benchmark_dla_models.sh yolov8n yolo11n
#   MODELS=yolov8n,yolo11n ./scripts/benchmark_dla_models.sh
#   (command-line names win if both are given)
#
# Environment overrides:
#   COUNT=1000        Number of inference repeats passed to neuronrt -c
#   BATCH=1           Batch/backlog value passed to neuronrt -b
#   RESULT_ROOT=...   Output directory for CSV, logs, and temporary output bins
#   MODELS=a,b        Comma-separated example dir names (e.g. yolov8n, yolo11n-seg)

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

COUNT="${COUNT:-1000}"
BATCH="${BATCH:-1}"
RESULT_ROOT="${RESULT_ROOT:-benchmark_results/$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${RESULT_ROOT}/logs"
OUTPUT_DIR="${RESULT_ROOT}/neuron_outputs"
CSV="${RESULT_ROOT}/results.csv"

if ! [[ "$COUNT" =~ ^[0-9]+$ ]] || ((COUNT <= 0)); then
  echo "error: COUNT must be a positive integer (got: ${COUNT})" >&2
  exit 1
fi

if ! [[ "$BATCH" =~ ^[0-9]+$ ]] || ((BATCH <= 0)); then
  echo "error: BATCH must be a positive integer (got: ${BATCH})" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

write_csv_row() {
  local first=1
  local value
  for value in "$@"; do
    if ((first)); then
      first=0
    else
      printf ',' >> "$CSV"
    fi
    csv_escape "$value" >> "$CSV"
  done
  printf '\n' >> "$CSV"
}

safe_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  printf '%s' "$value"
}

query_output_count() {
  local dla="$1"
  local log="$2"
  local output_count

  neuronrt -a "$dla" -d > "$log" 2>&1
  output_count="$(
    awk '
      /^[[:space:]]*Output/ { in_output = 1; next }
      in_output && /^[[:space:]]*Input/ { in_output = 0 }
      in_output && /Handle[[:space:]]*=/ { count++ }
      END { print count + 0 }
    ' "$log"
  )"

  printf '%s' "$output_count"
}

declare -a selected_models=()
if (($# > 0)); then
  selected_models=("$@")
else
  if [[ -n "${MODELS:-}" ]]; then
    IFS=',' read -r -a _tmp_models <<< "$MODELS"
    for _t in "${_tmp_models[@]}"; do
      _t="${_t//[[:space:]]/}"
      [[ -n "$_t" ]] && selected_models+=("$_t")
    done
  fi
fi

shopt -s nullglob
all_dlas=(examples/*/model/int8/*.dla examples/*/model/fp32/*.dla)
dlas=()
if ((${#selected_models[@]} == 0)); then
  dlas=("${all_dlas[@]}")
  filter_note="all examples"
else
  filter_note="${selected_models[*]}"
  for dla in "${all_dlas[@]}"; do
    model="$(awk -F/ '{print $2}' <<< "$dla")"
    for s in "${selected_models[@]}"; do
      if [[ "$model" == "$s" ]]; then
        dlas+=("$dla")
        break
      fi
    done
  done
fi

write_csv_row \
  "model" "precision" "dla" "input_bin" "count" "batch" \
  "output_count" "total_ms" "avg_ms" "status" "log_path"

if ((${#all_dlas[@]} == 0)); then
  echo "error: no .dla files found under examples/*/model/{int8,fp32}/" >&2
  echo "empty results written to: ${CSV}" >&2
  exit 1
fi

if ((${#selected_models[@]} > 0)) && ((${#dlas[@]} == 0)); then
  echo "error: no .dla matched filter: ${filter_note}" >&2
  echo "  (example names must match examples/<name>/...)" >&2
  exit 1
fi

echo "benchmark filter: ${filter_note}"

ok_count=0
failed_count=0
skipped_count=0

for dla in "${dlas[@]}"; do
  model="$(awk -F/ '{print $2}' <<< "$dla")"
  precision="$(awk -F/ '{print $4}' <<< "$dla")"
  model_dir="$(dirname "$dla")"
  dla_base="$(basename "$dla" .dla)"

  input_bins=("${model_dir}/input_bins/"*.bin)
  if ((${#input_bins[@]} == 0)); then
    echo "SKIP ${model} ${precision}: no input_bins/*.bin"
    write_csv_row "$model" "$precision" "$dla" "" "$COUNT" "$BATCH" "" "" "" "missing_input" ""
    ((skipped_count++)) || true
    continue
  fi

  input_bin="${input_bins[0]}"
  input_stem="$(basename "$input_bin" .bin)"
  safe="$(safe_name "${model}_${precision}_${dla_base}_${input_stem}")"
  io_log="${LOG_DIR}/${safe}_io.log"
  run_log="${LOG_DIR}/${safe}.log"

  output_count="$(query_output_count "$dla" "$io_log")"
  if ! [[ "$output_count" =~ ^[0-9]+$ ]] || ((output_count <= 0)); then
    echo "FAIL ${model} ${precision}: cannot query output count"
    write_csv_row "$model" "$precision" "$dla" "$input_bin" "$COUNT" "$BATCH" \
      "$output_count" "" "" "io_query_failed" "$io_log"
    ((failed_count++)) || true
    continue
  fi

  output_args=()
  for ((i = 0; i < output_count; ++i)); do
    output_args+=("-o" "${OUTPUT_DIR}/${safe}_out${i}.bin")
  done

  echo "RUN ${model} ${precision}: ${dla_base}, input=${input_stem}, outputs=${output_count}"
  neuronrt -m hw -a "$dla" -c "$COUNT" -b "$BATCH" -i "$input_bin" "${output_args[@]}" \
    > "$run_log" 2>&1
  run_status=$?

  total_ms="$(
    sed -n 's/.*Total inference time = \([0-9][0-9]*\) ms.*/\1/p' "$run_log" | tail -n 1
  )"

  if ((run_status == 0)) && [[ -n "$total_ms" ]]; then
    avg_ms="$(awk -v total="$total_ms" -v count="$COUNT" 'BEGIN { printf "%.3f", total / count }')"
    echo "OK  ${model} ${precision}: total=${total_ms} ms avg=${avg_ms} ms"
    write_csv_row "$model" "$precision" "$dla" "$input_bin" "$COUNT" "$BATCH" \
      "$output_count" "$total_ms" "$avg_ms" "ok" "$run_log"
    ((ok_count++)) || true
  else
    echo "FAIL ${model} ${precision}: neuronrt exit=${run_status}"
    write_csv_row "$model" "$precision" "$dla" "$input_bin" "$COUNT" "$BATCH" \
      "$output_count" "$total_ms" "" "failed" "$run_log"
    ((failed_count++)) || true
  fi
done

echo "---"
echo "results: ${CSV}"
echo "logs:    ${LOG_DIR}"
echo "ok: ${ok_count}, skipped: ${skipped_count}, failed: ${failed_count}"

if ((failed_count > 0)); then
  exit 1
fi
exit 0
