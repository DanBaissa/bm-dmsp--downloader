#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 7 ]; then
  echo "Usage: bash aws/run_fixed_shard_download.sh <shard_id> <shard_csv> <warmup_csv> [data_root] [cache_root] [patch_size] [max_workers]"
  exit 1
fi

shard_id="$1"
shard_csv="$2"
warmup_csv="$3"
data_root="${4:-/data/bm_dmsp_runs}"
cache_root="${5:-/data/bm_dmsp_cache}"
patch_size="${6:-1000}"
max_workers="${7:-8}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -n "${NASA_TOKEN:-}" ]; then
  NASA_TOKEN="${NASA_TOKEN%$'\r'}"
  export NASA_TOKEN
fi

if [ -z "${NASA_TOKEN:-}" ]; then
  echo "NASA_TOKEN is not set. Copy your .env to the repo root on EC2 before running."
  exit 1
fi

run_root="${data_root}/${shard_id}"
warmup_root="${run_root}/warmup"
full_root="${run_root}/full"
shard_cache_root="${cache_root}/${shard_id}"
complete_marker="${full_root}/.complete"

mkdir -p "$warmup_root" "$full_root" "$shard_cache_root"

if [ ! -f "$shard_csv" ]; then
  echo "Shard CSV not found: $shard_csv"
  exit 1
fi

if [ ! -f "$warmup_csv" ]; then
  echo "Warm-up CSV not found: $warmup_csv"
  exit 1
fi

shard_csv_abs="$(cd "$(dirname "$shard_csv")" && pwd)/$(basename "$shard_csv")"
warmup_csv_abs="$(cd "$(dirname "$warmup_csv")" && pwd)/$(basename "$warmup_csv")"

if [ -f "$complete_marker" ]; then
  echo "${shard_id} already complete"
  exit 0
fi

echo "Running warm-up subset for ${shard_id}"
python data_sampler.py \
  --skip-sampling \
  --locations-csv "$warmup_csv_abs" \
  --patch-size "$patch_size" \
  --max-workers "$max_workers" \
  --output-folder "$warmup_root" \
  --cache-root "$shard_cache_root"

echo "Running full shard for ${shard_id}"
python data_sampler.py \
  --skip-sampling \
  --locations-csv "$shard_csv_abs" \
  --patch-size "$patch_size" \
  --max-workers "$max_workers" \
  --output-folder "$full_root" \
  --cache-root "$shard_cache_root"

touch "$complete_marker"
echo "${shard_id} complete"
