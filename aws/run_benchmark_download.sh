#!/usr/bin/env bash
set -euo pipefail

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

BENCHMARK_ROOT="${BENCHMARK_ROOT:-ec2_benchmark_run}"
BENCHMARK_CSV_NAME="${BENCHMARK_CSV_NAME:-benchmark_locations.csv}"
SAMPLES_PER_BIN="${SAMPLES_PER_BIN:-5}"
PATCH_SIZE="${PATCH_SIZE:-1000}"
SAMPLING_SEED="${SAMPLING_SEED:-13492}"
DATE_SEED="${DATE_SEED:-13492}"
WORKER_MATRIX="${WORKER_MATRIX:-2 4 8}"
export BENCHMARK_ROOT

mkdir -p "$BENCHMARK_ROOT"

python data_sampler.py \
  --sample-only \
  --samples-per-bin "$SAMPLES_PER_BIN" \
  --patch-size "$PATCH_SIZE" \
  --output-folder "$BENCHMARK_ROOT" \
  --locations-csv "$BENCHMARK_CSV_NAME" \
  --sampling-seed "$SAMPLING_SEED" \
  --date-seed "$DATE_SEED"

benchmark_csv="$BENCHMARK_ROOT/$BENCHMARK_CSV_NAME"
benchmark_csv_abs="$(cd "$(dirname "$benchmark_csv")" && pwd)/$(basename "$benchmark_csv")"

for workers in $WORKER_MATRIX; do
  worker_root="$BENCHMARK_ROOT/workers_${workers}"
  cache_root="$worker_root/cache"
  cold_output="$worker_root/cold"
  warm_output="$worker_root/warm"

  rm -rf "$worker_root"
  mkdir -p "$worker_root"

  echo "Running cold-cache benchmark with max-workers=${workers}"
  python data_sampler.py \
    --skip-sampling \
    --locations-csv "$benchmark_csv_abs" \
    --patch-size "$PATCH_SIZE" \
    --max-workers "$workers" \
    --output-folder "$cold_output" \
    --cache-root "$cache_root"

  rm -rf "$warm_output"

  echo "Running warm-cache benchmark with max-workers=${workers}"
  python data_sampler.py \
    --skip-sampling \
    --locations-csv "$benchmark_csv_abs" \
    --patch-size "$PATCH_SIZE" \
    --max-workers "$workers" \
    --output-folder "$warm_output" \
    --cache-root "$cache_root"
done

python - <<'PY'
import csv
import json
import os
from pathlib import Path

benchmark_root = Path(os.environ.get("BENCHMARK_ROOT", "ec2_benchmark_run"))

rows = []
for timings_path in sorted(benchmark_root.glob("workers_*/**/timings.json")):
    worker_label = timings_path.parents[1].name
    mode = timings_path.parent.name
    payload = json.loads(timings_path.read_text(encoding="utf-8"))
    stage_seconds = payload.get("stage_seconds", {})
    bytes_downloaded = payload.get("bytes_downloaded", {})
    rows.append(
        {
            "workers": worker_label.replace("workers_", ""),
            "mode": mode,
            "total_seconds": stage_seconds.get("total", 0.0),
            "bm_seconds": stage_seconds.get("bm_processing", 0.0),
            "dmsp_seconds": stage_seconds.get("dmsp_processing", 0.0),
            "bm_bytes": bytes_downloaded.get("bm", 0),
            "dmsp_bytes": bytes_downloaded.get("dmsp", 0),
        }
    )

summary_path = benchmark_root / "summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "workers",
            "mode",
            "total_seconds",
            "bm_seconds",
            "dmsp_seconds",
            "bm_bytes",
            "dmsp_bytes",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote benchmark summary to {summary_path}")
PY
