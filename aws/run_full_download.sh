#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Windows CRLF in .env can leave a trailing carriage return in the token.
if [ -n "${NASA_TOKEN:-}" ]; then
  NASA_TOKEN="${NASA_TOKEN%$'\r'}"
  export NASA_TOKEN
fi

if [ -z "${NASA_TOKEN:-}" ]; then
  echo "NASA_TOKEN is not set. Copy your .env to the repo root on EC2 before running."
  exit 1
fi

mkdir -p ec2_full_run

for batch in 1 2 3 4; do
  sampling_seed=$((13491 + batch))
  date_seed=$((14491 + batch))
  batch_dir="ec2_full_run/batch_${batch}"
  complete_marker="${batch_dir}/.complete"
  sampled_csv="${batch_dir}/sampled_locations.csv"

  mkdir -p "$batch_dir"

  if [ -f "$complete_marker" ]; then
    echo "Skipping batch ${batch}; completion marker already exists"
    continue
  fi

  echo "Starting batch ${batch} with sampling seed ${sampling_seed} and date seed ${date_seed}"

  args=(
    data_sampler.py
    --samples-per-bin 500
    --patch-size 1000
    --max-workers 2
    --output-folder "$batch_dir"
    --sampling-seed "${sampling_seed}"
    --date-seed "${date_seed}"
  )

  if [ -f "$sampled_csv" ]; then
    echo "Resuming batch ${batch} from existing sampled_locations.csv"
    args+=(--skip-sampling)
  fi

  python "${args[@]}"
  touch "$complete_marker"
done
