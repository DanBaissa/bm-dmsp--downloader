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

mkdir -p ec2_test_run

python data_sampler.py \
  --samples-per-bin 2 \
  --patch-size 1000 \
  --max-workers 2 \
  --output-folder ec2_test_run \
  --sampling-seed 13492
