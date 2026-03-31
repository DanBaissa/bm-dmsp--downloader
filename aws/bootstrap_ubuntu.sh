#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y \
  gdal-bin \
  libgdal-dev \
  libgeos-dev \
  libproj-dev \
  proj-data \
  proj-bin \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Bootstrap complete."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  test -f .env && echo '.env found' || echo 'missing .env'"
echo "  bash aws/run_test_download.sh"
