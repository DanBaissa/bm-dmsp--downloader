#!/usr/bin/env python3
"""Validate a shard output directory produced by data_sampler.py."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a shard output directory.")
    parser.add_argument("--root", type=Path, required=True, help="Shard full output directory")
    parser.add_argument("--shard-id", required=True, help="Shard identifier for reporting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.root / "bm_dmsp_pairs.csv"
    timings = args.root / "timings.json"

    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    pattern = re.compile(r"F\d{2}(\d{8})")

    summary = {
        "shard_id": args.shard_id,
        "manifest_exists": manifest.exists(),
        "timings_exists": timings.exists(),
        "rows": len(rows),
        "source_date_mismatches": 0,
        "shape_mismatches": 0,
        "transform_mismatches": 0,
        "crs_mismatches": 0,
        "missing_bm_files": 0,
        "missing_dmsp_files": 0,
    }

    for row in rows:
        match = pattern.search(Path(row["dmsp_source_key"]).name)
        if not match:
            summary["source_date_mismatches"] += 1
        else:
            ymd = match.group(1)
            source_date = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
            if source_date != row["date"]:
                summary["source_date_mismatches"] += 1

        bm_path = Path(row["bm_patch"])
        dmsp_path = Path(row["dmsp_patch"])
        if not bm_path.exists():
            summary["missing_bm_files"] += 1
            continue
        if not dmsp_path.exists():
            summary["missing_dmsp_files"] += 1
            continue

        with rasterio.open(bm_path) as bm, rasterio.open(dmsp_path) as dmsp:
            if (bm.width, bm.height) != (dmsp.width, dmsp.height):
                summary["shape_mismatches"] += 1
            if tuple(bm.transform) != tuple(dmsp.transform):
                summary["transform_mismatches"] += 1
            if bm.crs != dmsp.crs:
                summary["crs_mismatches"] += 1

    if timings.exists():
        payload = json.loads(timings.read_text(encoding="utf-8"))
        summary["total_seconds"] = payload.get("stage_seconds", {}).get("total")
        summary["manifest_rows_metric"] = payload.get("counts", {}).get("manifest_rows")

    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
