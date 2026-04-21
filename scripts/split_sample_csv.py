#!/usr/bin/env python3
"""Split a fixed sampled_locations.csv into balanced shard CSVs plus warm-up subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a sampled locations CSV into shard CSVs.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input sampled_locations.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for shard CSVs")
    parser.add_argument("--shards", type=int, default=10, help="Number of output shards")
    parser.add_argument(
        "--warmup-rows-per-shard",
        type=int,
        default=25,
        help="Number of rows from each shard to write to the warm-up subset CSV",
    )
    return parser.parse_args()


def build_shards(df: pd.DataFrame, shard_count: int) -> list[pd.DataFrame]:
    if shard_count < 1:
        raise ValueError("--shards must be at least 1")

    working = df.copy()
    working["_source_order"] = range(len(working))
    shard_indices = [[] for _ in range(shard_count)]

    group_key = "population_bin" if "population_bin" in working.columns else None
    grouped = [(None, working)] if group_key is None else working.groupby(group_key, sort=True, dropna=False)

    for _, group in grouped:
        ordered = group.sort_values("_source_order")
        for offset, row_index in enumerate(ordered.index):
            shard_indices[offset % shard_count].append(row_index)

    shards: list[pd.DataFrame] = []
    for indices in shard_indices:
        shard = working.loc[indices].sort_values("_source_order").drop(columns="_source_order")
        shards.append(shard.reset_index(drop=True))
    return shards


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"{args.input_csv} is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = build_shards(df, args.shards)

    summary_rows = []
    for shard_number, shard_df in enumerate(shards, start=1):
        shard_name = f"shard_{shard_number:02d}"
        shard_path = args.output_dir / f"{shard_name}.csv"
        warmup_path = args.output_dir / f"{shard_name}_warmup.csv"

        shard_df.to_csv(shard_path, index=False)
        warmup_df = shard_df.head(args.warmup_rows_per_shard)
        warmup_df.to_csv(warmup_path, index=False)

        summary_rows.append(
            {
                "shard_id": shard_name,
                "rows": len(shard_df),
                "warmup_rows": len(warmup_df),
                "csv_path": str(shard_path.resolve()),
                "warmup_csv_path": str(warmup_path.resolve()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "shard_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(shards)} shards to {args.output_dir}")
    print(f"Wrote shard summary to {summary_path}")


if __name__ == "__main__":
    main()
