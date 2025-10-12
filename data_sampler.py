"""Entrypoint for generating samples and downloading paired BM/DMSP GeoTIFFs."""
from __future__ import annotations

import argparse
from pathlib import Path

from bm_dmsp_downloader import (
    SampleRequest,
    assign_dates_to_samples,
    build_patch_manifest,
    generate_population_balanced_samples,
    list_dmsp_dates,
    load_nasa_token,
    match_dmsp_to_bm_patches,
    process_samples_parallel,
)

DATA_DIR = Path("Data")
LANDSCAN_RASTER = DATA_DIR / "Global_2012" / "landscan-global-2012.tif"
COUNTRY_SHAPEFILE = DATA_DIR / "World_Countries" / "World_Countries_Generalized.shp"
TILE_SHAPEFILE = DATA_DIR / "Black_Marble_IDs" / "Black_Marble_World_tiles.shp"

RAW_DATA_DIR = Path("Raw_NL_Data")
BM_OUTPUT_DIR = RAW_DATA_DIR / "BM data"
DMSP_OUTPUT_DIR = RAW_DATA_DIR / "DMSP data"
SAMPLE_CSV = RAW_DATA_DIR / "sample_locations.csv"
MANIFEST_CSV = RAW_DATA_DIR / "bm_dmsp_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paired BM and DMSP GeoTIFFs")
    parser.add_argument("--patch-size", type=int, default=1000, help="Patch size in pixels")
    parser.add_argument("--bm-workers", type=int, default=4, help="Parallel workers for BM downloads")
    parser.add_argument("--dmsp-workers", type=int, default=4, help="Parallel workers for DMSP reprojection")
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=200,
        help="Number of LandScan samples per log population bin",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = load_nasa_token()

    samples = generate_population_balanced_samples(
        raster_path=LANDSCAN_RASTER,
        shapefile_path=COUNTRY_SHAPEFILE,
        samples_per_bin=args.samples_per_bin,
    )

    dmsp_dates = list_dmsp_dates()
    samples = assign_dates_to_samples(samples, dmsp_dates)

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(SAMPLE_CSV, index=False)
    print(f"Wrote {len(samples)} sample locations to {SAMPLE_CSV}")

    sample_requests = [SampleRequest.from_dict(row) for row in samples.to_dict(orient="records")]

    bm_paths = process_samples_parallel(
        sample_requests,
        patch_size_pix=args.patch_size,
        collection_id="C3365931269-LAADS",
        token=token,
        tile_shapefile_path=TILE_SHAPEFILE,
        output_folder=BM_OUTPUT_DIR,
        temp_folder=Path("temp_dl"),
        max_workers=args.bm_workers,
    )

    if not bm_paths:
        print("No BM patches were created. Aborting DMSP matching.")
        return

    dmsp_paths = match_dmsp_to_bm_patches(
        bm_patch_dir=BM_OUTPUT_DIR,
        dmsp_out_dir=DMSP_OUTPUT_DIR,
        temp_dir=Path("DMSP_Raw_Temp"),
        max_workers=args.dmsp_workers,
    )

    if not dmsp_paths:
        print("No DMSP matches were generated.")
        return

    build_patch_manifest(BM_OUTPUT_DIR, DMSP_OUTPUT_DIR, MANIFEST_CSV)


if __name__ == "__main__":
    main()
