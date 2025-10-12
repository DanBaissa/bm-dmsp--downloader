"""Simplified downloader for paired Black Marble and DMSP GeoTIFF patches.

This module focuses on two responsibilities:
    1. Download Black Marble nighttime lights patches at a configurable
       pixel resolution (default 1000×1000) centered on user supplied
       coordinates and dates.
    2. For every saved Black Marble patch, locate and save the best
       matching DMSP-OLS observation as a GeoTIFF reprojected onto the
       same grid.

The original Colab bootstrap, sampling logic, PNG conversions, and
augmentation routines have been removed so the script can be executed as a
repeatable command line tool.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import geopandas as gpd
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import Resampling, reproject
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """Geospatial request for a nighttime lights patch."""

    longitude: float
    latitude: float
    date: datetime

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Sample":
        try:
            lon = float(mapping["Longitude"])
            lat = float(mapping["Latitude"])
            date_value = mapping["date"]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Missing required column in sample: {exc}") from exc

        if isinstance(date_value, (datetime,)):
            date_obj = date_value
        else:
            date_obj = datetime.strptime(str(date_value), "%Y-%m-%d")
        return cls(longitude=lon, latitude=lat, date=date_obj)

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%m-%d")

    @property
    def dmsp_date_str(self) -> str:
        return self.date.strftime("%Y%m%d")


def load_samples(path: Path) -> List[Sample]:
    """Load sampling requests from a CSV or JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"Sample file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            rows = payload.get("samples", [])
        else:
            rows = payload
    else:  # pragma: no cover - command line validation
        raise ValueError("Samples file must be .csv or .json")

    samples = [Sample.from_mapping(row) for row in rows]
    if not samples:
        raise ValueError("No samples were loaded from the provided file.")
    return samples


# ---------------------------------------------------------------------------
# Black Marble download helpers
# ---------------------------------------------------------------------------


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    half_deg = (patch_size_pix * tif_res_deg) / 2
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    cmr_search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    bbox_str = ",".join(f"{coord:.6f}" for coord in bbox)
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": bbox_str,
        "page_size": 50,
    }
    response = requests.get(cmr_search_url, params=params, timeout=60)
    response.raise_for_status()
    h5_links: List[str] = []
    granules = response.json().get("feed", {}).get("entry", [])
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                h5_links.append(href)
    return h5_links


def download_file(url: str, destination: Path, token: str) -> Path:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    with destination.open("wb") as handle:
        handle.write(response.content)
    return destination


def convert_h5_to_geotiff(h5_path: Path, tile_shapefile: gpd.GeoDataFrame, temp_dir: Path) -> Path:
    with h5py.File(h5_path, "r") as handle:
        dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if dataset_path not in handle:
            raise KeyError(f"Dataset not found in {h5_path}: {dataset_path}")
        ntl_data = handle[dataset_path][...]

    match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not match:
        raise ValueError(f"Could not determine tile id from filename: {h5_path.name}")
    tile_id = match.group(0)
    bounds_row = tile_shapefile[tile_shapefile["TileID"] == tile_id]
    if bounds_row.empty:
        raise ValueError(f"Tile ID {tile_id} not present in tile shapefile")
    left, bottom, right, top = bounds_row.total_bounds

    tif_path = temp_dir / f"{h5_path.stem}.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=ntl_data.shape[0],
        width=ntl_data.shape[1],
        count=1,
        dtype=ntl_data.dtype,
        crs="EPSG:4326",
        transform=transform_from_bounds(left, bottom, right, top, ntl_data.shape[1], ntl_data.shape[0]),
    ) as dst:
        dst.write(ntl_data, 1)
    return tif_path


def enforce_patch_size(patch: np.ndarray, patch_size_pix: int) -> np.ndarray:
    patch = patch[:patch_size_pix, :patch_size_pix]
    pad_h = max(0, patch_size_pix - patch.shape[0])
    pad_w = max(0, patch_size_pix - patch.shape[1])
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return patch


def extract_patch_from_geotiff(tif_path: Path, sample: Sample, patch_size_pix: int) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif_path) as src:
        tif_res_deg = abs(src.transform.a)
        bbox = get_patch_bbox(sample.longitude, sample.latitude, patch_size_pix, tif_res_deg)
        window = window_from_bounds(*bbox, transform=src.transform, height=patch_size_pix, width=patch_size_pix)
        patch = src.read(1, window=window, boundless=True, fill_value=0)
        patch = enforce_patch_size(patch, patch_size_pix)
        patch_transform = src.window_transform(window)
        patch_meta = src.meta.copy()
        patch_meta.update({
            "height": patch_size_pix,
            "width": patch_size_pix,
            "transform": patch_transform,
        })
    return patch, patch_meta


def fetch_black_marble_patch(
    sample: Sample,
    patch_size_pix: int,
    collection_id: str,
    nasa_token: str,
    tile_shapefile: gpd.GeoDataFrame,
    output_folder: Path,
    temp_dir: Path,
) -> Optional[Path]:
    if sample.latitude < -60:
        print(f"Skipping sample south of -60°: {sample}")
        return None

    bbox = get_patch_bbox(sample.longitude, sample.latitude, patch_size_pix, tif_res_deg=0.004)
    if bbox[1] < -60:
        bbox[1] = -60
    if bbox[3] < -60:
        print(f"Bounding box crosses -60°S, skipping sample {sample}")
        return None

    try:
        links = search_nasa_cmr(collection_id, sample.date_str, bbox)
    except requests.HTTPError as exc:
        print(f"NASA CMR search failed for {sample.date_str} at ({sample.longitude:.3f}, {sample.latitude:.3f}): {exc}")
        return None

    if not links:
        print(f"No Black Marble granules found for {sample.date_str} at ({sample.longitude:.3f}, {sample.latitude:.3f})")
        return None

    temp_dir.mkdir(parents=True, exist_ok=True)
    h5_path = temp_dir / Path(links[0]).name
    if not h5_path.exists():
        try:
            download_file(links[0], h5_path, nasa_token)
        except requests.HTTPError as exc:
            print(f"Failed to download {links[0]}: {exc}")
            return None

    try:
        tif_path = convert_h5_to_geotiff(h5_path, tile_shapefile, temp_dir)
        patch, patch_meta = extract_patch_from_geotiff(tif_path, sample, patch_size_pix)
    except Exception as exc:
        print(f"Error creating patch for {sample}: {exc}")
        return None
    finally:
        if h5_path.exists():
            h5_path.unlink()
        if (temp_tif := temp_dir / f"{h5_path.stem}.tif").exists():
            temp_tif.unlink()

    out_name = f"BM_patch_{sample.date_str}_{sample.latitude:.4f}_{sample.longitude:.4f}.tif"
    out_path = output_folder / out_name
    with rasterio.open(out_path, "w", **patch_meta) as dst:
        dst.write(patch, 1)
    return out_path


# ---------------------------------------------------------------------------
# DMSP pairing helpers
# ---------------------------------------------------------------------------


def wait_for_file_release(path: Path, timeout: float = 10.0) -> None:
    start = time.time()
    while True:
        try:
            with path.open("rb"):
                return
        except PermissionError:
            if time.time() - start > timeout:
                raise
            time.sleep(0.5)


def safe_download(s3, bucket: str, key: str, outpath: Path, max_retries: int = 5) -> bool:
    for attempt in range(max_retries):
        try:
            s3.download_file(bucket, key, str(outpath))
            wait_for_file_release(outpath)
            return True
        except botocore.exceptions.EndpointConnectionError as exc:
            print(f"EndpointConnectionError on {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        except botocore.exceptions.ClientError as exc:
            print(f"ClientError on {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Other error on {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        time.sleep(2)
    print(f"Failed to download after {max_retries} attempts: {key}")
    return False


def group_by_f_number(file_keys: Iterable[str]) -> dict[str, List[str]]:
    groups: dict[str, List[str]] = {}
    for vis_key in file_keys:
        base = os.path.basename(vis_key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        groups.setdefault(f_number, []).append(vis_key)
    return groups


def reproject_to_bm_grid(src_path: Path, bm_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_data[src_data == 255] = np.nan
        dst = np.empty((bm_profile["height"], bm_profile["width"]), dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=bm_profile["transform"],
            dst_crs=bm_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dst


def process_bm_patch_for_best_fnumber(
    bm_patch_path: Path,
    file_keys: Sequence[str],
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
) -> List[Path]:
    saved_paths: List[Path] = []
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    groups = group_by_f_number(file_keys)
    for f_number, keys in groups.items():
        best_valid_pixels = 0
        best_vis_file: Optional[Path] = None
        best_vis_patch: Optional[np.ndarray] = None
        for vis_key in keys:
            temp_file = temp_dir / os.path.basename(vis_key)
            if not temp_file.exists():
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                if not safe_download(s3, bucket_name, vis_key, temp_file):
                    continue
            try:
                vis_patch = reproject_to_bm_grid(temp_file, bm_profile)
            except Exception as exc:  # pragma: no cover - logging only
                print(f"Error processing {vis_key}: {exc}")
                continue
            valid_pixels = np.sum(~np.isnan(vis_patch))
            median_val = float(np.nanmedian(vis_patch)) if valid_pixels > 0 else 0.0
            if valid_pixels > best_valid_pixels and median_val > 1:
                best_valid_pixels = int(valid_pixels)
                best_vis_file = temp_file
                best_vis_patch = vis_patch
        if best_vis_file is None or best_vis_patch is None:
            continue
        valid_fraction = best_valid_pixels / (bm_shape[0] * bm_shape[1])
        if valid_fraction < 0.10:
            print(f"Skipping {f_number} for {bm_patch_path.name}: only {valid_fraction:.2%} valid pixels")
            continue
        out_name = f"{f_number}_match_{bm_patch_path.stem}.tif"
        out_path = dmsp_out_dir / out_name
        out_profile = bm_profile.copy()
        out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(best_vis_patch.astype(np.float32), 1)
        saved_paths.append(out_path)
    return saved_paths


def collect_dmsp_keys(s3, bucket_name: str, dmsp_date_str: str) -> List[str]:
    satellites = [f"F{n}" for n in range(10, 19)]
    file_keys: List[str] = []
    for sat in satellites:
        prefix = f"{sat}{dmsp_date_str[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if dmsp_date_str in key and key.endswith(".vis.co.tif"):
                    file_keys.append(key)
    return file_keys


def match_dmsp_to_bm_patch(
    bm_patch_path: Path,
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
) -> None:
    bm_name = bm_patch_path.name
    match = re.match(r"BM_patch_(\d{4}-\d{2}-\d{2})_", bm_name)
    if not match:
        print(f"Unable to parse date from BM patch name: {bm_name}")
        return
    bm_date = match.group(1)
    dmsp_date = bm_date.replace("-", "")
    file_keys = collect_dmsp_keys(s3, bucket_name, dmsp_date)
    if not file_keys:
        print(f"No DMSP granules found for {bm_date}")
        return

    temp_dmsp_dir = temp_dir / "dmsp"
    temp_dmsp_dir.mkdir(parents=True, exist_ok=True)
    saved = process_bm_patch_for_best_fnumber(
        bm_patch_path=bm_patch_path,
        file_keys=file_keys,
        s3=s3,
        bucket_name=bucket_name,
        temp_dir=temp_dmsp_dir,
        dmsp_out_dir=dmsp_out_dir,
    )
    for file in temp_dmsp_dir.iterdir():
        try:
            file.unlink()
        except OSError:
            pass
    if not saved:
        print(f"No suitable DMSP patches saved for {bm_name}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_pipeline(
    samples: Sequence[Sample],
    collection_id: str,
    nasa_token: str,
    patch_size_pix: int,
    tile_shapefile_path: Path,
    bm_output: Path,
    dmsp_output: Path,
    temp_root: Path,
) -> None:
    if not nasa_token:
        raise ValueError("NASA token is required. Provide via --nasa-token or NASA_TOKEN env variable.")

    tile_shapefile = gpd.read_file(tile_shapefile_path)
    bm_output = ensure_directory(bm_output)
    dmsp_output = ensure_directory(dmsp_output)
    temp_bm_dir = ensure_directory(temp_root / "bm")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"

    for sample in tqdm(samples, desc="Samples processed", unit="sample"):
        bm_path = fetch_black_marble_patch(
            sample=sample,
            patch_size_pix=patch_size_pix,
            collection_id=collection_id,
            nasa_token=nasa_token,
            tile_shapefile=tile_shapefile,
            output_folder=bm_output,
            temp_dir=temp_bm_dir,
        )
        if bm_path is None:
            continue
        match_dmsp_to_bm_patch(
            bm_patch_path=bm_path,
            s3=s3,
            bucket_name=bucket_name,
            temp_dir=temp_root,
            dmsp_out_dir=dmsp_output,
        )

    shutil.rmtree(temp_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download paired Black Marble and DMSP GeoTIFF patches.")
    parser.add_argument("samples", type=Path, help="Path to CSV or JSON file with Longitude, Latitude, date columns.")
    parser.add_argument(
        "--nasa-token",
        dest="nasa_token",
        default=os.getenv("NASA_TOKEN"),
        help="NASA Earthdata token. Defaults to NASA_TOKEN environment variable.",
    )
    parser.add_argument(
        "--collection-id",
        default="C3365931269-LAADS",
        help="NASA CMR collection concept ID for Black Marble.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=1000,
        help="Patch size in pixels for both Black Marble and DMSP outputs.",
    )
    parser.add_argument(
        "--tile-shapefile",
        type=Path,
        default=Path("Data/Black_Marble_IDs/Black_Marble_World_tiles.shp"),
        help="Path to the Black Marble grid tile shapefile.",
    )
    parser.add_argument(
        "--bm-output",
        type=Path,
        default=Path("Raw_NL_Data/BM data"),
        help="Directory to write Black Marble GeoTIFF patches.",
    )
    parser.add_argument(
        "--dmsp-output",
        type=Path,
        default=Path("Raw_NL_Data/DMSP data"),
        help="Directory to write DMSP GeoTIFF patches.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temp_downloads"),
        help="Temporary working directory for intermediate downloads.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    samples = load_samples(args.samples)
    run_pipeline(
        samples=samples,
        collection_id=args.collection_id,
        nasa_token=args.nasa_token,
        patch_size_pix=args.patch_size,
        tile_shapefile_path=args.tile_shapefile,
        bm_output=args.bm_output,
        dmsp_output=args.dmsp_output,
        temp_root=args.temp_dir,
    )


if __name__ == "__main__":
    main()
