# -*- coding: utf-8 -*-
"""Simplified downloader for Black Marble and matching DMSP GeoTIFF patches.

This module replaces the original notebook-exported workflow with a focused
pipeline that:

1. Loads a user-supplied table of targets (longitude, latitude, acquisition
   date).
2. Downloads a 1000×1000 pixel Black Marble (VIIRS DNB) patch for each target.
3. Searches the DMSP archive for matching scenes and reprojects the best match
   onto the Black Marble grid.

The entry point is ``main`` which exposes a small CLI for orchestrating the
workflow.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import botocore
import geopandas as gpd
import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.windows import from_bounds
from rasterio.warp import Resampling, reproject
import requests

try:  # Optional dependency for local development convenience
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - best effort only
    load_dotenv = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATCH_SIZE_PIX = 1000
BM_COLLECTION_ID = "C3365931269-LAADS"
BM_TILE_SHAPEFILE = "Data/Black_Marble_IDs/Black_Marble_World_tiles.shp"
BM_OUTPUT_DIR = os.path.join("Raw_NL_Data", "BM data")
DMSP_OUTPUT_DIR = os.path.join("Raw_NL_Data", "DMSP data")
DEFAULT_TEMP_DIR = os.path.join("temp_dl")
NASA_TOKEN_ENV = "NASA_TOKEN"
DMSP_BUCKET = "globalnightlight"
DMSP_SATELLITES = [f"F{n}" for n in range(10, 19)]
BM_RESOLUTION_DEG = 0.004  # Approximate VIIRS DNB resolution in degrees/pixel


# ---------------------------------------------------------------------------
# Utility data structures
# ---------------------------------------------------------------------------
@dataclass
class Target:
    """Simple container for a download target."""

    longitude: float
    latitude: float
    date: datetime

    @property
    def iso_date(self) -> str:
        return self.date.strftime("%Y-%m-%d")

    @property
    def dmsp_date(self) -> str:
        return self.date.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# NASA token handling
# ---------------------------------------------------------------------------
def load_nasa_token() -> str:
    """Load the NASA token from the environment or raise a helpful error."""

    if load_dotenv is not None:
        load_dotenv()
    token = os.getenv(NASA_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            "NASA_TOKEN environment variable is not set. "
            "Create a .env file (see .env.example) with your NASA Earthdata token "
            "before running downloads."
        )
    return token


# ---------------------------------------------------------------------------
# Target table helpers
# ---------------------------------------------------------------------------
def _parse_date(value: str) -> datetime:
    """Parse a date string in either YYYY-MM-DD or YYYYMMDD format."""

    value = value.strip()
    if re.fullmatch(r"\d{8}", value):
        return datetime.strptime(value, "%Y%m%d")
    return datetime.strptime(value, "%Y-%m-%d")


def load_targets(path: str) -> List[Target]:
    """Load download targets from a CSV file.

    The CSV must contain the columns ``Longitude``, ``Latitude`` and ``date``.
    """

    targets: List[Target] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {col for col in ("Longitude", "Latitude", "date") if col not in reader.fieldnames}
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                lon = float(row["Longitude"])
                lat = float(row["Latitude"])
                date = _parse_date(row["date"])
            except Exception as exc:  # pragma: no cover - defensive casting
                raise ValueError(f"Could not parse row {row}: {exc}") from exc
            targets.append(Target(longitude=lon, latitude=lat, date=date))
    return targets


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def clamp_lat(lat: float) -> float:
    """Clamp latitude to the valid range supported by the tiles."""

    return max(min(lat, 89.999), -89.999)


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    """Create a bounding box centered at ``lon``, ``lat`` for a square patch."""

    lat = clamp_lat(lat)
    half_deg = (patch_size_pix * tif_res_deg) / 2.0
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def enforce_patch_size(patch: np.ndarray, patch_size_pix: int) -> np.ndarray:
    """Ensure the patch is exactly ``patch_size_pix`` square by clipping/padding."""

    patch = patch[:patch_size_pix, :patch_size_pix]
    pad_h = max(0, patch_size_pix - patch.shape[0])
    pad_w = max(0, patch_size_pix - patch.shape[1])
    if pad_h or pad_w:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return patch


# ---------------------------------------------------------------------------
# Black Marble download helpers
# ---------------------------------------------------------------------------
def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    """Query the CMR API for Black Marble HDF5 download URLs."""

    cmr_search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    bbox_str = ",".join(str(x) for x in bbox)
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": bbox_str,
        "page_size": 50,
    }
    response = requests.get(cmr_search_url, params=params, timeout=60)
    response.raise_for_status()
    entries = response.json().get("feed", {}).get("entry", [])
    h5_links: List[str] = []
    for granule in entries:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                h5_links.append(href)
    return h5_links


def extract_patch_from_geotiff(tif_path: str, lon: float, lat: float, patch_size_pix: int) -> Tuple[np.ndarray, Dict]:
    """Extract a square patch from a GeoTIFF centred on ``lon``, ``lat``."""

    with rasterio.open(tif_path) as src:
        tif_res_deg = abs(src.transform.a)
        bbox = get_patch_bbox(lon, lat, patch_size_pix, tif_res_deg)
        window = from_bounds(*bbox, transform=src.transform)
        patch = src.read(1, window=window)
        patch = enforce_patch_size(patch, patch_size_pix)
        patch_transform = src.window_transform(window)
        patch_meta = src.meta.copy()
        patch_meta.update({
            "height": patch_size_pix,
            "width": patch_size_pix,
            "transform": patch_transform,
        })
    return patch, patch_meta


def remove_with_retry(path: str, retries: int = 5, delay: float = 1.0) -> None:
    """Try to remove ``path`` with a few retries (handles Windows file locking)."""

    for _ in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except Exception:
            import time

            time.sleep(delay)


def convert_h5_to_patch(
    h5_path: str,
    lon: float,
    lat: float,
    patch_size_pix: int,
    tile_bounds: Dict[str, Tuple[float, float, float, float]],
    output_path: str,
) -> Optional[str]:
    """Convert an HDF5 granule to a GeoTIFF patch and save it to ``output_path``."""

    tif_path = h5_path.replace(".h5", ".tif")
    try:
        with h5py.File(h5_path, "r") as f:
            ntl_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
            if ntl_path not in f:
                return None
            ntl_data = f[ntl_path][...]
            match = re.search(r"h\d{2}v\d{2}", os.path.basename(h5_path))
            if not match:
                return None
            tile_id = match.group()
            if tile_id not in tile_bounds:
                return None
            left, bottom, right, top = tile_bounds[tile_id]
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=ntl_data.shape[0],
            width=ntl_data.shape[1],
            count=1,
            dtype=ntl_data.dtype,
            crs="EPSG:4326",
            transform=rio_from_bounds(left, bottom, right, top, ntl_data.shape[1], ntl_data.shape[0]),
        ) as dst:
            dst.write(ntl_data, 1)
        patch, patch_meta = extract_patch_from_geotiff(tif_path, lon, lat, patch_size_pix)
        if patch.shape[0] < patch_size_pix or patch.shape[1] < patch_size_pix:
            return None
        with rasterio.open(output_path, "w", **patch_meta) as dst:
            dst.write(patch, 1)
        return output_path
    finally:
        remove_with_retry(tif_path)


def download_black_marble_patch(
    target: Target,
    collection_id: str,
    token: str,
    tile_bounds: Dict[str, Tuple[float, float, float, float]],
    output_folder: str,
    temp_folder: str,
    patch_size_pix: int = PATCH_SIZE_PIX,
) -> Optional[str]:
    """Download and extract a Black Marble patch for ``target``."""

    bbox = get_patch_bbox(target.longitude, target.latitude, patch_size_pix, BM_RESOLUTION_DEG)
    urls = search_nasa_cmr(collection_id, target.iso_date, bbox)
    if not urls:
        print(f"No Black Marble file found for {target.iso_date} at ({target.longitude:.3f}, {target.latitude:.3f})")
        return None

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    output_name = f"BM_patch_{target.iso_date}_{target.longitude:.4f}_{target.latitude:.4f}_{patch_size_pix}px.tif"
    output_path = os.path.join(output_folder, output_name)
    if os.path.exists(output_path):
        print(f"Skipping existing patch {output_path}")
        return output_path

    for url in urls:
        h5_path = os.path.join(temp_folder, os.path.basename(url))
        try:
            if not os.path.exists(h5_path):
                response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
                response.raise_for_status()
                with open(h5_path, "wb") as fh:
                    fh.write(response.content)
            patch = convert_h5_to_patch(
                h5_path,
                target.longitude,
                target.latitude,
                patch_size_pix,
                tile_bounds,
                output_path,
            )
            if patch:
                print(f"Saved Black Marble patch to {patch}")
                remove_with_retry(h5_path)
                return patch
            else:
                print(f"Failed to create patch from {h5_path}, trying next granule")
        except Exception as exc:
            print(f"Error downloading {url}: {exc}")
        finally:
            remove_with_retry(h5_path)
    print(f"Unable to produce Black Marble patch for {target.iso_date} at ({target.longitude}, {target.latitude})")
    return None


# ---------------------------------------------------------------------------
# DMSP helpers
# ---------------------------------------------------------------------------
def wait_for_file_release(path: str, timeout: float = 10.0) -> None:
    """Wait until ``path`` is readable (handles S3 eventual consistency)."""

    import time

    start = time.time()
    while True:
        try:
            with open(path, "rb"):
                return
        except PermissionError:
            if time.time() - start > timeout:
                raise
            time.sleep(0.5)


def safe_download(s3, bucket: str, key: str, outpath: str, max_retries: int = 5) -> bool:
    """Download ``key`` from ``bucket`` into ``outpath`` with retries."""

    import time

    for attempt in range(max_retries):
        try:
            s3.download_file(bucket, key, outpath)
            wait_for_file_release(outpath)
            return True
        except botocore.exceptions.EndpointConnectionError as exc:
            print(f"EndpointConnectionError on {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        except botocore.exceptions.ClientError as exc:
            print(f"ClientError on {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"Error downloading {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        time.sleep(2)
    print(f"Failed to download after {max_retries} attempts: {key}")
    return False


def reproject_to_bm_grid(src_path: str, bm_profile: Dict) -> np.ndarray:
    """Reproject a DMSP GeoTIFF to the grid defined by ``bm_profile``."""

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


def find_candidate_dmsp_keys(s3, date_str: str) -> List[str]:
    """Find DMSP GeoTIFF keys matching ``date_str`` (YYYYMMDD)."""

    keys: List[str] = []
    year = date_str[:4]
    paginator = s3.get_paginator("list_objects_v2")
    for satellite in DMSP_SATELLITES:
        prefix = f"{satellite}{year}/"
        for page in paginator.paginate(Bucket=DMSP_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if not key or not key.endswith(".vis.co.tif"):
                    continue
                if date_str in key:
                    keys.append(key)
    return keys


def select_best_dmsp_patch(
    bm_patch_path: str,
    candidate_keys: Sequence[str],
    s3,
    temp_dir: str,
    dmsp_out_dir: str,
    valid_fraction_threshold: float = 0.10,
) -> List[str]:
    """Download, reproject, and save the best DMSP match for each F-number."""

    if not candidate_keys:
        print(f"No DMSP candidates found for {bm_patch_path}")
        return []

    os.makedirs(dmsp_out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    grouped: Dict[str, List[str]] = {}
    for key in candidate_keys:
        base = os.path.basename(key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        grouped.setdefault(f_number, []).append(key)

    saved_paths: List[str] = []
    s3_client = s3
    for f_number, keys in grouped.items():
        best_valid_pixels = 0
        best_vis_patch: Optional[np.ndarray] = None
        best_source_name: Optional[str] = None
        for key in keys:
            download_path = os.path.join(temp_dir, os.path.basename(key))
            if not safe_download(s3_client, DMSP_BUCKET, key, download_path):
                remove_with_retry(download_path)
                continue
            try:
                vis_patch = reproject_to_bm_grid(download_path, bm_profile)
            except Exception as exc:
                print(f"Error reprojecting {key}: {exc}")
                remove_with_retry(download_path)
                continue
            valid_pixels = int(np.sum(~np.isnan(vis_patch)))
            median_val = float(np.nanmedian(vis_patch)) if valid_pixels else 0.0
            if valid_pixels > best_valid_pixels and median_val > 1:
                best_valid_pixels = valid_pixels
                best_vis_patch = vis_patch.copy()
                best_source_name = os.path.basename(key)
            remove_with_retry(download_path)

        if best_vis_patch is None or best_valid_pixels == 0:
            print(f"No valid DMSP patch for {f_number} matching {bm_patch_path}")
            continue
        valid_fraction = best_valid_pixels / float(bm_shape[0] * bm_shape[1])
        if valid_fraction < valid_fraction_threshold:
            print(
                f"Skipping {f_number} for {os.path.basename(bm_patch_path)}: "
                f"only {valid_fraction:.2%} valid pixels"
            )
            continue
        out_name = (
            f"{f_number}_{best_source_name.replace('.vis.co.tif', '')}_"
            f"match_{os.path.basename(bm_patch_path).replace('.tif', '')}.tif"
        )
        out_path = os.path.join(dmsp_out_dir, out_name)
        out_profile = bm_profile.copy()
        out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(best_vis_patch.astype(np.float32), 1)
        saved_paths.append(out_path)
        print(f"Saved DMSP match to {out_path}")
    return saved_paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def prepare_tile_bounds(shapefile_path: str) -> Dict[str, Tuple[float, float, float, float]]:
    """Load the Black Marble tile shapefile and extract per-tile bounds."""

    tiles = gpd.read_file(shapefile_path)
    if "TileID" not in tiles.columns:
        raise ValueError("Tile shapefile must contain a 'TileID' column")
    bounds: Dict[str, Tuple[float, float, float, float]] = {}
    for _, row in tiles.iterrows():
        bounds[row["TileID"]] = row.geometry.bounds
    return bounds


def process_targets(
    targets: Sequence[Target],
    tile_bounds: Dict[str, Tuple[float, float, float, float]],
    token: str,
    bm_output_dir: str,
    dmsp_output_dir: str,
    temp_dir: str,
) -> None:
    """Process each target sequentially, downloading BM and DMSP patches."""

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    for target in targets:
        print(
            f"\nProcessing target lon={target.longitude:.4f}, lat={target.latitude:.4f}, date={target.iso_date}"
        )
        bm_path = download_black_marble_patch(
            target,
            BM_COLLECTION_ID,
            token,
            tile_bounds,
            bm_output_dir,
            temp_dir,
            patch_size_pix=PATCH_SIZE_PIX,
        )
        if not bm_path:
            continue
        candidate_keys = find_candidate_dmsp_keys(s3, target.dmsp_date)
        select_best_dmsp_patch(bm_path, candidate_keys, s3, temp_dir, dmsp_output_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Black Marble and DMSP patches")
    parser.add_argument(
        "--targets",
        required=True,
        help="Path to CSV file containing Longitude, Latitude, date columns",
    )
    parser.add_argument(
        "--bm-output",
        default=BM_OUTPUT_DIR,
        help=f"Directory to store Black Marble patches (default: {BM_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dmsp-output",
        default=DMSP_OUTPUT_DIR,
        help=f"Directory to store matched DMSP patches (default: {DMSP_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--temp-dir",
        default=DEFAULT_TEMP_DIR,
        help=f"Temporary directory for intermediate downloads (default: {DEFAULT_TEMP_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    token = load_nasa_token()
    targets = load_targets(args.targets)
    if not targets:
        print("No targets found in CSV – nothing to do.")
        return
    tile_bounds = prepare_tile_bounds(BM_TILE_SHAPEFILE)
    os.makedirs(args.bm_output, exist_ok=True)
    os.makedirs(args.dmsp_output, exist_ok=True)
    temp_dir = args.temp_dir
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    try:
        process_targets(targets, tile_bounds, token, args.bm_output, args.dmsp_output, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
