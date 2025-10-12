"""Simplified downloader for Black Marble (BM) and matching DMSP GeoTIFF patches.

This script replaces the previous notebook-exported workflow with a small CLI that:

1. Reads a CSV file containing target download points (longitude, latitude, date).
2. Downloads the matching Black Marble granules, converts them to GeoTIFF, and
   extracts 1000x1000 px patches centred on each target.
3. Searches the public DMSP archive for scenes on the same date and reprojects the
   best-per-satellite patch onto the BM grid.

The script keeps the previous directory structure (`Raw_NL_Data/BM data` and
`Raw_NL_Data/DMSP data`) so downstream tooling can remain unchanged.

Example usage:

    python data_sampler.py targets.csv

The CSV must expose the columns: `longitude`, `latitude`, and `date` (ISO format).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import boto3
import h5py
import numpy as np
import rasterio
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError
from geopandas import read_file as gpd_read_file
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

PATCH_SIZE = 1000
BM_COLLECTION_ID = "C3365931269-LAADS"
BM_DATASET_PATH = (
    "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
)
DEFAULT_TILE_SHP = "Data/Black_Marble_IDs/Black_Marble_World_tiles.shp"
BM_OUTPUT_DIR = Path("Raw_NL_Data/BM data")
DMSP_OUTPUT_DIR = Path("Raw_NL_Data/DMSP data")
DMSP_BUCKET = "globalnightlight"
DMSP_SATELLITES = tuple(f"F{idx}" for idx in range(10, 19))
MIN_VALID_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Target parsing utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadTarget:
    """Represents a single BM/DMSP download request."""

    longitude: float
    latitude: float
    date: datetime

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%m-%d")

    @property
    def dmsp_date_str(self) -> str:
        return self.date.strftime("%Y%m%d")

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "DownloadTarget":
        try:
            lon = float(row["longitude"])
            lat = float(row["latitude"])
            date = datetime.fromisoformat(row["date"]).date()
        except KeyError as exc:
            raise ValueError(f"Missing required column: {exc}") from exc
        except ValueError as exc:  # invalid float/date
            raise ValueError(f"Invalid row values: {row}") from exc
        return cls(longitude=lon, latitude=lat, date=datetime.combine(date, datetime.min.time()))


def load_targets_csv(path: Path) -> List[DownloadTarget]:
    logging.info("Loading targets from %s", path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        targets = [DownloadTarget.from_row(row) for row in reader]
    if not targets:
        raise ValueError("No targets found in CSV – provide at least one row")
    logging.info("Loaded %d targets", len(targets))
    return targets


# ---------------------------------------------------------------------------
# Black Marble helpers
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    BM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DMSP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clamp_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    min_lon = max(-180.0, min(180.0, min_lon))
    max_lon = max(-180.0, min(180.0, max_lon))
    min_lat = max(-90.0, min(90.0, min_lat))
    max_lat = max(-90.0, min(90.0, max_lat))
    return (min_lon, min_lat, max_lon, max_lat)


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": ",".join(f"{coord:.6f}" for coord in clamp_bbox(bbox)),
        "page_size": 50,
    }
    try:
        response = requests.get(
            "https://cmr.earthdata.nasa.gov/search/granules.json",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        entries = response.json().get("feed", {}).get("entry", [])
    except requests.RequestException as exc:
        logging.error("CMR query failed for %s: %s", date_str, exc)
        return []

    links: List[str] = []
    for granule in entries:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                links.append(href)
    return links


def download_h5(url: str, token: str, dest: Path) -> bool:
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
        return True
    except requests.RequestException as exc:
        logging.error("Failed to download %s: %s", url, exc)
        return False


def build_tile_bounds(tile_shapefile: Path) -> Dict[str, Tuple[float, float, float, float]]:
    logging.info("Loading tile shapefile: %s", tile_shapefile)
    gdf = gpd_read_file(tile_shapefile)
    if "TileID" not in gdf.columns:
        raise ValueError("Tile shapefile must include a 'TileID' column")
    bounds = {str(row.TileID): tuple(row.geometry.bounds) for _, row in gdf.iterrows()}
    logging.info("Loaded %d tile bounds", len(bounds))
    return bounds


def convert_h5_to_geotiff(h5_path: Path, tile_bounds: Dict[str, Tuple[float, float, float, float]]) -> Path:
    tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not tile_match:
        raise ValueError(f"Could not determine tile id from {h5_path.name}")
    tile_id = tile_match.group()
    if tile_id not in tile_bounds:
        raise ValueError(f"Tile {tile_id} not found in shapefile bounds")

    with h5py.File(h5_path, "r") as handle:
        if BM_DATASET_PATH not in handle:
            raise ValueError(f"Dataset {BM_DATASET_PATH} missing in {h5_path}")
        data = handle[BM_DATASET_PATH][...]

    left, bottom, right, top = tile_bounds[tile_id]
    tif_path = h5_path.with_suffix(".tif")
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "crs": "EPSG:4326",
        "transform": rio_from_bounds(left, bottom, right, top, data.shape[1], data.shape[0]),
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
    return tif_path


def extract_patch(tif_path: Path, lon: float, lat: float, patch_size: int) -> Tuple[np.ndarray, Dict[str, object]]:
    with rasterio.open(tif_path) as src:
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        pixel_size_deg = (res_x + res_y) / 2.0
        half_deg = (patch_size * pixel_size_deg) / 2.0
        bbox = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window, boundless=True, fill_value=src.nodata or 0)
        patch_meta = src.profile.copy()
        patch_meta.update({
            "height": patch_size,
            "width": patch_size,
            "transform": src.window_transform(window),
        })

    patch = enforce_patch_size(patch, patch_size)
    return patch, patch_meta


def enforce_patch_size(patch: np.ndarray, patch_size: int) -> np.ndarray:
    patch = patch[:patch_size, :patch_size]
    pad_h = max(0, patch_size - patch.shape[0])
    pad_w = max(0, patch_size - patch.shape[1])
    if pad_h or pad_w:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return patch


def save_patch(patch: np.ndarray, meta: Dict[str, object], out_path: Path) -> None:
    meta = meta.copy()
    meta.update({"height": patch.shape[0], "width": patch.shape[1], "count": 1})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(patch, 1)


def download_black_marble(targets: Sequence[DownloadTarget], token: str, tile_bounds: Dict[str, Tuple[float, float, float, float]], patch_size: int = PATCH_SIZE) -> List[Path]:
    saved_paths: List[Path] = []
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for target in targets:
            bbox = (target.longitude - 1.0, target.latitude - 1.0, target.longitude + 1.0, target.latitude + 1.0)
            urls = search_nasa_cmr(BM_COLLECTION_ID, target.date_str, bbox)
            if not urls:
                logging.warning("No Black Marble granule found for %s (%s, %s)", target.date_str, target.longitude, target.latitude)
                continue

            url = urls[0]
            h5_path = tmp_dir / os.path.basename(url)
            if not download_h5(url, token, h5_path):
                continue

            try:
                tif_path = convert_h5_to_geotiff(h5_path, tile_bounds)
                patch, meta = extract_patch(tif_path, target.longitude, target.latitude, patch_size)
            except Exception as exc:
                logging.error("Failed to convert/extract BM patch for %s: %s", target.date_str, exc)
                continue

            out_name = f"BM_patch_{target.date_str}_{target.latitude:.6f}_{target.longitude:.6f}.tif"
            out_path = BM_OUTPUT_DIR / out_name
            save_patch(patch, meta, out_path)
            saved_paths.append(out_path)
            logging.info("Saved Black Marble patch -> %s", out_path)
    return saved_paths


# ---------------------------------------------------------------------------
# DMSP helpers
# ---------------------------------------------------------------------------


def list_dmsp_scene_keys(s3_client, date_str: str) -> List[str]:
    keys: List[str] = []
    for sat in DMSP_SATELLITES:
        prefix = f"{sat}{date_str[:4]}/"
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=DMSP_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if date_str in key and key.endswith(".vis.co.tif"):
                    keys.append(key)
    return keys


def safe_download(s3_client, key: str, dest: Path, max_retries: int = 5) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            s3_client.download_file(DMSP_BUCKET, key, str(dest))
            return True
        except (EndpointConnectionError, ClientError, BotoCoreError) as exc:
            logging.warning(
                "Download failed for %s (attempt %d/%d): %s", key, attempt, max_retries, exc
            )
            time.sleep(2)
    logging.error("Failed to download %s after %d attempts", key, max_retries)
    return False


def group_by_satellite(keys: Iterable[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for key in keys:
        base = os.path.basename(key)
        sat = base.split("_")[0] if "_" in base else base[:3]
        grouped.setdefault(sat, []).append(key)
    return grouped


def reproject_dmsp_scene(scene_path: Path, bm_profile: Dict[str, object]) -> np.ndarray:
    with rasterio.open(scene_path) as src:
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


def match_dmsp_to_bm_patch(bm_patch: Path, s3_client, temp_dir: Path) -> List[Path]:
    with rasterio.open(bm_patch) as src:
        bm_profile = src.profile.copy()
        bm_shape = (src.height, src.width)

    keys = list_dmsp_scene_keys(s3_client, Path(bm_patch).stem.split("_")[2].replace("-", ""))
    if not keys:
        logging.warning("No DMSP scenes found for %s", bm_patch.name)
        return []

    outputs: List[Path] = []
    grouped = group_by_satellite(keys)
    total_pixels = bm_shape[0] * bm_shape[1]

    for sat, sat_keys in grouped.items():
        best_patch = None
        best_key = None
        best_valid_pixels = 0

        for key in sat_keys:
            local_path = temp_dir / os.path.basename(key)
            if not safe_download(s3_client, key, local_path):
                continue
            try:
                candidate = reproject_dmsp_scene(local_path, bm_profile)
            except Exception as exc:
                logging.error("Failed to reproject %s: %s", key, exc)
                local_path.unlink(missing_ok=True)
                continue
            finally:
                local_path.unlink(missing_ok=True)

            valid_pixels = np.sum(~np.isnan(candidate))
            median_val = np.nanmedian(candidate) if valid_pixels else 0.0
            if valid_pixels > best_valid_pixels and median_val > 1:
                best_valid_pixels = valid_pixels
                best_patch = candidate
                best_key = key

        if best_patch is None:
            logging.info("No suitable DMSP patch for %s (%s)", sat, bm_patch.name)
            continue

        if best_valid_pixels / total_pixels < MIN_VALID_FRACTION:
            logging.info(
                "Skipping %s for %s: only %.2f%% valid pixels",
                sat,
                bm_patch.name,
                100 * best_valid_pixels / total_pixels,
            )
            continue

        out_name = f"{sat}_{Path(best_key).stem}_match_{bm_patch.stem}.tif"
        out_path = DMSP_OUTPUT_DIR / out_name
        out_profile = bm_profile.copy()
        out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(best_patch.astype(np.float32), 1)
        outputs.append(out_path)
        logging.info(
            "Saved DMSP match -> %s (valid pixels: %.1f%%)",
            out_path,
            100 * best_valid_pixels / total_pixels,
        )

    return outputs


def match_dmsp_for_bm_patches(bm_patches: Sequence[Path]) -> List[Path]:
    if not bm_patches:
        logging.info("No BM patches to match with DMSP")
        return []

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    saved: List[Path] = []
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        temp_dir = Path(tmp_dir_str)
        for bm_patch in bm_patches:
            saved.extend(match_dmsp_to_bm_patch(bm_patch, s3_client, temp_dir))
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BM + DMSP patches for given targets")
    parser.add_argument("targets_csv", type=Path, help="CSV containing longitude, latitude, date")
    parser.add_argument(
        "--tile-shapefile",
        type=Path,
        default=Path(DEFAULT_TILE_SHP),
        help="Path to the Black Marble tile shapefile",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help="Patch size in pixels (defaults to 1000)",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    ensure_directories()

    nasa_token = os.getenv("NASA_TOKEN")
    if not nasa_token:
        raise RuntimeError("NASA_TOKEN environment variable is not set")

    targets = load_targets_csv(args.targets_csv)
    tile_bounds = build_tile_bounds(args.tile_shapefile)

    bm_patches = download_black_marble(targets, nasa_token, tile_bounds, patch_size=args.patch_size)
    logging.info("Downloaded %d BM patches", len(bm_patches))

    dmsp_matches = match_dmsp_for_bm_patches(bm_patches)
    logging.info("Saved %d DMSP patches", len(dmsp_matches))


if __name__ == "__main__":
    main()
