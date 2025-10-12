#!/usr/bin/env python3
"""Simplified downloader for Black Marble (BM) and matching DMSP GeoTIFFs.

The original notebook-exported script performed population sampling, data
augmentation, and PNG exports.  This rewrite keeps only the core
functionality required for downloading Black Marble night-time lights
patches and pairing each patch with the best-available DMSP scene.

Usage
-----
1. Create a CSV file listing the targets you would like to download.  The
   file must contain the columns ``longitude``, ``latitude`` and ``date``
   (``YYYY-MM-DD``).
2. Ensure the ``NASA_TOKEN`` environment variable is set (``.env`` files are
   respected if ``python-dotenv`` is installed).
3. Run ``python data_sampler.py --targets my_targets.csv``.

The script downloads 1000×1000 pixel BM patches and saves them under
``Raw_NL_Data/BM data``.  DMSP matches are downloaded from the public
NASA ``globalnightlight`` bucket, reprojected onto the BM grid, and stored
under ``Raw_NL_Data/DMSP data``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import botocore
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import requests
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

try:
    from dotenv import load_dotenv  # pragma: no cover - optional dependency
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

LOG = logging.getLogger(__name__)

PATCH_SIZE_PIX = 1000
BM_COLLECTION_ID = "C3365931269-LAADS"
BM_TILE_SHAPEFILE = Path("Data/Black_Marble_IDs/Black_Marble_World_tiles.shp")
BM_OUTPUT_DIR = Path("Raw_NL_Data/BM data")
DMSP_OUTPUT_DIR = Path("Raw_NL_Data/DMSP data")
TEMP_DOWNLOAD_DIR = Path("temp_dl")
TEMP_DMSP_DIR = Path("DMSP_Raw_Temp")
DMSP_BUCKET = "globalnightlight"
DMSP_SATELLITES = [f"F{n}" for n in range(10, 19)]
CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


@dataclass
class Target:
    """Download request for a BM patch."""

    longitude: float
    latitude: float
    date: datetime

    @property
    def date_string(self) -> str:
        return self.date.strftime("%Y-%m-%d")


class DownloadError(RuntimeError):
    """Raised when a download cannot be completed."""


def load_targets(path: Path) -> List[Target]:
    """Load download targets from a CSV file."""

    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")

    with path.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames:
            raise ValueError("Target CSV must contain a header row")
        field_map = {name.lower(): name for name in reader.fieldnames}
        required = {"longitude", "latitude", "date"}
        missing = required - set(field_map)
        if missing:
            raise ValueError(f"Target CSV missing required columns: {sorted(missing)}")

        targets: List[Target] = []
        for row in reader:
            try:
                lon = float(row[field_map["longitude"]])
                lat = float(row[field_map["latitude"]])
                date = datetime.strptime(row[field_map["date"]].strip(), "%Y-%m-%d")
            except Exception as exc:  # noqa: BLE001 - provide context
                raise ValueError(f"Invalid row in {path}: {row}") from exc
            targets.append(Target(lon, lat, date))

    LOG.info("Loaded %d targets from %s", len(targets), path)
    return targets


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    """Bounding box (min_lon, min_lat, max_lon, max_lat) for a patch."""

    half_deg = (patch_size_pix * tif_res_deg) / 2.0
    min_lat = max(lat - half_deg, -60.0)  # Avoid Antarctica requests
    max_lat = max(lat + half_deg, -60.0)
    return [lon - half_deg, min_lat, lon + half_deg, max_lat]


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    """Find candidate Black Marble granules that cover the bounding box."""

    bbox_str = ",".join(f"{coord:.6f}" for coord in bbox)
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": bbox_str,
        "page_size": 50,
    }
    response = requests.get(CMR_SEARCH_URL, params=params, timeout=60)
    response.raise_for_status()
    h5_links: List[str] = []
    granules = response.json().get("feed", {}).get("entry", [])
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                h5_links.append(href)
    return h5_links


def download_file(url: str, token: str, dest: Path) -> None:
    """Download a file using the NASA token for authentication."""

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=120)
    if response.status_code == 401:
        raise DownloadError("NASA token rejected; ensure it is valid")
    response.raise_for_status()
    dest.write_bytes(response.content)


def convert_h5_to_geotiff(h5_path: Path, tile_shapes: gpd.GeoDataFrame, temp_dir: Path) -> Path:
    """Convert the Gap-Filled BRDF-corrected band to a GeoTIFF."""

    tif_path = temp_dir / (h5_path.stem + ".tif")
    with h5py.File(h5_path, "r") as h5_file:
        dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if dataset_path not in h5_file:
            raise DownloadError(f"Dataset not found in {h5_path.name}")
        ntl_data = h5_file[dataset_path][...]
    match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not match:
        raise DownloadError(f"Could not determine tile id from {h5_path.name}")
    tile_id = match.group(0)
    bounds_row = tile_shapes.loc[tile_shapes["TileID"] == tile_id]
    if bounds_row.empty:
        raise DownloadError(f"Tile ID {tile_id} not found in shapefile")
    left, bottom, right, top = bounds_row.total_bounds
    profile = {
        "driver": "GTiff",
        "height": ntl_data.shape[0],
        "width": ntl_data.shape[1],
        "count": 1,
        "dtype": ntl_data.dtype,
        "crs": "EPSG:4326",
        "transform": rio_from_bounds(left, bottom, right, top, ntl_data.shape[1], ntl_data.shape[0]),
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(ntl_data, 1)
    return tif_path


def extract_patch_from_geotiff(tif_path: Path, lon: float, lat: float, patch_size_pix: int) -> Tuple[np.ndarray, Dict]:
    """Clip a BM patch centered on ``lon, lat``."""

    with rasterio.open(tif_path) as src:
        bbox = get_patch_bbox(lon, lat, patch_size_pix, tif_res_deg=src.transform[0])
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window, boundless=True, fill_value=0)
        patch_meta = src.meta.copy()
        patch_meta.update({
            "height": patch.shape[0],
            "width": patch.shape[1],
            "transform": src.window_transform(window),
        })
    return patch, patch_meta


def enforce_patch_size(patch: np.ndarray, patch_size_pix: int) -> np.ndarray:
    """Ensure the patch is exactly ``patch_size_pix`` square by padding or cropping."""

    patch = patch[:patch_size_pix, :patch_size_pix]
    pad_y = max(0, patch_size_pix - patch.shape[0])
    pad_x = max(0, patch_size_pix - patch.shape[1])
    if pad_x or pad_y:
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), mode="constant", constant_values=0)
    return patch


def process_single_sample(
    sample: Target,
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapes: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
) -> Optional[Path]:
    """Download and extract a BM patch for a single target."""

    lon, lat, date_str = sample.longitude, sample.latitude, sample.date_string
    bbox = get_patch_bbox(lon, lat, patch_size_pix, tif_res_deg=0.004)
    try:
        urls = search_nasa_cmr(collection_id, date_str, bbox)
    except Exception as exc:  # noqa: BLE001
        LOG.error("CMR search failed for %s (%s, %s): %s", date_str, lon, lat, exc)
        return None
    if not urls:
        LOG.warning("No Black Marble granule found for %s at (%s, %s)", date_str, lon, lat)
        return None

    h5_url = urls[0]
    temp_dir.mkdir(parents=True, exist_ok=True)
    h5_path = temp_dir / Path(h5_url).name
    try:
        if not h5_path.exists():
            LOG.info("Downloading BM granule %s", h5_path.name)
            download_file(h5_url, token, h5_path)
        tif_path = convert_h5_to_geotiff(h5_path, tile_shapes, temp_dir)
        patch, patch_meta = extract_patch_from_geotiff(tif_path, lon, lat, patch_size_pix)
        patch = enforce_patch_size(patch, patch_size_pix)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Failed to build patch for %s (%s, %s): %s", date_str, lon, lat, exc)
        return None
    finally:
        if h5_path.exists():
            h5_path.unlink(missing_ok=True)
        tif_candidate = temp_dir / f"{h5_path.stem}.tif"
        if tif_candidate.exists():
            tif_candidate.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"BM_patch_{date_str}_{lon:.4f}_{lat:.4f}.tif"
    with rasterio.open(out_path, "w", **patch_meta) as dst:
        dst.write(patch, 1)
    LOG.info("Saved BM patch %s", out_path)
    return out_path


def process_samples_parallel(
    samples: Sequence[Target],
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapes: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
    max_workers: int = 4,
) -> List[Path]:
    """Download BM patches in parallel."""

    from concurrent.futures import ThreadPoolExecutor

    results: List[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_sample,
                sample,
                patch_size_pix,
                collection_id,
                token,
                tile_shapes,
                output_dir,
                temp_dir,
            )
            for sample in samples
        ]
        for future in futures:
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def wait_for_file_release(path: Path, timeout: int = 10) -> None:
    """Wait until a downloaded file can be opened (Windows safety)."""

    import time

    start = time.time()
    while True:
        try:
            with path.open("rb"):
                return
        except PermissionError:
            if time.time() - start > timeout:
                raise
            time.sleep(0.5)


def safe_download(s3, bucket: str, key: str, outpath: Path) -> bool:
    """Download an S3 object with retries."""

    import time

    for attempt in range(5):
        try:
            s3.download_file(bucket, key, str(outpath))
            wait_for_file_release(outpath)
            return True
        except botocore.exceptions.EndpointConnectionError as exc:
            LOG.warning("Endpoint connection error on %s (attempt %s): %s", key, attempt + 1, exc)
        except botocore.exceptions.ClientError as exc:
            LOG.warning("Client error on %s (attempt %s): %s", key, attempt + 1, exc)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Unexpected error on %s (attempt %s): %s", key, attempt + 1, exc)
        time.sleep(2)
    LOG.error("Failed to download %s after multiple attempts", key)
    return False


def group_by_f_number(file_keys: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in file_keys:
        base = Path(key).name
        f_number = base.split("_")[0] if "_" in base else base[:3]
        groups[f_number].append(key)
    return groups


def reproject_to_bm_grid(src_path: Path, bm_profile: Dict) -> np.ndarray:
    """Reproject a DMSP scene to match the BM patch grid."""

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


def process_bm_patch_for_best_dmsp(
    bm_patch_path: Path,
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
) -> List[Path]:
    bm_patch_name = bm_patch_path.name
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    bm_date = bm_patch_name.split("_")[2]
    dmsp_date_str = bm_date.replace("-", "")

    file_keys: List[str] = []
    for sat in DMSP_SATELLITES:
        prefix = f"{sat}{dmsp_date_str[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if dmsp_date_str in key and key.endswith(".vis.co.tif"):
                    file_keys.append(key)

    saved_paths: List[Path] = []
    grouped = group_by_f_number(file_keys)
    for f_number, keys in grouped.items():
        best_valid = 0
        best_patch: Optional[np.ndarray] = None
        best_key: Optional[str] = None
        for key in keys:
            local_path = temp_dir / Path(key).name
            if not local_path.exists():
                temp_dir.mkdir(parents=True, exist_ok=True)
                if not safe_download(s3, bucket_name, key, local_path):
                    continue
            try:
                vis_patch = reproject_to_bm_grid(local_path, bm_profile)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to reproject %s: %s", key, exc)
                continue
            valid_pixels = int(np.sum(~np.isnan(vis_patch)))
            if valid_pixels == 0:
                continue
            median_val = float(np.nanmedian(vis_patch)) if valid_pixels else 0.0
            if valid_pixels > best_valid and median_val > 1:
                best_valid = valid_pixels
                best_patch = vis_patch
                best_key = key
        if best_patch is None:
            LOG.info("No suitable DMSP scene found for %s (%s)", bm_patch_name, f_number)
            continue
        valid_fraction = best_valid / float(bm_shape[0] * bm_shape[1])
        if valid_fraction < 0.10:
            LOG.info(
                "Skipping %s for %s: only %.2f%% valid pixels",
                f_number,
                bm_patch_name,
                valid_fraction * 100,
            )
            continue
        dmsp_out_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(best_key).stem
        out_name = f"{f_number}_{base_name}_match_{bm_patch_name}"
        out_path = dmsp_out_dir / out_name
        out_profile = bm_profile.copy()
        out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(best_patch.astype(np.float32), 1)
        saved_paths.append(out_path)
        LOG.info("Saved DMSP match %s", out_path)

    for file in temp_dir.glob("*.tif"):
        try:
            file.unlink()
        except OSError:
            pass

    return saved_paths


def match_dmsp_to_bm(bm_patch_paths: Sequence[Path], dmsp_out_dir: Path, temp_dir: Path, max_workers: int = 4) -> List[Path]:
    """Match each BM patch with the best DMSP scene."""

    if not bm_patch_paths:
        return []

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    from concurrent.futures import ThreadPoolExecutor

    results: List[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_bm_patch_for_best_dmsp,
                Path(path),
                s3,
                DMSP_BUCKET,
                temp_dir,
                dmsp_out_dir,
            )
            for path in bm_patch_paths
        ]
        for future in futures:
            matches = future.result()
            results.extend(matches)
    return results


def run_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = os.getenv("NASA_TOKEN")
    if not token:
        raise RuntimeError(
            "NASA_TOKEN environment variable is not set. "
            "Create a .env file (see .env.example) with your NASA Earthdata token before running downloads."
        )

    targets = load_targets(Path(args.targets))
    tile_shapes = gpd.read_file(BM_TILE_SHAPEFILE)

    LOG.info("Downloading Black Marble patches (patch size: %d px)", args.patch_size)
    bm_paths = process_samples_parallel(
        samples=targets,
        patch_size_pix=args.patch_size,
        collection_id=BM_COLLECTION_ID,
        token=token,
        tile_shapes=tile_shapes,
        output_dir=Path(args.bm_output),
        temp_dir=Path(args.temp_dir),
        max_workers=args.workers,
    )

    LOG.info("Saved %d Black Marble patches", len(bm_paths))
    LOG.info("Matching DMSP scenes")
    dmsp_matches = match_dmsp_to_bm(
        bm_patch_paths=bm_paths,
        dmsp_out_dir=Path(args.dmsp_output),
        temp_dir=Path(args.dmsp_temp_dir),
        max_workers=args.workers,
    )
    LOG.info("Saved %d DMSP patches", len(dmsp_matches))

    if args.cleanup:
        shutil.rmtree(args.temp_dir, ignore_errors=True)
        shutil.rmtree(args.dmsp_temp_dir, ignore_errors=True)
        LOG.info("Cleaned up temporary directories")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="targets.csv", help="CSV file with longitude, latitude, date columns")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_PIX, help="Patch size in pixels (default: 1000)")
    parser.add_argument("--bm-output", default=str(BM_OUTPUT_DIR), help="Directory to store BM GeoTIFF patches")
    parser.add_argument("--dmsp-output", default=str(DMSP_OUTPUT_DIR), help="Directory to store matched DMSP GeoTIFF patches")
    parser.add_argument("--temp-dir", default=str(TEMP_DOWNLOAD_DIR), help="Temporary directory for BM downloads")
    parser.add_argument(
        "--dmsp-temp-dir",
        default=str(TEMP_DMSP_DIR),
        help="Temporary directory for raw DMSP downloads",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--cleanup", action="store_true", help="Remove temporary folders after completion")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    run_pipeline(parser.parse_args())
