"""Simplified downloader for Black Marble and matching DMSP GeoTIFF patches.

The original notebook export in this repository bundled population sampling,
plotting, PNG conversions, and many other responsibilities.  This module keeps
only the two stages that matter for the current workflow:

1. Download a set of Black Marble (VIIRS) night-light patches centred on the
   user supplied coordinates.
2. For each saved Black Marble patch, download the best matching DMSP
   GeoTIFF(s) and reproject them onto the same grid.

Usage
-----
Create a CSV file with at least three columns – ``Longitude``, ``Latitude`` and
``date`` (YYYY-MM-DD).  Then invoke this script:

```
python data_sampler.py --targets my_targets.csv \\
    --bm-output "Raw_NL_Data/BM data" --dmsp-output "Raw_NL_Data/DMSP data"
```

The NASA Earthdata token must be available through the ``NASA_TOKEN``
environment variable (``python-dotenv`` is honoured if installed so a ``.env``
file will also work).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore import exceptions as boto_exceptions
from botocore.config import Config
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import Resampling, reproject

try:  # Optional dependency – makes .env files convenient.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional import
    load_dotenv = None

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - required runtime dependency
    raise RuntimeError("geopandas must be installed to run the downloader") from exc

import h5py
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATCH_SIZE_PIXELS = 1000
VIIRS_PIXEL_DEGREES = 0.004  # approximate native resolution of the BM grid
BM_COLLECTION_ID = "C3365931269-LAADS"
BM_TILE_SHP = Path("Data/Black_Marble_IDs/Black_Marble_World_tiles.shp")
DEFAULT_BM_OUTPUT = Path("Raw_NL_Data/BM data")
DEFAULT_DMSP_OUTPUT = Path("Raw_NL_Data/DMSP data")
DEFAULT_BM_TEMP = Path("temp_dl")
DEFAULT_DMSP_TEMP = Path("DMSP_Raw_Temp")
DMSP_BUCKET = "globalnightlight"

# ---------------------------------------------------------------------------
# Data structures and helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    """Download target defined by lon/lat and observation date."""

    longitude: float
    latitude: float
    date: datetime

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%m-%d")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_targets_from_csv(csv_path: Path) -> List[Target]:
    targets: List[Target] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {col for col in ["Longitude", "Latitude", "date"] if col not in reader.fieldnames}
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for idx, row in enumerate(reader, start=1):
            try:
                lon = float(row["Longitude"])
                lat = float(row["Latitude"])
                date = datetime.strptime(row["date"], "%Y-%m-%d")
            except Exception as exc:  # pragma: no cover - user supplied data
                raise ValueError(f"Invalid row #{idx} in {csv_path}: {exc}") from exc
            targets.append(Target(lon, lat, date))
    if not targets:
        raise ValueError(f"No download targets found in {csv_path}")
    return targets


def get_nasa_token() -> str:
    if load_dotenv is not None:
        load_dotenv()
    token = os.getenv("NASA_TOKEN")
    if not token:
        raise RuntimeError(
            "NASA_TOKEN environment variable is not set. Provide an Earthdata token "
            "via the environment or a .env file before running downloads."
        )
    return token


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def build_tile_lookup(tile_shapefile: Path) -> Dict[str, Tuple[float, float, float, float]]:
    gdf = gpd.read_file(tile_shapefile)
    if "TileID" not in gdf.columns:
        raise RuntimeError("Tile shapefile is missing the 'TileID' column")
    lookup: Dict[str, Tuple[float, float, float, float]] = {}
    for _, row in gdf.iterrows():
        tile_id = str(row["TileID"])
        lookup[tile_id] = row.geometry.bounds  # (minx, miny, maxx, maxy)
    return lookup


def clip_bbox(lon: float, lat: float, patch_size_pix: int, pixel_deg: float) -> Tuple[float, float, float, float]:
    half = (patch_size_pix * pixel_deg) / 2
    west = max(-180.0, lon - half)
    east = min(180.0, lon + half)
    south = max(-90.0, lat - half)
    north = min(90.0, lat + half)
    return west, south, east, north


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    cmr_search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    bbox_str = ",".join(f"{coord:.6f}" for coord in bbox)
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": bbox_str,
        "page_size": 50,
    }
    logging.debug("CMR search params: %s", params)
    response = requests.get(cmr_search_url, params=params, timeout=60)
    response.raise_for_status()
    granules = response.json().get("feed", {}).get("entry", [])
    urls: List[str] = []
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                urls.append(href)
    return urls


def download_with_token(url: str, dest: Path, token: str) -> None:
    logging.debug("Downloading %s", url)
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)


def remove_safely(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logging.warning("Could not delete temporary file %s", path)


def enforce_patch_size(patch: np.ndarray, patch_size_pix: int) -> np.ndarray:
    patch = patch[:patch_size_pix, :patch_size_pix]
    pad_h = max(0, patch_size_pix - patch.shape[0])
    pad_w = max(0, patch_size_pix - patch.shape[1])
    if pad_h or pad_w:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return patch


def extract_patch_from_geotiff(tif_path: Path, lon: float, lat: float, patch_size_pix: int) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(tif_path) as src:
        tif_res_deg = src.transform[0]
        bbox = clip_bbox(lon, lat, patch_size_pix, tif_res_deg)
        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        patch = src.read(1, window=window)
        patch = enforce_patch_size(patch, patch_size_pix)
        patch_transform = src.window_transform(window)
        profile = src.meta.copy()
        profile.update({"height": patch_size_pix, "width": patch_size_pix, "transform": patch_transform})
    return patch, profile


def convert_h5_to_geotiff(h5_path: Path, tile_lookup: Dict[str, Tuple[float, float, float, float]], temp_dir: Path) -> Path:
    tif_path = temp_dir / (h5_path.stem + ".tif")
    with h5py.File(h5_path, "r") as f:
        dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if dataset_path not in f:
            raise RuntimeError(f"Dataset {dataset_path} not found in {h5_path.name}")
        ntl_data = f[dataset_path][...]
        tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
        if not tile_match:
            raise RuntimeError(f"Could not infer tile ID from {h5_path.name}")
        tile_id = tile_match.group()
        if tile_id not in tile_lookup:
            raise RuntimeError(f"Tile {tile_id} not present in tile shapefile")
        left, bottom, right, top = tile_lookup[tile_id]
    transform = rio_from_bounds(left, bottom, right, top, ntl_data.shape[1], ntl_data.shape[0])
    profile = {
        "driver": "GTiff",
        "height": ntl_data.shape[0],
        "width": ntl_data.shape[1],
        "count": 1,
        "dtype": ntl_data.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(ntl_data, 1)
    return tif_path


def download_black_marble_patch(
    target: Target,
    token: str,
    tile_lookup: Dict[str, Tuple[float, float, float, float]],
    output_dir: Path,
    temp_dir: Path,
    patch_size_pix: int = PATCH_SIZE_PIXELS,
) -> Path | None:
    bbox = clip_bbox(target.longitude, target.latitude, patch_size_pix, VIIRS_PIXEL_DEGREES)
    urls = search_nasa_cmr(BM_COLLECTION_ID, target.date_str, bbox)
    if not urls:
        logging.warning("No Black Marble granules found for %s at (%.3f, %.3f)", target.date_str, target.longitude, target.latitude)
        return None

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for url in urls:
        h5_path = temp_dir / Path(url).name
        try:
            if not h5_path.exists():
                download_with_token(url, h5_path, token)
            tif_path = convert_h5_to_geotiff(h5_path, tile_lookup, temp_dir)
            patch, profile = extract_patch_from_geotiff(tif_path, target.longitude, target.latitude, patch_size_pix)
            if patch.shape[0] < patch_size_pix or patch.shape[1] < patch_size_pix:
                logging.warning("Patch smaller than requested size for %s (%s)", url, target.date_str)
                continue
            out_name = f"BM_patch_{target.date_str}_{target.longitude:.3f}_{target.latitude:.3f}.tif"
            out_path = output_dir / out_name
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(patch, 1)
            logging.info("Saved Black Marble patch %s", out_path)
            return out_path
        except Exception as exc:  # pragma: no cover - network and IO heavy
            logging.error("Failed to process %s: %s", url, exc)
        finally:
            remove_safely(h5_path)
            tif_candidate = h5_path.with_suffix(".tif")
            remove_safely(tif_candidate)
    return None


def download_black_marble_batch(
    targets: Sequence[Target],
    token: str,
    tile_lookup: Dict[str, Tuple[float, float, float, float]],
    output_dir: Path,
    temp_dir: Path,
    patch_size_pix: int = PATCH_SIZE_PIXELS,
    max_workers: int = 4,
) -> List[Path]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = []
    results: List[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for target in targets:
            jobs.append(
                executor.submit(
                    download_black_marble_patch,
                    target,
                    token,
                    tile_lookup,
                    output_dir,
                    temp_dir,
                    patch_size_pix,
                )
            )
        for future in as_completed(jobs):
            result = future.result()
            if result is not None:
                results.append(result)
    results.sort()
    return results


# ---------------------------------------------------------------------------
# DMSP matching utilities
# ---------------------------------------------------------------------------
def wait_for_file_release(path: Path, timeout: int = 30) -> None:
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
    for attempt in range(5):
        try:
            s3.download_file(bucket, key, str(outpath))
            wait_for_file_release(outpath)
            return True
        except (boto_exceptions.EndpointConnectionError, boto_exceptions.ClientError) as exc:
            logging.warning("Retry %s for %s due to %s", attempt + 1, key, exc)
        except Exception as exc:  # pragma: no cover - best effort
            logging.error("Unexpected error downloading %s: %s", key, exc)
        import time

        time.sleep(2)
    logging.error("Failed to download %s after multiple attempts", key)
    return False


def group_by_f_number(file_keys: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for key in file_keys:
        base = Path(key).name
        f_number = base.split("_")[0] if "_" in base else base[:3]
        groups.setdefault(f_number, []).append(key)
    return groups


def reproject_to_bm_grid(src_path: Path, bm_profile: Dict) -> np.ndarray:
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
    file_keys: Iterable[str],
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
) -> List[Path]:
    saved: List[Path] = []
    dmsp_out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    groups = group_by_f_number(file_keys)
    for f_number, keys in groups.items():
        best_valid_pixels = 0
        best_vis_patch = None
        best_source = None
        for key in keys:
            vis_file = temp_dir / Path(key).name
            if not vis_file.exists():
                if not safe_download(s3, bucket_name, key, vis_file):
                    continue
            try:
                vis_patch = reproject_to_bm_grid(vis_file, bm_profile)
            except Exception as exc:  # pragma: no cover - GDAL heavy
                logging.error("Failed to reproject %s: %s", vis_file, exc)
                continue
            valid_pixels = np.sum(~np.isnan(vis_patch))
            median_val = np.nanmedian(vis_patch) if valid_pixels else 0
            if valid_pixels > best_valid_pixels and median_val > 1:
                best_valid_pixels = valid_pixels
                best_vis_patch = vis_patch
                best_source = vis_file
        if best_vis_patch is None:
            logging.info("No usable DMSP scene found for %s and %s", f_number, bm_patch_path.name)
            continue
        valid_fraction = best_valid_pixels / (bm_shape[0] * bm_shape[1])
        if valid_fraction < 0.10:
            logging.info(
                "Skipping %s for %s – only %.2f%% valid pixels",
                f_number,
                bm_patch_path.name,
                valid_fraction * 100,
            )
            continue
        out_name = f"{f_number}_{best_source.stem}_match_{bm_patch_path.stem}.tif"
        out_path = dmsp_out_dir / out_name
        profile = bm_profile.copy()
        profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(best_vis_patch.astype(np.float32), 1)
        saved.append(out_path)
        logging.info("Saved DMSP match %s", out_path)
    for file in temp_dir.iterdir():
        try:
            file.unlink()
        except OSError:
            logging.warning("Could not delete temporary DMSP file %s", file)
    return saved


def list_dmsp_candidates(s3, bucket_name: str, dmsp_date: str) -> List[str]:
    keys: List[str] = []
    satellites = [f"F{n}" for n in range(10, 19)]
    for satellite in satellites:
        prefix = f"{satellite}{dmsp_date[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if dmsp_date in key and key.endswith(".vis.co.tif"):
                    keys.append(key)
    return keys


def download_dmsp_matches(
    bm_patch_paths: Sequence[Path],
    dmsp_out_dir: Path,
    temp_dir: Path,
    max_workers: int = 4,
) -> List[Path]:
    ensure_directories(dmsp_out_dir, temp_dir)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    saved: List[Path] = []

    def worker(bm_patch_path: Path) -> List[Path]:
        date_token = bm_patch_path.stem.split("_")[2]
        dmsp_date = date_token.replace("-", "")
        file_keys = list_dmsp_candidates(s3, DMSP_BUCKET, dmsp_date)
        if not file_keys:
            logging.warning("No DMSP scenes found for %s", bm_patch_path.name)
            return []
        return process_bm_patch_for_best_fnumber(bm_patch_path, file_keys, s3, DMSP_BUCKET, temp_dir, dmsp_out_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, path): path for path in bm_patch_paths}
        for future in as_completed(futures):
            try:
                saved.extend(future.result())
            except Exception as exc:  # pragma: no cover - network heavy
                logging.error("Error processing %s: %s", futures[future], exc)
    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Black Marble and matching DMSP GeoTIFF patches.")
    parser.add_argument("--targets", type=Path, required=True, help="CSV file containing Longitude, Latitude, date columns.")
    parser.add_argument("--bm-output", type=Path, default=DEFAULT_BM_OUTPUT)
    parser.add_argument("--dmsp-output", type=Path, default=DEFAULT_DMSP_OUTPUT)
    parser.add_argument("--bm-temp", type=Path, default=DEFAULT_BM_TEMP)
    parser.add_argument("--dmsp-temp", type=Path, default=DEFAULT_DMSP_TEMP)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_PIXELS)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    targets = load_targets_from_csv(args.targets)
    token = get_nasa_token()
    tile_lookup = build_tile_lookup(BM_TILE_SHP)
    ensure_directories(args.bm_output, args.dmsp_output, args.bm_temp, args.dmsp_temp)

    logging.info("Downloading %s Black Marble patches (patch size %spx)", len(targets), args.patch_size)
    bm_patches = download_black_marble_batch(
        targets,
        token,
        tile_lookup,
        args.bm_output,
        args.bm_temp,
        patch_size_pix=args.patch_size,
        max_workers=args.max_workers,
    )
    if not bm_patches:
        logging.warning("No Black Marble patches were saved; aborting DMSP matching.")
        return

    logging.info("Starting DMSP matching for %s Black Marble patches", len(bm_patches))
    dmsp_matches = download_dmsp_matches(
        bm_patches,
        args.dmsp_output,
        args.dmsp_temp,
        max_workers=args.max_workers,
    )
    logging.info("Finished – saved %s DMSP GeoTIFFs", len(dmsp_matches))

    # Clean up temporary directories to avoid stale large files
    for tmp_dir in [args.bm_temp, args.dmsp_temp]:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
