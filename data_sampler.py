#!/usr/bin/env python3
"""Utilities for downloading paired Black Marble and DMSP GeoTIFF patches.

The original notebook that lived in this repository mixed together sampling,
plotting, augmentation, and download routines.  This module keeps only the
components required to fetch Black Marble (BM) patches and the corresponding
Defense Meteorological Satellite Program (DMSP) nightlight rasters.

The script can be invoked directly with a CSV or JSON file that lists the
requested downloads.  Each entry must contain ``latitude``, ``longitude`` and a
``date`` (``YYYY-MM-DD``).  For every request the script will:

1. Query NASA's CMR for the specified BM collection and download the matching
   HDF5 granule.
2. Convert the granule to GeoTIFF and crop a 1000x1000 patch centred on the
   requested coordinate.
3. Discover and download the best DMSP overpass for the same date, reproject it
   onto the BM grid, and persist the result as another GeoTIFF.

Example usage::

    python data_sampler.py \
        --requests requests.csv \
        --tile-shapefile path/to/BlackMarbleTiles.shp \
        --nasa-token $EARTHDATA_TOKEN
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.windows import Window
from rasterio.warp import Resampling, reproject
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 1000
DEFAULT_COLLECTION_ID = "C2021957295-LPCLOUD"  # VNP46A2 Daily At-Sensor Radiance
BM_OUTPUT_DIR = Path("Raw_NL_Data/BM data")
DMSP_OUTPUT_DIR = Path("Raw_NL_Data/DMSP data")
DMSP_BUCKET = "globalnightlight"
VIIRS_DEGREES_PER_PIXEL = 0.004  # Approximate resolution of daily BM tiles


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class DownloadRequest:
    """Normalized download request."""

    latitude: float
    longitude: float
    date: str  # YYYY-MM-DD


def load_requests(path: Path) -> List[DownloadRequest]:
    """Parse a CSV or JSON file into :class:`DownloadRequest` objects."""

    if not path.exists():
        raise FileNotFoundError(f"Request file not found: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            records = list(reader)
    elif path.suffix.lower() in {".json", ".geojson"}:
        with path.open(encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, dict):
            records = payload.get("features") or payload.get("requests") or []
            if records and "geometry" in records[0]:
                tmp = []
                for feature in records:
                    props = feature.get("properties", {})
                    geom = feature.get("geometry", {})
                    coords = geom.get("coordinates")
                    if not coords:
                        continue
                    lon, lat = coords[:2]
                    props = {**props, "longitude": lon, "latitude": lat}
                    tmp.append(props)
                records = tmp
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError("Unsupported JSON structure for requests file")
    else:
        raise ValueError("Requests file must be .csv or .json")

    parsed: List[DownloadRequest] = []
    for idx, record in enumerate(records):
        if record is None:
            continue
        if hasattr(record, "items"):
            lower = {str(k).lower(): v for k, v in record.items()}
        else:
            raise ValueError(f"Request #{idx} is not a mapping: {record!r}")

        try:
            lat = float(lower["latitude"])
            lon = float(lower["longitude"])
            date = str(lower["date"]).strip()
        except KeyError as exc:  # pragma: no cover - validation guard
            raise ValueError(f"Missing field {exc} in request #{idx}: {record}") from exc

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            logger.warning("Skipping request with out-of-range coordinates: %s", record)
            continue
        parsed.append(DownloadRequest(latitude=lat, longitude=lon, date=date))

    if not parsed:
        raise ValueError("No valid download requests were found.")
    return parsed


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    half_deg = (patch_size_pix * tif_res_deg) / 2.0
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def search_nasa_cmr(
    session: requests.Session,
    collection_id: str,
    date_str: str,
    bbox: Sequence[float],
) -> List[str]:
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": ",".join(f"{b:.6f}" for b in bbox),
        "page_size": 50,
    }
    response = session.get("https://cmr.earthdata.nasa.gov/search/granules.json", params=params, timeout=60)
    response.raise_for_status()
    h5_links: List[str] = []
    granules = response.json().get("feed", {}).get("entry", [])
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                h5_links.append(href)
    return h5_links


def download_cmr_granule(
    session: requests.Session,
    url: str,
    token: str,
    temp_dir: Path,
) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    destination = temp_dir / Path(url).name
    if destination.exists():
        return destination

    headers = {"Authorization": f"Bearer {token}"}
    with session.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return destination


def convert_h5_to_geotiff(h5_path: Path, tile_gdf: gpd.GeoDataFrame, temp_dir: Path) -> Path:
    tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not tile_match:
        raise RuntimeError(f"Could not infer tile id from {h5_path.name}")
    tile_id = tile_match.group()
    bounds_row = tile_gdf.loc[tile_gdf["TileID"] == tile_id]
    if bounds_row.empty:
        raise RuntimeError(f"Tile {tile_id} not present in provided shapefile")
    left, bottom, right, top = bounds_row.total_bounds

    with h5py.File(h5_path, "r") as h5:
        dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if dataset_path not in h5:
            raise RuntimeError(f"Dataset {dataset_path} absent from {h5_path.name}")
        data = np.array(h5[dataset_path][...], dtype=np.float32)

    tif_path = temp_dir / f"{h5_path.stem}.tif"
    transform = rio_from_bounds(left, bottom, right, top, data.shape[1], data.shape[0])
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
    return tif_path


def extract_patch_from_geotiff(tif_path: Path, lon: float, lat: float, patch_size_pix: int) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif_path) as src:
        row, col = src.index(lon, lat)
        half = patch_size_pix // 2
        window = Window(col - half, row - half, patch_size_pix, patch_size_pix)
        patch = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan,
            out_dtype=np.float32,
        )
        patch_transform = src.window_transform(window)
        patch_meta = src.meta.copy()
        patch_meta.update(
            height=patch_size_pix,
            width=patch_size_pix,
            transform=patch_transform,
            dtype="float32",
            count=1,
            nodata=np.nan,
        )
    return patch, patch_meta


def save_patch(patch: np.ndarray, patch_meta: dict, output_dir: Path, request: DownloadRequest) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"BM_patch_{request.date}_{request.latitude:.6f}_{request.longitude:.6f}.tif"
    out_path = output_dir / filename
    with rasterio.open(out_path, "w", **patch_meta) as dst:
        dst.write(patch, 1)
    logger.info("Saved BM patch -> %s", out_path)
    return out_path


def download_bm_patch(
    session: requests.Session,
    request: DownloadRequest,
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_gdf: gpd.GeoDataFrame,
    output_dir: Path,
    temp_dir: Path,
) -> Optional[Path]:
    if request.latitude < -60:
        logger.warning("Skipping Antarctica sample at (%.3f, %.3f)", request.longitude, request.latitude)
        return None

    bbox = get_patch_bbox(request.longitude, request.latitude, patch_size_pix, VIIRS_DEGREES_PER_PIXEL)
    if bbox[1] < -60:
        bbox = [bbox[0], -60.0, bbox[2], bbox[3]]
    try:
        urls = search_nasa_cmr(session, collection_id, request.date, bbox)
    except requests.HTTPError as exc:
        logger.error("NASA CMR search failed for %s: %s", request, exc)
        return None

    if not urls:
        logger.warning("No BM granules found for %s", request)
        return None

    for url in urls:
        h5_path: Optional[Path] = None
        tif_path: Optional[Path] = None
        try:
            h5_path = download_cmr_granule(session, url, token, temp_dir)
            tif_path = convert_h5_to_geotiff(h5_path, tile_gdf, temp_dir)
            patch, patch_meta = extract_patch_from_geotiff(tif_path, request.longitude, request.latitude, patch_size_pix)
            return save_patch(patch, patch_meta, output_dir, request)
        except Exception as exc:  # pragma: no cover - network & IO heavy
            logger.error("Failed to process %s: %s", url, exc)
            continue
        finally:
            for tmp in (tif_path, h5_path):
                if tmp is None:
                    continue
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
    logger.error("All candidate granules failed for %s", request)
    return None


def safe_download(s3, bucket: str, key: str, destination: Path, max_retries: int = 5) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries):
        try:
            s3.download_file(bucket, key, str(destination))
            return True
        except botocore.exceptions.EndpointConnectionError as exc:
            logger.warning("Endpoint error fetching %s (attempt %d/%d): %s", key, attempt + 1, max_retries, exc)
        except botocore.exceptions.ClientError as exc:
            logger.warning("Client error fetching %s (attempt %d/%d): %s", key, attempt + 1, max_retries, exc)
        except Exception as exc:  # pragma: no cover - network heavy
            logger.warning("Unexpected error fetching %s (attempt %d/%d): %s", key, attempt + 1, max_retries, exc)
        if attempt < max_retries - 1:
            import time

            time.sleep(2)
    logger.error("Failed to download %s after %d attempts", key, max_retries)
    return False


def group_by_f_number(keys: Iterable[str]) -> dict[str, List[str]]:
    groups: dict[str, List[str]] = {}
    for key in keys:
        base = Path(key).name
        f_number = base.split("_")[0] if "_" in base else base[:3]
        groups.setdefault(f_number, []).append(key)
    return groups


def reproject_to_bm_grid(src_path: Path, bm_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_data[src_data == 255] = np.nan
        dst = np.full((bm_profile["height"], bm_profile["width"]), np.nan, dtype=np.float32)
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


def select_dmsp_patches(
    bm_patch_path: Path,
    dmsp_keys: Sequence[str],
    s3,
    bucket: str,
    temp_dir: Path,
    output_dir: Path,
) -> List[Path]:
    results: List[Path] = []
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    grouped = group_by_f_number(dmsp_keys)
    for f_number, keys in grouped.items():
        best_patch: Optional[np.ndarray] = None
        best_path: Optional[Path] = None
        best_valid_pixels = 0
        for key in keys:
            temp_file = temp_dir / Path(key).name
            if not temp_file.exists():
                if not safe_download(s3, bucket, key, temp_file):
                    continue
            try:
                dmsp_patch = reproject_to_bm_grid(temp_file, bm_profile)
            except Exception as exc:  # pragma: no cover - reprojection heavy
                logger.warning("Failed to reproject %s: %s", key, exc)
                continue
            valid_pixels = int(np.sum(~np.isnan(dmsp_patch)))
            median_val = float(np.nanmedian(dmsp_patch)) if valid_pixels else 0.0
            if valid_pixels > best_valid_pixels and median_val > 1.0:
                best_valid_pixels = valid_pixels
                best_patch = dmsp_patch
                best_path = temp_file
        if best_patch is None:
            logger.info("No valid DMSP candidate for %s (%s)", bm_patch_path.name, f_number)
            continue
        valid_fraction = best_valid_pixels / (bm_shape[0] * bm_shape[1])
        if valid_fraction < 0.10:
            logger.info(
                "Skipping %s for %s due to sparse coverage (%.2f%%)",
                f_number,
                bm_patch_path.name,
                valid_fraction * 100,
            )
            continue
        out_profile = bm_profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=np.nan)
        out_name = f"{f_number}_{best_path.stem}_match_{bm_patch_path.stem}.tif"
        out_file = output_dir / out_name
        output_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_file, "w", **out_profile) as dst:
            dst.write(best_patch.astype(np.float32), 1)
        results.append(out_file)
        logger.info("Saved DMSP patch -> %s", out_file)
    return results


def list_dmsp_keys_for_date(date_str: str, s3, bucket: str) -> List[str]:
    dmsp_keys: List[str] = []
    satellites = [f"F{num}" for num in range(10, 19)]
    for sat in satellites:
        prefix = f"{sat}{date_str[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if date_str in key and key.endswith(".vis.co.tif"):
                    dmsp_keys.append(key)
    return dmsp_keys


def download_dmsp_for_patch(
    bm_patch_path: Path,
    s3,
    bucket: str,
    temp_root: Path,
    output_dir: Path,
) -> List[Path]:
    patch_parts = bm_patch_path.stem.split("_")
    if len(patch_parts) < 3:
        logger.warning("Unable to parse date from %s", bm_patch_path.name)
        return []
    date_str = patch_parts[2].replace("-", "")
    dmsp_keys = list_dmsp_keys_for_date(date_str, s3, bucket)
    if not dmsp_keys:
        logger.info("No DMSP data found for %s", bm_patch_path.name)
        return []

    temp_dir = temp_root / bm_patch_path.stem
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        return select_dmsp_patches(bm_patch_path, dmsp_keys, s3, bucket, temp_dir, output_dir)
    finally:
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paired BM and DMSP GeoTIFFs")
    parser.add_argument("--requests", required=True, type=Path, help="CSV or JSON file describing download requests")
    parser.add_argument("--tile-shapefile", required=True, type=Path, help="Shapefile containing Black Marble tile bounds")
    parser.add_argument("--nasa-token", default=None, help="NASA Earthdata authentication token")
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID, help="NASA CMR collection concept ID")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size in pixels (default: 1000)")
    parser.add_argument("--bm-output", type=Path, default=BM_OUTPUT_DIR, help="Output directory for BM patches")
    parser.add_argument("--dmsp-output", type=Path, default=DMSP_OUTPUT_DIR, help="Output directory for DMSP patches")
    parser.add_argument("--temp-dir", type=Path, default=Path("temp_dl"), help="Temporary working directory")
    parser.add_argument("--max-workers", type=int, default=4, help="Max threads for DMSP downloads")
    parser.add_argument("--skip-dmsp", action="store_true", help="Only download BM patches")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.verbose)

    token = args.nasa_token or os.environ.get("NASA_TOKEN") or os.environ.get("EARTHDATA_TOKEN")
    if not token:
        raise ValueError("A NASA Earthdata token must be provided via --nasa-token or the NASA_TOKEN/EARTHDATA_TOKEN environment variables.")

    requests_list = load_requests(args.requests)
    tile_gdf = gpd.read_file(args.tile_shapefile)

    bm_paths: List[Path] = []
    bm_temp_dir = args.temp_dir / "bm"
    bm_temp_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for request in tqdm(requests_list, desc="BM requests"):
            path = download_bm_patch(
                session=session,
                request=request,
                patch_size_pix=args.patch_size,
                collection_id=args.collection_id,
                token=token,
                tile_gdf=tile_gdf,
                output_dir=args.bm_output,
                temp_dir=bm_temp_dir,
            )
            if path:
                bm_paths.append(path)

    if not bm_paths:
        logger.warning("No BM patches were downloaded; skipping DMSP stage.")
        return

    if args.skip_dmsp:
        logger.info("--skip-dmsp supplied; finishing after BM downloads.")
        return

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    dmsp_temp_root = args.temp_dir / "dmsp"
    dmsp_temp_root.mkdir(parents=True, exist_ok=True)

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                download_dmsp_for_patch,
                bm_path,
                s3,
                DMSP_BUCKET,
                dmsp_temp_root,
                args.dmsp_output,
            )
            for bm_path in bm_paths
        ]
        for future in tqdm(futures, total=len(futures), desc="DMSP patches"):
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - network heavy
                logger.error("DMSP worker failed: %s", exc)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
