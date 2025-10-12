"""Simplified downloader for paired Black Marble and DMSP GeoTIFF patches.

The original notebook-derived script orchestrated an entire sampling and
augmentation pipeline.  This module now focuses solely on downloading
Black Marble (BM) night-lights patches together with the best-matching
Defense Meteorological Satellite Program (DMSP) composites.

Usage::

    python data_sampler.py --requests samples.csv --nasa-token <token>

The requests file should contain longitude, latitude, and date
(YYYY-MM-DD) columns (CSV) or keys (JSON).  Patches are saved under
``Raw_NL_Data/BM data`` and ``Raw_NL_Data/DMSP data`` by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import boto3
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import requests
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.windows import from_bounds
from rasterio.warp import Resampling, reproject

# ---------------------------------------------------------------------------
# Data containers and parsing helpers


@dataclass
class SampleRequest:
    """A single download request."""

    longitude: float
    latitude: float
    date: datetime

    @property
    def date_str(self) -> str:
        return self.date.strftime("%Y-%m-%d")

    @property
    def date_compact(self) -> str:
        return self.date.strftime("%Y%m%d")


def _normalise_key(key: str) -> str:
    return key.strip().lower()


def _parse_date(value) -> datetime:
    value_str = str(value).strip()
    value_str = value_str.replace("/", "-")
    if value_str.isdigit() and len(value_str) == 8:
        value_str = f"{value_str[:4]}-{value_str[4:6]}-{value_str[6:8]}"
    return datetime.fromisoformat(value_str)


def load_requests(path: Path) -> List[SampleRequest]:
    """Load download requests from a CSV or JSON file.

    The loader is case-insensitive to field names and expects longitude,
    latitude, and date (YYYY-MM-DD).
    """

    if not path.exists():
        raise FileNotFoundError(f"Request file not found: {path}")

    def to_request(row: Dict[str, str]) -> SampleRequest:
        mapping = {_normalise_key(str(k)): v for k, v in row.items()}
        try:
            lon = float(mapping["longitude"])
            lat = float(mapping["latitude"])
            date = _parse_date(mapping["date"])
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"Missing field in request row: {exc}") from exc
        return SampleRequest(longitude=lon, latitude=lat, date=date)

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("requests", [])
        if not isinstance(payload, Sequence):
            raise ValueError("JSON request file must be a list or contain a 'requests' list")
        return [to_request(item) for item in payload]

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [to_request(row) for row in reader]


# ---------------------------------------------------------------------------
# Black Marble helpers


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    half_deg = (patch_size_pix * tif_res_deg) / 2.0
    bottom = max(-60.0, lat - half_deg)
    top = max(-60.0, lat + half_deg)
    return [lon - half_deg, bottom, lon + half_deg, top]


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    cmr_search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": ",".join(f"{coord:.6f}" for coord in bbox),
        "page_size": 50,
    }
    response = requests.get(cmr_search_url, params=params, timeout=60)
    response.raise_for_status()
    granules = response.json().get("feed", {}).get("entry", [])
    h5_links: List[str] = []
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                h5_links.append(href)
    return h5_links


def download_h5(url: str, token: str, destination: Path) -> Path:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def convert_h5_to_geotiff(h5_path: Path, tile_index: gpd.GeoDataFrame, tif_path: Path) -> Path:
    dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
    with h5py.File(h5_path, "r") as handle:
        if dataset_path not in handle:
            raise RuntimeError(f"Dataset {dataset_path} not present in {h5_path.name}")
        ntl_data = handle[dataset_path][...]
    tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if tile_match is None:
        raise RuntimeError(f"Unable to parse tile ID from {h5_path.name}")
    tile_id_match = tile_match.group()
    bounds_row = tile_index[tile_index["TileID"] == tile_id_match]
    if bounds_row.empty:
        raise RuntimeError(f"Tile ID {tile_id_match} missing from tile index")
    left, bottom, right, top = bounds_row.total_bounds
    transform = rio_from_bounds(left, bottom, right, top, ntl_data.shape[1], ntl_data.shape[0])
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=ntl_data.shape[0],
        width=ntl_data.shape[1],
        count=1,
        dtype=ntl_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(ntl_data, 1)
    return tif_path


def enforce_patch_size(patch: np.ndarray, patch_size: int) -> np.ndarray:
    clipped = patch[:patch_size, :patch_size]
    pad_h = max(0, patch_size - clipped.shape[0])
    pad_w = max(0, patch_size - clipped.shape[1])
    if pad_h > 0 or pad_w > 0:
        clipped = np.pad(clipped, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return clipped


def extract_patch(tif_path: Path, lon: float, lat: float, patch_size: int) -> tuple[np.ndarray, Dict]:
    with rasterio.open(tif_path) as src:
        resolution_deg = src.transform[0]
        bbox = get_patch_bbox(lon, lat, patch_size, resolution_deg)
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window)
        patch = enforce_patch_size(patch, patch_size)
        meta = src.meta.copy()
        meta.update({"height": patch_size, "width": patch_size, "transform": src.window_transform(window)})
    return patch, meta


def save_patch(patch: np.ndarray, meta: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(patch, 1)


def download_bm_patch(
    sample: SampleRequest,
    tile_index: gpd.GeoDataFrame,
    collection_id: str,
    token: str,
    output_dir: Path,
    patch_size: int = 1000,
) -> Optional[Path]:
    if sample.latitude < -60:
        print(f"Skipping latitude below -60°: {sample}")
        return None

    bbox = get_patch_bbox(sample.longitude, sample.latitude, patch_size, tif_res_deg=0.004)
    try:
        urls = search_nasa_cmr(collection_id, sample.date_str, bbox)
    except requests.HTTPError as exc:
        print(f"NASA CMR query failed for {sample.date_str} @ ({sample.longitude}, {sample.latitude}): {exc}")
        return None

    if not urls:
        print(f"No Black Marble granule found for {sample.date_str} at {sample.longitude:.3f}, {sample.latitude:.3f}")
        return None

    output_path = output_dir / f"BM_patch_{sample.date_str}_{sample.latitude:.3f}_{sample.longitude:.3f}.tif"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        h5_path = tmpdir_path / Path(urls[0]).name
        tif_path = h5_path.with_suffix(".tif")
        try:
            download_h5(urls[0], token, h5_path)
            convert_h5_to_geotiff(h5_path, tile_index, tif_path)
            patch, meta = extract_patch(tif_path, sample.longitude, sample.latitude, patch_size)
            if patch.shape != (patch_size, patch_size):
                print(f"Patch size mismatch ({patch.shape}) for {sample}; skipping")
                return None
            save_patch(patch, meta, output_path)
        except Exception as exc:
            print(f"Failed to download BM patch for {sample}: {exc}")
            return None

    print(f"Saved BM patch: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# DMSP helpers


def create_s3_client() -> boto3.client:
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def find_dmsp_keys(s3_client, bucket: str, date_compact: str) -> List[str]:
    keys: List[str] = []
    year_prefix = date_compact[:4]
    satellites = [f"F{n}" for n in range(10, 19)]
    paginator = s3_client.get_paginator("list_objects_v2")
    for satellite in satellites:
        prefix = f"{satellite}{year_prefix}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".vis.co.tif") and date_compact in key:
                    keys.append(key)
    return keys


def reproject_to_bm_grid(src_path: Path, bm_profile: Dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_data[src_data == 255] = np.nan
        destination = np.full((bm_profile["height"], bm_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=bm_profile["transform"],
            dst_crs=bm_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return destination


def download_best_dmsp_match(
    sample: SampleRequest,
    bm_patch_path: Path,
    s3_client,
    output_dir: Path,
    bucket: str = "globalnightlight",
    min_valid_fraction: float = 0.1,
) -> Optional[Path]:
    keys = find_dmsp_keys(s3_client, bucket, sample.date_compact)
    if not keys:
        print(f"No DMSP scenes found for {sample.date_compact}")
        return None

    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_area = bm_src.width * bm_src.height

    best_patch: Optional[np.ndarray] = None
    best_key: Optional[str] = None
    best_valid = -1

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for key in keys:
            local_path = tmpdir_path / Path(key).name
            try:
                s3_client.download_file(bucket, key, str(local_path))
            except (BotoCoreError, ClientError) as exc:
                print(f"Failed to download {key}: {exc}")
                continue
            try:
                patch = reproject_to_bm_grid(local_path, bm_profile)
            except Exception as exc:
                print(f"Failed to reproject {key}: {exc}")
                continue
            valid = np.sum(~np.isnan(patch))
            if valid > best_valid:
                best_valid = valid
                best_patch = patch
                best_key = key

    if not best_patch or best_key is None:
        print(f"No suitable DMSP match for {bm_patch_path.name}")
        return None

    valid_fraction = best_valid / float(bm_area)
    if valid_fraction < min_valid_fraction:
        print(
            f"Best DMSP patch for {bm_patch_path.name} only has {valid_fraction:.1%} valid pixels; skipping"
        )
        return None

    output_path = output_dir / f"DMSP_match_{Path(best_key).stem}_{bm_patch_path.stem}.tif"
    output_profile = bm_profile.copy()
    output_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(best_patch.astype(np.float32), 1)

    print(f"Saved DMSP match: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main orchestration


def process_requests(
    requests_file: Path,
    nasa_token: str,
    tile_index_path: Path,
    bm_output: Path,
    dmsp_output: Path,
    collection_id: str,
    patch_size: int,
    limit: Optional[int] = None,
) -> None:
    samples = load_requests(requests_file)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        print("No samples to process.")
        return

    tile_index = gpd.read_file(tile_index_path)
    s3_client = create_s3_client()

    ensure_directory(bm_output)
    ensure_directory(dmsp_output)

    for sample in samples:
        bm_path = download_bm_patch(sample, tile_index, collection_id, nasa_token, bm_output, patch_size)
        if not bm_path:
            continue
        download_best_dmsp_match(sample, bm_path, s3_client, dmsp_output)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paired BM and DMSP GeoTIFF patches")
    parser.add_argument("--requests", required=True, type=Path, help="Path to CSV or JSON request file")
    parser.add_argument(
        "--tile-index",
        type=Path,
        default=Path("Data/Black_Marble_IDs/Black_Marble_World_tiles.shp"),
        help="Path to the Black Marble tile shapefile",
    )
    parser.add_argument("--bm-output", type=Path, default=Path("Raw_NL_Data/BM data"))
    parser.add_argument("--dmsp-output", type=Path, default=Path("Raw_NL_Data/DMSP data"))
    parser.add_argument(
        "--collection-id",
        default="C3365931269-LAADS",
        help="NASA CMR collection concept ID for Black Marble",
    )
    parser.add_argument("--patch-size", type=int, default=1000, help="Patch size in pixels")
    parser.add_argument("--limit", type=int, help="Optional limit on number of requests to process")
    parser.add_argument("--nasa-token", type=str, help="NASA Earthdata bearer token")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    nasa_token = args.nasa_token or os.environ.get("NASA_TOKEN")
    if not nasa_token:
        print("A NASA token is required. Provide --nasa-token or set NASA_TOKEN.")
        sys.exit(1)

    process_requests(
        requests_file=args.requests,
        nasa_token=nasa_token,
        tile_index_path=args.tile_index,
        bm_output=args.bm_output,
        dmsp_output=args.dmsp_output,
        collection_id=args.collection_id,
        patch_size=args.patch_size,
        limit=args.limit,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
