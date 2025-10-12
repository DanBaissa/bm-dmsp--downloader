"""Utility script for downloading paired Black Marble and DMSP GeoTIFF patches.

This module focuses solely on pulling Black Marble (BM) tiles from NASA's
collection and matching them with Defense Meteorological Satellite Program
(DMSP) observations.  It accepts a CSV or JSON file describing the target
locations/dates and produces paired GeoTIFFs with a configurable patch size
(default 1000×1000 px).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import botocore
import geopandas as gpd
import h5py
import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

DEFAULT_COLLECTION_ID = "C3365931269-LAADS"
DEFAULT_DMSP_BUCKET = "globalnightlight"
BM_FILENAME_TEMPLATE = "BM_patch_{date}_{lon:.6f}_{lat:.6f}.tif"
PATCH_RESOLUTION_DEG = 0.004  # Approximate BM VIIRS resolution (deg per pixel)
MIN_VALID_FRACTION = 0.10


@dataclass
class SampleRequest:
    longitude: float
    latitude: float
    date: str  # YYYY-MM-DD

    @classmethod
    def from_mapping(cls, mapping: Dict[str, object]) -> "SampleRequest":
        return cls(
            longitude=float(mapping["Longitude"]),
            latitude=float(mapping["Latitude"]),
            date=str(mapping["date"]).strip(),
        )


def load_requests(path: Path) -> List[SampleRequest]:
    """Load requests from a CSV or JSON file."""
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [SampleRequest.from_mapping(row) for row in reader]
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "samples" in payload:
            payload = payload["samples"]
        if not isinstance(payload, Iterable):  # pragma: no cover - defensive
            raise ValueError("JSON payload must be a list or contain 'samples'.")
        return [SampleRequest.from_mapping(item) for item in payload]
    raise ValueError("Unsupported request file format. Use CSV or JSON.")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    half_deg = (patch_size_pix * tif_res_deg) / 2.0
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def search_nasa_cmr(collection_id: str, date_str: str, bbox: List[float]) -> List[str]:
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
    granules = response.json().get("feed", {}).get("entry", [])
    links = []
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                links.append(href)
    return links


def download_h5(url: str, token: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def convert_h5_to_geotiff(h5_path: Path, tile_index: gpd.GeoDataFrame, dest: Path) -> Path:
    with h5py.File(h5_path, "r") as hf:
        dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if dataset_path not in hf:
            raise RuntimeError(f"Dataset {dataset_path} missing in {h5_path.name}")
        ntl_data = hf[dataset_path][...]
    tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not tile_match:
        raise RuntimeError(f"Could not parse tile ID from {h5_path.name}")
    tile_id = tile_match.group()
    bounds_row = tile_index[tile_index["TileID"] == tile_id]
    if bounds_row.empty:
        raise RuntimeError(f"Tile ID {tile_id} not found in tile index")
    left, bottom, right, top = bounds_row.total_bounds
    with rasterio.open(
        dest,
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
    return dest


def extract_patch_from_geotiff(tif_path: Path, lon: float, lat: float, patch_size_pix: int) -> Optional[np.ndarray]:
    with rasterio.open(tif_path) as src:
        bbox = get_patch_bbox(lon, lat, patch_size_pix, src.transform[0])
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window, boundless=True, fill_value=0)
        if patch.size == 0:
            return None
        patch = patch[:patch_size_pix, :patch_size_pix]
        pad_h = patch_size_pix - patch.shape[0]
        pad_w = patch_size_pix - patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant", constant_values=0)
        return patch


def build_patch_meta(src: rasterio.io.DatasetReader, window, patch_size_pix: int) -> Dict[str, object]:
    meta = src.meta.copy()
    meta.update({
        "height": patch_size_pix,
        "width": patch_size_pix,
        "transform": src.window_transform(window),
        "count": 1,
    })
    return meta


def save_patch(
    tif_path: Path,
    lon: float,
    lat: float,
    date_str: str,
    patch_size_pix: int,
    output_dir: Path,
) -> Optional[Path]:
    with rasterio.open(tif_path) as src:
        bbox = get_patch_bbox(lon, lat, patch_size_pix, src.transform[0])
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window, boundless=True, fill_value=0)
        if patch.size == 0:
            return None
        patch = patch[:patch_size_pix, :patch_size_pix]
        pad_h = patch_size_pix - patch.shape[0]
        pad_w = patch_size_pix - patch.shape[1]
        if pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="constant", constant_values=0)
        meta = build_patch_meta(src, window, patch_size_pix)
    filename = BM_FILENAME_TEMPLATE.format(date=date_str, lon=lon, lat=lat)
    dest_path = output_dir / filename
    with rasterio.open(dest_path, "w", **meta) as dst:
        dst.write(patch, 1)
    return dest_path


def wait_for_file_release(path: Path, timeout: float = 10.0) -> None:
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


def safe_download(s3, bucket: str, key: str, outpath: Path, max_retries: int = 5) -> bool:
    for attempt in range(max_retries):
        try:
            s3.download_file(bucket, key, str(outpath))
            wait_for_file_release(outpath)
            return True
        except botocore.exceptions.BotoCoreError as exc:
            print(f"Error downloading {key} (attempt {attempt + 1}/{max_retries}): {exc}")
    return False


def reproject_to_bm_grid(src_path: Path, bm_profile: Dict[str, object]) -> np.ndarray:
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


def find_dmsp_keys(s3, bucket: str, dmsp_date: str) -> List[str]:
    satellites = [f"F{n}" for n in range(10, 19)]
    keys: List[str] = []
    for sat in satellites:
        prefix = f"{sat}{dmsp_date[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if dmsp_date in key and key.endswith(".vis.co.tif"):
                    keys.append(key)
    return keys


def download_best_dmsp_match(
    bm_patch_path: Path,
    dmsp_out_dir: Path,
    s3,
    bucket: str = DEFAULT_DMSP_BUCKET,
) -> Optional[Path]:
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)
    bm_name = bm_patch_path.stem
    parts = bm_name.split("_")
    if len(parts) < 5:
        print(f"Unable to parse BM patch name: {bm_patch_path.name}")
        return None
    bm_date = parts[2]
    dmsp_date = bm_date.replace("-", "")
    keys = find_dmsp_keys(s3, bucket, dmsp_date)
    if not keys:
        print(f"No DMSP scenes found for {bm_patch_path.name}")
        return None

    ensure_dir(dmsp_out_dir)
    best_patch: Optional[np.ndarray] = None
    best_key: Optional[str] = None
    best_valid = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for key in keys:
            local_path = tmpdir_path / Path(key).name
            if not safe_download(s3, bucket, key, local_path):
                continue
            try:
                patch = reproject_to_bm_grid(local_path, bm_profile)
            finally:
                if local_path.exists():
                    local_path.unlink()
            valid_pixels = int(np.sum(~np.isnan(patch)))
            if valid_pixels == 0:
                continue
            median_val = float(np.nanmedian(patch))
            if valid_pixels > best_valid and median_val > 1:
                best_valid = valid_pixels
                best_patch = patch
                best_key = key

    if best_patch is None or best_valid / (bm_shape[0] * bm_shape[1]) < MIN_VALID_FRACTION:
        print(f"No suitable DMSP match for {bm_patch_path.name}")
        return None

    out_profile = bm_profile.copy()
    out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
    f_number = Path(best_key).name.split("_")[0]
    out_name = f"{f_number}_{Path(best_key).stem}_match_{bm_patch_path.stem}.tif"
    out_path = dmsp_out_dir / out_name
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(best_patch.astype(np.float32), 1)
    print(f"Saved DMSP patch: {out_path}")
    return out_path


def download_black_marble_patch(
    sample: SampleRequest,
    tile_index: gpd.GeoDataFrame,
    collection_id: str,
    token: str,
    patch_size_pix: int,
    output_dir: Path,
    temp_dir: Path,
) -> Optional[Path]:
    if sample.latitude < -60:
        print(f"Skipping Antarctica sample at ({sample.longitude:.3f}, {sample.latitude:.3f})")
        return None

    bbox = get_patch_bbox(sample.longitude, sample.latitude, patch_size_pix, PATCH_RESOLUTION_DEG)
    if bbox[1] < -60:
        bbox[1] = -60
        bbox[3] = max(bbox[3], -60 + PATCH_RESOLUTION_DEG)

    try:
        urls = search_nasa_cmr(collection_id, sample.date, bbox)
    except requests.HTTPError as exc:
        print(f"CMR search failed for {sample.date} at ({sample.longitude:.3f}, {sample.latitude:.3f}): {exc}")
        return None
    if not urls:
        print(f"No Black Marble granule for {sample.date} at ({sample.longitude:.3f}, {sample.latitude:.3f})")
        return None

    ensure_dir(output_dir)
    ensure_dir(temp_dir)
    for url in urls:
        h5_path = temp_dir / Path(url).name
        try:
            download_h5(url, token, h5_path)
            tif_path = h5_path.with_suffix(".tif")
            convert_h5_to_geotiff(h5_path, tile_index, tif_path)
            patch_path = save_patch(
                tif_path,
                lon=sample.longitude,
                lat=sample.latitude,
                date_str=sample.date,
                patch_size_pix=patch_size_pix,
                output_dir=output_dir,
            )
            if patch_path is not None:
                print(f"Saved BM patch: {patch_path}")
                return patch_path
        except Exception as exc:
            print(f"Failed to process {url}: {exc}")
        finally:
            if h5_path.exists():
                h5_path.unlink()
            tif_path = h5_path.with_suffix(".tif")
            if tif_path.exists():
                tif_path.unlink()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paired BM and DMSP GeoTIFF patches.")
    parser.add_argument("requests", type=Path, help="Path to CSV or JSON describing samples.")
    parser.add_argument("tile_shapefile", type=Path, help="Path to the Black Marble tile shapefile.")
    parser.add_argument("--bm-output", type=Path, default=Path("Raw_NL_Data/BM data"), help="Directory for BM patches.")
    parser.add_argument(
        "--dmsp-output",
        type=Path,
        default=Path("Raw_NL_Data/DMSP data"),
        help="Directory for DMSP patches.",
    )
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--nasa-token", default=None, help="NASA Earthdata token. Defaults to NASA_TOKEN env var.")
    parser.add_argument("--patch-size", type=int, default=1000, help="Patch size in pixels (default 1000).")
    parser.add_argument("--dmsp-bucket", default=DEFAULT_DMSP_BUCKET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.nasa_token or os.environ.get("NASA_TOKEN")
    if not token:
        raise SystemExit("NASA token required. Provide --nasa-token or set NASA_TOKEN env var.")

    samples = load_requests(args.requests)
    if not samples:
        raise SystemExit("No samples found in request file.")

    tile_index = gpd.read_file(args.tile_shapefile)
    temp_dir = Path(tempfile.mkdtemp(prefix="bm_dl_"))
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    try:
        for sample in samples:
            bm_path = download_black_marble_patch(
                sample,
                tile_index=tile_index,
                collection_id=args.collection_id,
                token=token,
                patch_size_pix=args.patch_size,
                output_dir=args.bm_output,
                temp_dir=temp_dir,
            )
            if bm_path is None:
                continue
            download_best_dmsp_match(
                bm_patch_path=bm_path,
                dmsp_out_dir=args.dmsp_output,
                s3=s3,
                bucket=args.dmsp_bucket,
            )
    finally:
        if temp_dir.exists():
            for child in temp_dir.iterdir():
                child.unlink()
            temp_dir.rmdir()


if __name__ == "__main__":
    main()
