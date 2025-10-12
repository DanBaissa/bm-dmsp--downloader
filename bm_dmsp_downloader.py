"""Utilities for sampling locations and downloading paired Black Marble and DMSP GeoTIFFs."""
from __future__ import annotations

import concurrent.futures
import os
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import boto3
import botocore
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
import requests
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from rasterio.features import rasterize
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

load_dotenv()

NASA_TOKEN_ENV = "NASA_TOKEN"
DEFAULT_COLLECTION_ID = "C3365931269-LAADS"
DEFAULT_PATCH_SIZE = 1000


class DownloadError(RuntimeError):
    """Raised when a granule cannot be downloaded or processed."""


@dataclass
class SampleRequest:
    longitude: float
    latitude: float
    date: str

    @classmethod
    def from_dict(cls, row: Dict[str, float]) -> "SampleRequest":
        return cls(
            longitude=float(row["Longitude"]),
            latitude=float(row["Latitude"]),
            date=str(row["date"]),
        )


def load_nasa_token(env_var: str = NASA_TOKEN_ENV) -> str:
    token = os.getenv(env_var)
    if not token:
        raise RuntimeError(
            f"{env_var} environment variable is not set. "
            "Create a .env file (see .env.example) with your NASA Earthdata token before running downloads."
        )
    return token


def generate_population_balanced_samples(
    raster_path: Path,
    shapefile_path: Path,
    *,
    scale: int = 4,
    samples_per_bin: int = 200,
    random_seed: int = 2024,
    antarctica_fid: int = 8,
    south_latitude_cutoff: float = -60.0,
) -> pd.DataFrame:
    """Sample LandScan pixels while balancing by log population bins."""
    with rasterio.open(raster_path) as src:
        new_height = src.height // scale
        new_width = src.width // scale
        array = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        nodata = src.nodata

    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)

    gdf = gpd.read_file(shapefile_path)
    antarctica = gdf[gdf["FID"] == antarctica_fid]

    with rasterio.open(raster_path) as src:
        downsampled_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height),
        )

    mask = rasterize(
        [(geom, 1) for geom in antarctica.geometry],
        out_shape=(new_height, new_width),
        transform=downsampled_transform,
        fill=0,
        dtype="uint8",
    )

    with rasterio.open(raster_path) as src:
        ys = np.linspace(src.bounds.top, src.bounds.bottom, new_height)
    lat_mask = np.repeat(ys[:, np.newaxis], new_width, axis=1)
    south_polar_mask = lat_mask < south_latitude_cutoff

    combined_mask = (mask == 1) | south_polar_mask.astype(bool)
    array = array.astype(float)
    array[combined_mask] = np.nan

    log_array = np.log1p(array)
    valid_mask = (~np.isnan(array)) & (array >= 0)
    rows, cols = np.where(valid_mask)
    logp = log_array[rows, cols]
    pops = array[rows, cols]

    bin_edges = list(range(0, 11)) + [np.inf]
    bin_labels = [f"{i}–{i+1}" for i in range(0, 10)] + ["10+"]
    bins = np.digitize(logp, bin_edges) - 1

    rng = np.random.default_rng(random_seed)
    sampled = []
    for b, label in enumerate(bin_labels):
        idxs = np.where(bins == b)[0]
        if len(idxs) == 0:
            continue
        picks = rng.choice(idxs, size=min(samples_per_bin, len(idxs)), replace=False)
        sampled.extend((rows[p], cols[p], logp[p], pops[p], label) for p in picks)

    with rasterio.open(raster_path) as src:
        scale_x = src.width / new_width
        scale_y = src.height / new_height
        coords = [src.xy(int(r * scale_y), int(c * scale_x)) for r, c, *_ in sampled]

    df = pd.DataFrame(
        {
            "Bin": [s[4] for s in sampled],
            "Longitude": [c[0] for c in coords],
            "Latitude": [c[1] for c in coords],
            "Population": [int(round(s[3])) for s in sampled],
            "log(pop+1)": [float(s[2]) for s in sampled],
        }
    )

    return df


def list_dmsp_dates(min_date: Optional[str] = "20120120") -> List[str]:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"
    prefix = "F"

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    all_dates: set[str] = set()
    min_dt = None
    if min_date:
        min_dt = time.strptime(min_date, "%Y%m%d")

    for page in page_iterator:
        for obj in page.get("Contents", []):
            file_key = obj["Key"]
            if not file_key.endswith(".vis.co.tif"):
                continue
            fname = os.path.basename(file_key)
            if len(fname) < 11:
                continue
            date_str = fname[3:11]
            try:
                dt = time.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue
            if min_dt and dt < min_dt:
                continue
            all_dates.add(date_str)
    return sorted(all_dates)


def assign_dates_to_samples(
    samples: pd.DataFrame,
    dmsp_dates: Sequence[str],
    *,
    random_seed: int = 13492,
) -> pd.DataFrame:
    if len(dmsp_dates) == 0:
        raise ValueError("DMSP date list is empty")
    rng = random.Random(random_seed)
    chosen = [rng.choice(dmsp_dates) for _ in range(len(samples))]
    samples = samples.copy()
    samples["date"] = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in chosen]
    return samples


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    half_deg = (patch_size_pix * tif_res_deg) / 2
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def enforce_patch_size(patch: np.ndarray, patch_size_pix: int) -> np.ndarray:
    patch = patch[:patch_size_pix, :patch_size_pix]
    pad_h = max(0, patch_size_pix - patch.shape[0])
    pad_w = max(0, patch_size_pix - patch.shape[1])
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return patch


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    cmr_search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
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


def download_file(url: str, dest: Path, token: str) -> Path:
    if dest.exists():
        return dest
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=300)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def convert_h5_to_geotiff(
    h5_path: Path,
    tile_bounds: Tuple[float, float, float, float],
    out_path: Path,
) -> Tuple[Path, float]:
    with h5py.File(h5_path, "r") as f:
        ntl_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if ntl_path not in f:
            raise DownloadError(f"Dataset {ntl_path} not found in {h5_path}")
        ntl_data = f[ntl_path][...]
    left, bottom, right, top = tile_bounds
    height, width = ntl_data.shape
    transform = rio_from_bounds(left, bottom, right, top, width, height)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=ntl_data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(ntl_data, 1)
    tif_res_deg = abs((right - left) / width)
    return out_path, tif_res_deg


def extract_patch_from_geotiff(
    tif_path: Path,
    lon: float,
    lat: float,
    patch_size_pix: int,
    tif_res_deg: float,
) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(tif_path) as src:
        bbox = get_patch_bbox(lon, lat, patch_size_pix, tif_res_deg)
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window, boundless=True, fill_value=0)
        patch = enforce_patch_size(patch, patch_size_pix)
        patch_transform = src.window_transform(window)
        patch_meta = src.meta.copy()
        patch_meta.update(
            {
                "height": patch_size_pix,
                "width": patch_size_pix,
                "transform": patch_transform,
            }
        )
    return patch, patch_meta


def process_single_sample(
    sample: SampleRequest,
    *,
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_bounds_lookup: Dict[str, Tuple[float, float, float, float]],
    output_folder: Path,
    temp_folder: Path,
) -> Optional[Path]:
    lon, lat, date_str = sample.longitude, sample.latitude, sample.date
    if lat < -60:
        print(f"Skipping Antarctica sample at ({lon:.3f}, {lat:.3f})")
        return None

    bbox = get_patch_bbox(lon, lat, patch_size_pix, 0.004)
    urls = search_nasa_cmr(collection_id, date_str, bbox)
    if not urls:
        print(f"No Black Marble file found for {date_str} at ({lon:.3f}, {lat:.3f})")
        return None

    h5_url = urls[0]
    temp_folder.mkdir(parents=True, exist_ok=True)
    h5_path = temp_folder / os.path.basename(h5_url)
    try:
        download_file(h5_url, h5_path, token)
    except requests.HTTPError as exc:
        print(f"Failed to download {h5_url}: {exc}")
        return None

    tile_id_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not tile_id_match:
        print(f"Could not determine tile ID from {h5_path.name}")
        h5_path.unlink(missing_ok=True)
        return None

    tile_id = tile_id_match.group()
    if tile_id not in tile_bounds_lookup:
        print(f"Tile {tile_id} not found in tile index")
        h5_path.unlink(missing_ok=True)
        return None

    tif_path = temp_folder / f"{h5_path.stem}.tif"
    try:
        tif_path, tif_res_deg = convert_h5_to_geotiff(h5_path, tile_bounds_lookup[tile_id], tif_path)
    except DownloadError as exc:
        print(exc)
        h5_path.unlink(missing_ok=True)
        return None

    patch, patch_meta = extract_patch_from_geotiff(tif_path, lon, lat, patch_size_pix, tif_res_deg)
    if patch.shape[0] < patch_size_pix or patch.shape[1] < patch_size_pix:
        print(
            f"Patch at ({lon:.3f}, {lat:.3f}) on {date_str} is too small ({patch.shape[0]}x{patch.shape[1]})."
        )
        h5_path.unlink(missing_ok=True)
        tif_path.unlink(missing_ok=True)
        return None

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f"BM_patch_{date_str}_{lon:.3f}_{lat:.3f}.tif"
    with rasterio.open(out_path, "w", **patch_meta) as dst:
        dst.write(patch, 1)

    h5_path.unlink(missing_ok=True)
    tif_path.unlink(missing_ok=True)
    print(f"Saved patch: {out_path}")
    return out_path


def process_samples_parallel(
    samples: Iterable[SampleRequest],
    *,
    patch_size_pix: int = DEFAULT_PATCH_SIZE,
    collection_id: str = DEFAULT_COLLECTION_ID,
    token: str,
    tile_shapefile_path: Path,
    output_folder: Path,
    temp_folder: Path,
    max_workers: int = 4,
) -> List[Path]:
    tile_df = gpd.read_file(tile_shapefile_path)
    tile_bounds_lookup: Dict[str, Tuple[float, float, float, float]] = {
        str(row["TileID"]): tuple(row.geometry.bounds)  # type: ignore[index]
        for _, row in tile_df.iterrows()
    }

    saved_paths: List[Path] = []
    samples_list = list(samples)
    temp_folder.mkdir(parents=True, exist_ok=True)

    def worker(sample_dict: SampleRequest) -> Optional[Path]:
        return process_single_sample(
            sample_dict,
            patch_size_pix=patch_size_pix,
            collection_id=collection_id,
            token=token,
            tile_bounds_lookup=tile_bounds_lookup,
            output_folder=output_folder,
            temp_folder=temp_folder,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, sample) for sample in samples_list]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                saved_paths.append(result)

    shutil.rmtree(temp_folder, ignore_errors=True)
    return saved_paths


def wait_for_file_release(path: Path, timeout: int = 10) -> None:
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
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Other error on {key} (attempt {attempt + 1}/{max_retries}): {exc}")
        time.sleep(2)
    print(f"Failed to download after {max_retries} attempts: {key}")
    return False


def group_by_f_number(file_keys: Sequence[Tuple[str, Optional[str]]]) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    groups: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    for vis_key, _ in file_keys:
        base = os.path.basename(vis_key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        groups[f_number].append((vis_key, None))
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
    file_keys: Sequence[Tuple[str, Optional[str]]],
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
    *,
    min_valid_fraction: float = 0.10,
) -> List[Path]:
    bm_files_saved: List[Path] = []
    bm_patch_name = bm_patch_path.name
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    groups = group_by_f_number(file_keys)
    for f_number, scene_keys in groups.items():
        best_valid_pixels = 0
        best_vis_file: Optional[Path] = None
        best_vis_patch: Optional[np.ndarray] = None
        for vis_key, _ in scene_keys:
            vis_file = temp_dir / os.path.basename(vis_key)
            if not vis_file.exists():
                print(f"Downloading {vis_key} ...")
                if not safe_download(s3, bucket_name, vis_key, vis_file):
                    continue
            try:
                vis_patch = reproject_to_bm_grid(vis_file, bm_profile)
                valid_pixels = int(np.sum(~np.isnan(vis_patch)))
                median_val = float(np.nanmedian(vis_patch)) if valid_pixels > 0 else 0.0
                if valid_pixels > best_valid_pixels and median_val > 1:
                    best_valid_pixels = valid_pixels
                    best_vis_file = vis_file
                    best_vis_patch = vis_patch
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Error processing {vis_file}: {exc}")
                continue
        if best_vis_file is None or best_vis_patch is None:
            print(f"No good patch found for {f_number} in {bm_patch_name}")
            continue
        valid_fraction = best_valid_pixels / (bm_shape[0] * bm_shape[1])
        if valid_fraction < min_valid_fraction:
            print(
                f"Skipping {f_number} for {bm_patch_name}: only {valid_fraction:.2%} valid pixels"
            )
            continue
        out_fname = f"{f_number}_{best_vis_file.stem}_match_{bm_patch_name.replace('.tif', '')}.tif"
        out_path = dmsp_out_dir / out_fname
        out_profile = bm_profile.copy()
        out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
        dmsp_out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(best_vis_patch.astype(np.float32), 1)
        bm_files_saved.append(out_path)
    return bm_files_saved


def match_dmsp_to_bm_patches(
    bm_patch_dir: Path,
    dmsp_out_dir: Path,
    temp_dir: Path,
    *,
    max_workers: int = 4,
) -> List[Path]:
    bm_patch_paths = sorted(p for p in bm_patch_dir.glob("*.tif"))
    if not bm_patch_paths:
        print("No Black Marble patches available for DMSP matching.")
        return []

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"

    def worker(bm_patch_path: Path) -> List[Path]:
        bm_date = bm_patch_path.name.split("_")[2]
        dmsp_date_str = bm_date.replace("-", "")
        file_keys: List[Tuple[str, Optional[str]]] = []
        satellites = [f"F{n}" for n in range(10, 19)]
        for sat in satellites:
            prefix = f"{sat}{dmsp_date_str[:4]}/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if dmsp_date_str in key and key.endswith(".vis.co.tif"):
                        file_keys.append((key, None))
        worker_temp_dir = temp_dir / bm_patch_path.stem
        worker_temp_dir.mkdir(parents=True, exist_ok=True)
        saved = process_bm_patch_for_best_fnumber(
            bm_patch_path,
            file_keys,
            s3,
            bucket_name,
            worker_temp_dir,
            dmsp_out_dir,
        )
        for f in worker_temp_dir.glob("*"):
            f.unlink(missing_ok=True)
        worker_temp_dir.rmdir()
        return saved

    saved_files: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, path) for path in bm_patch_paths]
        for future in concurrent.futures.as_completed(futures):
            saved_files.extend(future.result())

    shutil.rmtree(temp_dir, ignore_errors=True)
    return saved_files


def build_patch_manifest(
    bm_patch_dir: Path,
    dmsp_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    bm_files = sorted(p for p in bm_patch_dir.glob("*.tif"))
    dmsp_files = sorted(p for p in dmsp_dir.glob("*.tif"))
    records: List[Dict[str, str]] = []

    dmsp_lookup = {p.name: p for p in dmsp_files}
    for bm_file in bm_files:
        key = bm_file.stem
        matching_dmsp = [name for name in dmsp_lookup if key in name]
        if not matching_dmsp:
            continue
        for dmsp_name in matching_dmsp:
            records.append(
                {
                    "bm_patch": bm_file.name,
                    "dmsp_patch": dmsp_name,
                }
            )

    manifest = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote manifest with {len(manifest)} pairs to {manifest_path}")
    return manifest


__all__ = [
    "SampleRequest",
    "assign_dates_to_samples",
    "build_patch_manifest",
    "generate_population_balanced_samples",
    "list_dmsp_dates",
    "load_nasa_token",
    "match_dmsp_to_bm_patches",
    "process_samples_parallel",
]
