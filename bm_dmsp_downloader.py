"""Utilities for sampling LandScan population, downloading NASA Black Marble patches,
matching DMSP scenes, and exporting paired GeoTIFFs.

This module reorganises the original Colab notebook logic into composable functions that:

* draw population-balanced geographic samples from the LandScan 2012 raster while
  excluding Antarctica and optionally exporting a CSV manifest;
* download 1000×1000 pixel Black Marble (BM) patches for the sampled locations; and
* fetch and reproject matching DMSP scenes to align with the BM patches, saving only
  GeoTIFF outputs and a manifest of the matched pairs.

Run ``python bm_dmsp_downloader.py --help`` for usage instructions.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import os
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

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
from rasterio.features import rasterize
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


BM_COLLECTION_ID = "C3365931269-LAADS"
BM_PATCH_DIR = Path("Raw_NL_Data/BM data")
DMSP_PATCH_DIR = Path("Raw_NL_Data/DMSP data")
DMSP_TEMP_DIR = Path("DMSP_Raw_Temp")
BM_TEMP_DIR = Path("temp_dl")
LANDSCAN_RASTER = Path("Data/Global_2012/landscan-global-2012.tif")
WORLD_SHAPEFILE = Path("Data/World_Countries/World_Countries_Generalized.shp")
BM_TILE_SHP = Path("Data/Black_Marble_IDs/Black_Marble_World_tiles.shp")
DEFAULT_PATCH_SIZE = 1000
DEFAULT_WORKERS = 4


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_nasa_token() -> str:
    """Load the NASA token from environment variables.

    The original notebook relied on ``python-dotenv``; we keep that behaviour so
    that ``.env`` files continue to work locally.
    """

    if load_dotenv is not None:
        load_dotenv()

    token = os.getenv("NASA_TOKEN")
    if not token:
        raise RuntimeError(
            "NASA_TOKEN environment variable is not set. Create a .env file "
            "with your NASA Earthdata token before running downloads."
        )
    return token


# ---------------------------------------------------------------------------
# LandScan sampling
# ---------------------------------------------------------------------------

@dataclass
class LandscanSampleConfig:
    downsample_factor: int = 4
    log_bin_edges: Sequence[float] = tuple(range(0, 11)) + (float("inf"),)
    samples_per_bin: int = 200
    random_seed: int = 2024
    min_latitude: float = -60.0  # exclude Antarctica and extreme polar region
    export_plot: Optional[Path] = Path("plots/population_bins_sampled.pdf")


def _prepare_landscan_arrays(config: LandscanSampleConfig):
    with rasterio.open(LANDSCAN_RASTER) as src:
        scale = config.downsample_factor
        new_height = src.height // scale
        new_width = src.width // scale
        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=rasterio.enums.Resampling.bilinear,
        )
        nodata = src.nodata
        transform = src.transform

    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    gdf = gpd.read_file(WORLD_SHAPEFILE)
    antarctica = gdf[gdf["FID"] == 8]

    with rasterio.open(LANDSCAN_RASTER) as src:
        downsampled_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height),
        )

    antarctica_mask = rasterize(
        [(geom, 1) for geom in antarctica.geometry],
        out_shape=(new_height, new_width),
        transform=downsampled_transform,
        fill=0,
        dtype="uint8",
    )

    with rasterio.open(LANDSCAN_RASTER) as src:
        ys = np.linspace(src.bounds.top, src.bounds.bottom, new_height)
    lat_mask = np.repeat(ys[:, np.newaxis], new_width, axis=1)

    south_polar_mask = lat_mask < config.min_latitude
    combined_mask = (antarctica_mask == 1) | south_polar_mask
    data[combined_mask] = np.nan

    return data, transform, downsampled_transform, antarctica_mask


def sample_landscan_population(config: LandscanSampleConfig = LandscanSampleConfig()) -> pd.DataFrame:
    """Sample population-balanced coordinates from the LandScan raster.

    Returns a DataFrame with longitude/latitude, population, and log population
    bin labels. The random seed ensures reproducibility.
    """

    np.random.seed(config.random_seed)
    data, _, downsampled_transform, _ = _prepare_landscan_arrays(config)

    log_data = np.log1p(data)
    valid_mask = (~np.isnan(data)) & (data >= 0)
    rows, cols = np.where(valid_mask)
    logs = log_data[rows, cols]
    pops = data[rows, cols]

    bin_edges = list(config.log_bin_edges)
    bin_labels = [
        f"{int(bin_edges[i])}–{int(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 2)
    ] + [f"{int(bin_edges[-2])}+"]
    bins = np.digitize(logs, bin_edges) - 1

    samples = []
    for idx, label in enumerate(bin_labels):
        candidate_indices = np.where(bins == idx)[0]
        if len(candidate_indices) == 0:
            continue
        picks = np.random.choice(
            candidate_indices,
            size=min(config.samples_per_bin, len(candidate_indices)),
            replace=False,
        )
        for pick in picks:
            row, col = rows[pick], cols[pick]
            lon, lat = rasterio.transform.xy(downsampled_transform, row, col)
            samples.append(
                {
                    "Bin": label,
                    "Longitude": float(lon),
                    "Latitude": float(lat),
                    "Population": float(pops[pick]),
                    "log(pop+1)": float(logs[pick]),
                }
            )

    df = pd.DataFrame(samples)
    df.sort_values(["Bin", "Population"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def export_samples(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def maybe_plot_samples(df: pd.DataFrame, config: LandscanSampleConfig) -> None:
    if config.export_plot is None:
        return
    import matplotlib.pyplot as plt  # Local import avoids heavy dependency during tests

    data, _, downsampled_transform, _ = _prepare_landscan_arrays(config)
    extent = (
        downsampled_transform[2],
        downsampled_transform[2] + downsampled_transform[0] * data.shape[1],
        downsampled_transform[5] + downsampled_transform[4] * data.shape[0],
        downsampled_transform[5],
    )
    plt.figure(figsize=(14, 7))
    plt.imshow(np.log1p(data), cmap="viridis", extent=extent, origin="upper")
    plt.title("LandScan 2012 samples by log population bin")
    plt.axis("off")
    plt.colorbar(label="log(population + 1)")

    unique_bins = sorted(df["Bin"].unique())
    color_map = plt.cm.get_cmap("tab10", len(unique_bins))
    bin_to_color = {bin_label: color_map(i) for i, bin_label in enumerate(unique_bins)}
    plt.scatter(
        df["Longitude"],
        df["Latitude"],
        s=15,
        c=[bin_to_color[label] for label in df["Bin"]],
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=bin_to_color[label], markersize=6)
        for label in unique_bins
    ]
    plt.legend(handles=handles, loc="lower left", frameon=False, title="Bin")

    config.export_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.export_plot, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Black Marble download helpers
# ---------------------------------------------------------------------------


def get_patch_bbox(lon: float, lat: float, patch_size_pix: int, tif_res_deg: float) -> List[float]:
    half_deg = (patch_size_pix * tif_res_deg) / 2
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def enforce_patch_size(patch: np.ndarray, patch_size_pix: int) -> np.ndarray:
    patch = patch[:patch_size_pix, :patch_size_pix]
    pad_h = max(0, patch_size_pix - patch.shape[0])
    pad_w = max(0, patch_size_pix - patch.shape[1])
    if pad_h or pad_w:
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
    response = requests.get(cmr_search_url, params=params)
    response.raise_for_status()
    granules = response.json().get("feed", {}).get("entry", [])
    links = []
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                links.append(href)
    return links


def download_file(url: str, dest: Path, token: str) -> Path:
    if dest.exists():
        return dest
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def convert_h5_to_geotiff(h5_path: Path, tile_shapefile: gpd.GeoDataFrame, tif_path: Path) -> float:
    with h5py.File(h5_path, "r") as h5_file:
        dataset_path = (
            "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        )
        if dataset_path not in h5_file:
            raise KeyError(f"Dataset {dataset_path} not found in {h5_path}")
        ntl_data = h5_file[dataset_path][...]

    tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not tile_match:
        raise ValueError(f"Could not extract tile ID from {h5_path}")
    tile_id = tile_match.group()
    row = tile_shapefile[tile_shapefile["TileID"] == tile_id]
    if row.empty:
        raise ValueError(f"Tile ID {tile_id} not found in shapefile")
    left, bottom, right, top = row.total_bounds

    height, width = ntl_data.shape
    transform = rio_from_bounds(left, bottom, right, top, width, height)

    tif_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        tif_path,
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

    return (right - left) / width


@dataclass
class SampleRequest:
    longitude: float
    latitude: float
    date: str  # YYYY-MM-DD


class RequestValidationError(Exception):
    pass


def validate_sample_request(request: SampleRequest) -> None:
    try:
        datetime.strptime(request.date, "%Y-%m-%d")
    except ValueError as exc:
        raise RequestValidationError(f"Invalid date format for {request.date}; expected YYYY-MM-DD") from exc

    if not -180.0 <= request.longitude <= 180.0:
        raise RequestValidationError("Longitude must be within [-180, 180]")
    if not -90.0 <= request.latitude <= 90.0:
        raise RequestValidationError("Latitude must be within [-90, 90]")
    if request.latitude < -60.0:
        raise RequestValidationError("Sample latitude below -60° is not supported")


def process_single_sample(
    request: SampleRequest,
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapefile: gpd.GeoDataFrame,
    output_folder: Path,
    temp_folder: Path,
    search_resolution_deg: float = 0.004,
) -> Optional[Path]:
    validate_sample_request(request)

    bbox = get_patch_bbox(request.longitude, request.latitude, patch_size_pix, search_resolution_deg)
    if bbox[1] < -60:
        bbox[1] = -60
    if bbox[3] < -60:
        return None

    urls = search_nasa_cmr(collection_id, request.date, bbox)
    if not urls:
        return None

    h5_url = urls[0]
    temp_folder.mkdir(parents=True, exist_ok=True)
    h5_path = temp_folder / Path(h5_url).name
    download_file(h5_url, h5_path, token)

    tif_path = h5_path.with_suffix(".tif")
    tif_res_deg = convert_h5_to_geotiff(h5_path, tile_shapefile, tif_path)

    with rasterio.open(tif_path) as src:
        bbox = get_patch_bbox(request.longitude, request.latitude, patch_size_pix, tif_res_deg)
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window)
        patch_meta = src.meta.copy()
        patch_meta.update(
            {
                "height": patch.shape[0],
                "width": patch.shape[1],
                "transform": src.window_transform(window),
            }
        )

    patch = enforce_patch_size(patch, patch_size_pix)
    if patch.shape[0] < patch_size_pix or patch.shape[1] < patch_size_pix:
        return None

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / f"BM_patch_{request.date}_{request.longitude:.3f}_{request.latitude:.3f}.tif"
    with rasterio.open(out_path, "w", **patch_meta) as dst:
        dst.write(patch, 1)

    gc.collect()
    h5_path.unlink(missing_ok=True)
    tif_path.unlink(missing_ok=True)
    return out_path


def process_samples_parallel(
    requests_list: Sequence[SampleRequest],
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapefile_path: Path,
    output_folder: Path = BM_PATCH_DIR,
    temp_folder: Path = BM_TEMP_DIR,
    max_workers: int = DEFAULT_WORKERS,
) -> List[Path]:
    tile_shapefile = gpd.read_file(tile_shapefile_path)
    tasks = [
        (req, patch_size_pix, collection_id, token, tile_shapefile, output_folder, temp_folder)
        for req in requests_list
    ]

    results: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {
            executor.submit(process_single_sample, *task): task[0]
            for task in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_request),
            total=len(future_to_request),
            desc="BM patches",
            unit="patch",
        ):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                request = future_to_request[future]
                print(f"Error processing {request}: {exc}")
    return results


# ---------------------------------------------------------------------------
# DMSP matching
# ---------------------------------------------------------------------------

def list_dmsp_dates(min_date: datetime = datetime(2012, 1, 20)) -> List[str]:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"
    prefix = "F"

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    all_dates = set()
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".vis.co.tif"):
                continue
            name = os.path.basename(key)
            if len(name) < 11:
                continue
            date_str = name[3:11]
            try:
                group_date = datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                continue
            if group_date >= min_date:
                all_dates.add(date_str)
    return sorted(all_dates)


def assign_dates_to_samples(df: pd.DataFrame, random_seed: int = 13492) -> pd.DataFrame:
    all_dates = list_dmsp_dates()
    random.seed(random_seed)
    sampled_dates = random.choices(all_dates, k=len(df))
    df = df.copy()
    df["date"] = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sampled_dates]
    return df


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
            print(f"EndpointConnectionError on {key} (attempt {attempt+1}/{max_retries}): {exc}")
        except botocore.exceptions.ClientError as exc:
            print(f"ClientError on {key} (attempt {attempt+1}/{max_retries}): {exc}")
        except Exception as exc:
            print(f"Other error on {key} (attempt {attempt+1}/{max_retries}): {exc}")
        time.sleep(2)
    print(f"Failed to download after {max_retries} attempts: {key}")
    return False


def group_by_f_number(file_keys: Sequence[str]) -> dict:
    groups: dict[str, List[str]] = defaultdict(list)
    for key in file_keys:
        base = os.path.basename(key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        groups[f_number].append(key)
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


@dataclass
class DmspMatch:
    bm_patch: Path
    dmsp_patch: Path
    f_number: str
    valid_fraction: float


def process_bm_patch_for_best_fnumber(
    bm_patch_path: Path,
    file_keys: Sequence[str],
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
    min_valid_fraction: float = 0.10,
) -> List[DmspMatch]:
    matches: List[DmspMatch] = []
    bm_patch_name = bm_patch_path.name
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)

    groups = group_by_f_number(file_keys)
    for f_number, scene_keys in groups.items():
        best_valid_pixels = 0
        best_vis_file: Optional[Path] = None
        best_vis_patch: Optional[np.ndarray] = None

        for vis_key in scene_keys:
            vis_file = temp_dir / os.path.basename(vis_key)
            if not vis_file.exists():
                if not safe_download(s3, bucket_name, vis_key, vis_file):
                    continue
            try:
                vis_patch = reproject_to_bm_grid(vis_file, bm_profile)
            except Exception as exc:
                print(f"Error processing {vis_file}: {exc}")
                continue
            valid_pixels = int(np.sum(~np.isnan(vis_patch)))
            median_val = float(np.nanmedian(vis_patch)) if valid_pixels > 0 else 0.0
            if valid_pixels > best_valid_pixels and median_val > 1:
                best_valid_pixels = valid_pixels
                best_vis_file = vis_file
                best_vis_patch = vis_patch

        if best_vis_file and best_valid_pixels > 0:
            valid_fraction = best_valid_pixels / float(bm_shape[0] * bm_shape[1])
            if valid_fraction < min_valid_fraction:
                continue
            out_fname = (
                f"{f_number}_{best_vis_file.stem}_match_{bm_patch_name.replace('.tif', '')}.tif"
            )
            out_path = dmsp_out_dir / out_fname
            out_profile = bm_profile.copy()
            out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
            dmsp_out_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(best_vis_patch.astype(np.float32), 1)
            matches.append(
                DmspMatch(
                    bm_patch=bm_patch_path,
                    dmsp_patch=out_path,
                    f_number=f_number,
                    valid_fraction=valid_fraction,
                )
            )
    for item in temp_dir.glob("*"):
        try:
            item.unlink()
        except Exception as exc:
            print(f"Error deleting {item}: {exc}")
    return matches


def match_dmsp_to_bm_patches(
    bm_patch_dir: Path = BM_PATCH_DIR,
    dmsp_out_dir: Path = DMSP_PATCH_DIR,
    temp_dir: Path = DMSP_TEMP_DIR,
    max_workers: int = DEFAULT_WORKERS,
    manifest_path: Optional[Path] = None,
) -> List[DmspMatch]:
    bm_patch_paths = sorted(bm_patch_dir.glob("*.tif"))
    if not bm_patch_paths:
        print("No BM patches found to match.")
        return []

    temp_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"

    def gather_dmsp_keys(bm_patch_path: Path) -> List[str]:
        bm_date = bm_patch_path.name.split("_")[2]
        dmsp_date_str = bm_date.replace("-", "")
        keys: List[str] = []
        satellites = [f"F{n}" for n in range(10, 19)]
        for sat in satellites:
            prefix = f"{sat}{dmsp_date_str[:4]}/"
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if dmsp_date_str in key and key.endswith(".vis.co.tif"):
                        keys.append(key)
        return keys

    matches: List[DmspMatch] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for bm_patch_path in bm_patch_paths:
            file_keys = gather_dmsp_keys(bm_patch_path)
            futures.append(
                executor.submit(
                    process_bm_patch_for_best_fnumber,
                    bm_patch_path,
                    file_keys,
                    s3,
                    bucket_name,
                    temp_dir,
                    dmsp_out_dir,
                )
            )
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="DMSP matches",
            unit="bm_patch",
        ):
            try:
                matches.extend(future.result())
            except Exception as exc:
                print(f"Error matching DMSP scenes: {exc}")

    if manifest_path and matches:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df = pd.DataFrame(
            [
                {
                    "bm_patch": match.bm_patch.as_posix(),
                    "dmsp_patch": match.dmsp_patch.as_posix(),
                    "f_number": match.f_number,
                    "valid_fraction": match.valid_fraction,
                }
                for match in matches
            ]
        )
        manifest_df.sort_values(["bm_patch", "f_number"], inplace=True)
        manifest_df.to_csv(manifest_path, index=False)

    return matches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_requests_from_dataframe(df: pd.DataFrame) -> List[SampleRequest]:
    requests_list = []
    for _, row in df.iterrows():
        requests_list.append(
            SampleRequest(
                longitude=float(row["Longitude"]),
                latitude=float(row["Latitude"]),
                date=str(row["date"]),
            )
        )
    return requests_list


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BM/DMSP downloader pipeline")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Patch size in pixels for BM and DMSP GeoTIFFs (default: 1000)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        help="Existing CSV of samples to use (skip LandScan sampling if provided)",
    )
    parser.add_argument(
        "--export-samples",
        type=Path,
        default=Path("Data/sample_requests.csv"),
        help="Where to export the sampled coordinate CSV",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Raw_NL_Data/dmsp_bm_manifest.csv"),
        help="Where to save the DMSP/BM pairing manifest",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading BM patches (assume they already exist)",
    )
    parser.add_argument(
        "--skip-dmsp",
        action="store_true",
        help="Skip DMSP matching",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip exporting the LandScan sampling plot",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.sample_csv and args.sample_csv.exists():
        samples_df = pd.read_csv(args.sample_csv)
    else:
        config = LandscanSampleConfig(export_plot=None if args.no_plot else LandscanSampleConfig().export_plot)
        samples_df = sample_landscan_population(config)
        export_samples(samples_df, args.export_samples)
        if not args.no_plot:
            maybe_plot_samples(samples_df, config)

    if "date" not in samples_df.columns:
        samples_df = assign_dates_to_samples(samples_df)
        export_samples(samples_df, args.export_samples)

    if not args.skip_download:
        token = load_nasa_token()
        requests_list = build_requests_from_dataframe(samples_df)
        process_samples_parallel(
            requests_list=requests_list,
            patch_size_pix=args.patch_size,
            collection_id=BM_COLLECTION_ID,
            token=token,
            tile_shapefile_path=BM_TILE_SHP,
            output_folder=BM_PATCH_DIR,
            temp_folder=BM_TEMP_DIR,
            max_workers=args.max_workers,
        )

    if not args.skip_dmsp:
        match_dmsp_to_bm_patches(
            bm_patch_dir=BM_PATCH_DIR,
            dmsp_out_dir=DMSP_PATCH_DIR,
            temp_dir=DMSP_TEMP_DIR,
            max_workers=args.max_workers,
            manifest_path=args.manifest,
        )

    for folder in (BM_TEMP_DIR, DMSP_TEMP_DIR):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
