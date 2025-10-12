"""Simplified pipeline for sampling LandScan and downloading paired BM/DMSP GeoTIFFs."""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Sequence

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import botocore
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling as WarpResampling
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

LOGGER = logging.getLogger(__name__)

NASA_TOKEN_ENV = "NASA_TOKEN"
DEFAULT_PATCH_SIZE = 1000
DEFAULT_COLLECTION_ID = "C3365931269-LAADS"
DEFAULT_TILE_SHAPEFILE = Path("Data/Black_Marble_IDs/Black_Marble_World_tiles.shp")
LANDSCAN_RASTER = Path("Data/Global_2012/landscan-global-2012.tif")
COUNTRIES_SHP = Path("Data/World_Countries/World_Countries_Generalized.shp")
BM_OUTPUT_DIR = Path("Raw_NL_Data/BM data")
DMSP_OUTPUT_DIR = Path("Raw_NL_Data/DMSP data")
PLOTS_DIR = Path("plots")
DEFAULT_LOCATIONS_CSV = Path("sampled_locations.csv")
DEFAULT_MANIFEST = Path("Raw_NL_Data/bm_dmsp_pairs.csv")


def infer_country_column(gdf: gpd.GeoDataFrame, country_column: str | None = None) -> str:
    if country_column and country_column in gdf.columns:
        return country_column
    preferred = [
        "COUNTRY",
        "COUNTRY_NA",
        "NAME",
        "NAME_EN",
        "ADMIN",
        "CNTRY_NAME",
        "SOVEREIGNT",
    ]
    for candidate in preferred:
        if candidate in gdf.columns:
            return candidate
    object_columns = [col for col in gdf.columns if gdf[col].dtype == object]
    if not object_columns:
        raise ValueError("Unable to infer country column from shapefile")
    return object_columns[0]


def load_nasa_token() -> str:
    token = os.getenv(NASA_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            "NASA_TOKEN environment variable is not set. Create a .env file (see .env.example) or set the variable before running downloads."
        )
    return token


def sample_landscan_population(
    scale: int = 4,
    samples_per_bin: int = 200,
    random_seed: int = 2024,
    min_valid_lat: float = -60.0,
    plots_dir: Path | None = PLOTS_DIR,
    output_csv: Path | None = DEFAULT_LOCATIONS_CSV,
    countries: Sequence[str] | None = None,
    country_column: str | None = None,
) -> pd.DataFrame:
    """Downsample LandScan, balance by log population bins, and optionally persist a CSV."""
    if not LANDSCAN_RASTER.exists():
        raise FileNotFoundError(f"LandScan raster not found at {LANDSCAN_RASTER}")
    if not COUNTRIES_SHP.exists():
        raise FileNotFoundError(f"World countries shapefile not found at {COUNTRIES_SHP}")

    LOGGER.info("Loading LandScan raster %s", LANDSCAN_RASTER)
    with rasterio.open(LANDSCAN_RASTER) as src:
        new_height = src.height // scale
        new_width = src.width // scale
        array = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.bilinear,
        )
        nodata = src.nodata
        downsampled_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height),
        )
        ys = np.linspace(src.bounds.top, src.bounds.bottom, new_height)
        scale_x = src.width / new_width
        scale_y = src.height / new_height

    if nodata is not None:
        array = np.where(array == nodata, np.nan, array)

    LOGGER.info("Masking Antarctica and southern polar regions")
    gdf = gpd.read_file(COUNTRIES_SHP).copy()
    antarctica = gdf[gdf.get("FID") == 8]
    antarctica_mask = rasterize(
        [(geom, 1) for geom in antarctica.geometry],
        out_shape=array.shape,
        transform=downsampled_transform,
        fill=0,
        dtype="uint8",
    )

    include_mask = None
    if countries:
        column_name = infer_country_column(gdf, country_column)
        target = {c.strip().lower() for c in countries}
        gdf["__country_name"] = gdf[column_name].astype(str).str.strip().str.lower()
        selected = gdf[gdf["__country_name"].isin(target)]
        if selected.empty:
            raise ValueError(
                "None of the requested countries were found in the shapefile. "
                f"Requested: {sorted(target)}"
            )
        LOGGER.info(
            "Restricting sampling to %s", ", ".join(sorted(selected[column_name].unique()))
        )
        include_mask = rasterize(
            [(geom, 1) for geom in selected.geometry],
            out_shape=array.shape,
            transform=downsampled_transform,
            fill=0,
            dtype="uint8",
        )
        gdf = gdf.drop(columns=["__country_name"], errors="ignore")

    lat_mask = np.repeat(ys[:, np.newaxis], array.shape[1], axis=1)
    combined_mask = (antarctica_mask == 1) | (lat_mask < min_valid_lat)
    if include_mask is not None:
        combined_mask |= include_mask == 0
    array = np.where(combined_mask, np.nan, array)

    log_array = np.log1p(array)
    valid_mask = (~np.isnan(array)) & (array >= 0)
    rows, cols = np.where(valid_mask)
    logp = log_array[rows, cols]
    pops = array[rows, cols]

    bin_edges = list(range(0, 11)) + [np.inf]
    bin_labels = [f"{i}–{i + 1}" for i in range(0, 10)] + ["10+"]
    bins = np.digitize(logp, bin_edges) - 1

    np.random.seed(random_seed)
    sampled = []
    for b, label in enumerate(bin_labels):
        idxs = np.where(bins == b)[0]
        if len(idxs) == 0:
            continue
        picks = np.random.choice(idxs, size=min(samples_per_bin, len(idxs)), replace=False)
        for pick in picks:
            sampled.append((rows[pick], cols[pick], logp[pick], pops[pick], label))

    with rasterio.open(LANDSCAN_RASTER) as src:
        coords = [src.xy(int(r * scale_y), int(c * scale_x)) for r, c, *_ in sampled]

    df = pd.DataFrame(
        {
            "Bin": [x[4] for x in sampled],
            "Longitude": [c[0] for c in coords],
            "Latitude": [c[1] for c in coords],
            "Population": [int(round(x[3])) for x in sampled],
            "log(pop+1)": [float(x[2]) for x in sampled],
        }
    )

    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(14, 7))
        plt.imshow(log_array, cmap="viridis")
        plt.title("LandScan 2012: Samples per Integer Log Population Bin")
        plt.axis("off")
        plt.colorbar(label="log(population + 1)")
        for (r, c, *_), label in zip(sampled, df["Bin"]):
            plt.plot(c, r, "o", markersize=4)
        plt.savefig(plots_dir / "population_bins_sampled.pdf", bbox_inches="tight")
        plt.close()

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        LOGGER.info("Wrote sampled locations to %s", output_csv)

    return df


def list_dmsp_dates(min_date: pd.Timestamp | None = None) -> List[str]:
    min_dt = min_date or pd.Timestamp(2012, 1, 20)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"
    prefix = "F"
    paginator = s3.get_paginator("list_objects_v2")
    all_dates: set[str] = set()
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            file_key = obj["Key"]
            if not file_key.endswith(".vis.co.tif"):
                continue
            fname = os.path.basename(file_key)
            if len(fname) < 11:
                continue
            date_str = fname[3:11]
            try:
                group_date = pd.to_datetime(date_str, format="%Y%m%d")
            except ValueError:
                continue
            if group_date >= min_dt:
                all_dates.add(date_str)
    return sorted(all_dates)


def assign_random_dates(df: pd.DataFrame, dmsp_dates: Sequence[str], seed: int = 13492) -> pd.DataFrame:
    if "date" in df.columns:
        return df.copy()
    if not dmsp_dates:
        raise ValueError("No DMSP dates available to assign")
    rng = random.Random(seed)
    sampled_dates = [rng.choice(dmsp_dates) for _ in range(len(df))]
    df_with_dates = df.copy()
    df_with_dates["date"] = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sampled_dates]
    return df_with_dates


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
    params = {
        "collection_concept_id": collection_id,
        "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
        "bounding_box": ",".join(map(str, bbox)),
        "page_size": 50,
    }
    response = requests.get("https://cmr.earthdata.nasa.gov/search/granules.json", params=params)
    response.raise_for_status()
    h5_links: List[str] = []
    granules = response.json().get("feed", {}).get("entry", [])
    for granule in granules:
        for link in granule.get("links", []):
            href = link.get("href", "")
            if href.startswith("https") and href.endswith(".h5"):
                h5_links.append(href)
    return h5_links


def extract_patch_from_geotiff(tif_path: Path, lon: float, lat: float, patch_size_pix: int) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif_path) as src:
        tif_res_deg = src.transform[0]
        bbox = get_patch_bbox(lon, lat, patch_size_pix, tif_res_deg)
        window = from_bounds(*bbox, src.transform)
        patch = src.read(1, window=window)
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
    sample: dict,
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapefile: gpd.GeoDataFrame,
    output_folder: Path,
    temp_folder: Path,
) -> tuple[str, Path | None]:
    lon, lat, date_str = sample["Longitude"], sample["Latitude"], sample["date"]
    if lat < -60:
        return (f"Skipping Antarctica sample at ({lon:.3f}, {lat:.3f})", None)

    bbox = get_patch_bbox(lon, lat, patch_size_pix, tif_res_deg=0.004)
    if bbox[1] < -60:
        bbox[1] = -60
    if bbox[3] < -60:
        return (f"Skipping search below -60°S for bbox {bbox}", None)

    urls = search_nasa_cmr(collection_id, date_str, bbox)
    if not urls:
        return (f"No Black Marble file found for {date_str} at ({lon:.3f}, {lat:.3f})", None)

    h5_url = urls[0]
    h5_path = temp_folder / os.path.basename(h5_url)
    if not h5_path.exists():
        response = requests.get(h5_url, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        h5_path.write_bytes(response.content)

    tif_path = h5_path.with_suffix(".tif")
    with h5py.File(h5_path, "r") as f:
        dataset_path = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
        if dataset_path not in f:
            h5_path.unlink(missing_ok=True)
            return (f"Dataset not found in {h5_path.name}", None)
        ntl_data = f[dataset_path][...]
        tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
        if not tile_match:
            h5_path.unlink(missing_ok=True)
            return (f"Could not determine tile ID for {h5_path.name}", None)
        tile_id = tile_match.group()
        bounds_row = tile_shapefile[tile_shapefile["TileID"] == tile_id]
        if bounds_row.empty:
            h5_path.unlink(missing_ok=True)
            return (f"Tile ID {tile_id} not found in shapefile", None)
        left, bottom, right, top = bounds_row.total_bounds

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
    out_path = output_folder / f"BM_patch_{date_str}_{lon:.3f}_{lat:.3f}.tif"
    with rasterio.open(out_path, "w", **patch_meta) as dst:
        dst.write(patch, 1)

    h5_path.unlink(missing_ok=True)
    tif_path.unlink(missing_ok=True)
    return (f"Saved patch: {out_path}", out_path)


def process_samples_parallel(
    sample_list: Sequence[dict],
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapefile_path: Path,
    output_folder: Path,
    temp_folder: Path,
    max_workers: int = 4,
) -> List[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder.mkdir(parents=True, exist_ok=True)
    tile_shapefile = gpd.read_file(tile_shapefile_path)

    results: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_sample,
                sample,
                patch_size_pix,
                collection_id,
                token,
                tile_shapefile,
                output_folder,
                temp_folder,
            )
            for sample in sample_list
        ]
        for future in concurrent.futures.as_completed(futures):
            message, path = future.result()
            LOGGER.info(message)
            if path is not None:
                results.append(path)

    shutil.rmtree(temp_folder, ignore_errors=True)
    return results


def wait_for_file_release(path: Path, timeout: float = 10.0) -> None:
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


def safe_download(s3, bucket: str, key: str, outpath: Path, max_retries: int = 5) -> bool:
    import time

    for attempt in range(max_retries):
        try:
            s3.download_file(bucket, key, str(outpath))
            wait_for_file_release(outpath)
            return True
        except botocore.exceptions.EndpointConnectionError as exc:  # pragma: no cover - network
            LOGGER.warning("EndpointConnectionError on %s (attempt %s/%s): %s", key, attempt + 1, max_retries, exc)
        except botocore.exceptions.ClientError as exc:  # pragma: no cover - network
            LOGGER.warning("ClientError on %s (attempt %s/%s): %s", key, attempt + 1, max_retries, exc)
        except Exception as exc:  # pragma: no cover - network
            LOGGER.warning("Other error on %s (attempt %s/%s): %s", key, attempt + 1, max_retries, exc)
        time.sleep(2)
    LOGGER.error("Failed to download %s after %s attempts", key, max_retries)
    return False


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
            resampling=WarpResampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    return dst


def compute_patch_correlation(bm_patch: np.ndarray, dmsp_patch: np.ndarray) -> float | None:
    mask = (~np.isnan(bm_patch)) & (~np.isnan(dmsp_patch))
    if mask.sum() < 2:
        return None
    bm_vals = bm_patch[mask]
    dmsp_vals = dmsp_patch[mask]
    if np.std(bm_vals) == 0 or np.std(dmsp_vals) == 0:
        return None
    corr = np.corrcoef(bm_vals, dmsp_vals)[0, 1]
    if np.isnan(corr):
        return None
    return float(corr)


def select_best_dmsp_match(
    bm_patch_path: Path,
    file_keys: Sequence[tuple[str, None]],
    s3,
    bucket_name: str,
    download_dir: Path,
    dmsp_out_dir: Path,
    min_valid_fraction: float = 0.10,
) -> List[Path]:
    bm_patch_name = bm_patch_path.name
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)
        bm_patch = bm_src.read(1).astype(np.float32)
        bm_patch[bm_patch < 0] = np.nan

    total_pixels = bm_shape[0] * bm_shape[1]
    best_corr = None
    best_vis_patch: np.ndarray | None = None
    best_vis_file: Path | None = None
    best_valid_fraction = 0.0
    best_f_number = ""

    for vis_key, _ in file_keys:
        base = os.path.basename(vis_key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        vis_file = download_dir / base
        if not vis_file.exists():
            LOGGER.info("Downloading %s", vis_key)
            if not safe_download(s3, bucket_name, vis_key, vis_file):
                continue
        try:
            vis_patch = reproject_to_bm_grid(vis_file, bm_profile)
        except Exception as exc:  # pragma: no cover - reprojection failure
            LOGGER.warning("Error processing %s: %s", vis_file, exc)
            continue
        valid_pixels = np.sum(~np.isnan(vis_patch))
        valid_fraction = valid_pixels / total_pixels if total_pixels else 0
        if valid_fraction < min_valid_fraction:
            LOGGER.debug(
                "Skipping %s for %s due to low coverage (%.2f%%)",
                base,
                bm_patch_name,
                valid_fraction * 100,
            )
            continue
        corr = compute_patch_correlation(bm_patch, vis_patch)
        if corr is None:
            LOGGER.debug("Unable to compute correlation for %s", base)
            continue
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_vis_patch = vis_patch
            best_vis_file = vis_file
            best_valid_fraction = valid_fraction
            best_f_number = f_number

    if best_vis_file is None or best_vis_patch is None or best_corr is None:
        LOGGER.info("No DMSP scene correlated well with %s", bm_patch_name)
        return []

    out_fname = (
        f"{best_f_number}_{best_vis_file.stem}_corr_{best_corr:.3f}_"
        f"match_{bm_patch_name.replace('.tif', '')}.tif"
    )
    out_path = dmsp_out_dir / out_fname
    out_profile = bm_profile.copy()
    out_profile.update({"dtype": "float32", "count": 1, "nodata": np.nan})
    dmsp_out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(best_vis_patch.astype(np.float32), 1)
    LOGGER.info(
        "Saved DMSP patch %s (correlation %.3f, %.1f%% valid)",
        out_path,
        best_corr,
        best_valid_fraction * 100,
    )
    return [out_path]


def parallel_process_bm_patch(
    bm_patch_path: Path,
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
) -> List[Path]:
    bm_date = bm_patch_path.name.split("_")[2]
    dmsp_date_str = bm_date.replace("-", "")
    satellites = [f"F{n}" for n in range(10, 19)]
    file_keys: List[tuple[str, None]] = []
    for sat in satellites:
        prefix = f"{sat}{dmsp_date_str[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if dmsp_date_str in key and key.endswith(".vis.co.tif"):
                    file_keys.append((key, None))
    if not file_keys:
        LOGGER.info("No DMSP scenes found for %s", bm_patch_path.name)
        return []
    saved = select_best_dmsp_match(
        bm_patch_path,
        file_keys,
        s3,
        bucket_name,
        temp_dir,
        dmsp_out_dir,
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    return saved


def download_dmsp_matches(
    bm_patch_paths: Sequence[Path],
    dmsp_out_dir: Path,
    temp_dir: Path,
    max_workers: int = 4,
) -> List[Path]:
    dmsp_out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"
    saved_paths: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path in bm_patch_paths:
            patch_temp = temp_dir / path.stem
            patch_temp.mkdir(parents=True, exist_ok=True)
            futures.append(
                executor.submit(
                    parallel_process_bm_patch,
                    path,
                    s3,
                    bucket_name,
                    patch_temp,
                    dmsp_out_dir,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            saved_paths.extend(future.result())
    shutil.rmtree(temp_dir, ignore_errors=True)
    return saved_paths


def create_pair_manifest(bm_dir: Path, dmsp_dir: Path, manifest_path: Path) -> pd.DataFrame:
    bm_files = {path.stem: path for path in bm_dir.glob("*.tif")}
    rows = []
    pattern = re.compile(
        r"^(?P<f_number>F\d+)_.+_corr_(?P<corr>-?\d+(?:\.\d+)?)_match_(?P<bm_key>BM_patch_.+)$"
    )
    for dmsp_path in dmsp_dir.glob("*.tif"):
        match = pattern.match(dmsp_path.stem)
        if not match:
            continue
        bm_key = match.group("bm_key")
        if bm_key in bm_files:
            rows.append(
                {
                    "bm_patch": str(bm_files[bm_key]),
                    "dmsp_patch": str(dmsp_path),
                    "f_number": match.group("f_number"),
                    "correlation": float(match.group("corr")),
                }
            )
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        LOGGER.info("Wrote manifest to %s", manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paired BM/DMSP GeoTIFFs")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size in pixels")
    parser.add_argument("--samples-per-bin", type=int, default=200, help="Number of LandScan samples per population bin")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for downloads")
    parser.add_argument("--locations-csv", type=Path, default=DEFAULT_LOCATIONS_CSV, help="CSV of locations to process")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output CSV manifest for BM/DMSP pairs")
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Use the existing locations CSV instead of regenerating samples",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Optional list of country names to constrain LandScan sampling",
    )
    parser.add_argument(
        "--country-column",
        type=str,
        help="Shapefile column to use when matching country names",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=DEFAULT_COLLECTION_ID,
        help="NASA CMR collection concept ID",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    if args.skip_sampling and not args.locations_csv.exists():
        raise FileNotFoundError(f"--skip-sampling provided but {args.locations_csv} does not exist")

    if args.skip_sampling:
        if args.countries:
            LOGGER.warning("--countries ignored because --skip-sampling was provided")
        df = pd.read_csv(args.locations_csv)
    else:
        df = sample_landscan_population(
            samples_per_bin=args.samples_per_bin,
            output_csv=args.locations_csv,
            countries=args.countries,
            country_column=args.country_column,
        )

    dmsp_dates = list_dmsp_dates()
    df = assign_random_dates(df, dmsp_dates)
    sample_list = df[["Longitude", "Latitude", "date"]].to_dict(orient="records")

    token = load_nasa_token()
    bm_patches = process_samples_parallel(
        sample_list=sample_list,
        patch_size_pix=args.patch_size,
        collection_id=args.collection_id,
        token=token,
        tile_shapefile_path=DEFAULT_TILE_SHAPEFILE,
        output_folder=BM_OUTPUT_DIR,
        temp_folder=Path("temp_dl"),
        max_workers=args.max_workers,
    )

    dmsp_patches = download_dmsp_matches(
        bm_patch_paths=bm_patches,
        dmsp_out_dir=DMSP_OUTPUT_DIR,
        temp_dir=Path("DMSP_Raw_Temp"),
        max_workers=args.max_workers,
    )

    LOGGER.info("Downloaded %s BM patches and %s DMSP patches", len(bm_patches), len(dmsp_patches))
    create_pair_manifest(BM_OUTPUT_DIR, DMSP_OUTPUT_DIR, args.manifest)


if __name__ == "__main__":
    main()
