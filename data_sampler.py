"""Simplified pipeline for sampling LandScan and downloading paired BM/DMSP GeoTIFFs."""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

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
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import reproject, Resampling as WarpResampling
import rasterio.windows
import requests

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    class tqdm:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.iterable = kwargs.get("iterable")

        def update(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

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
BM_OUTPUT_DIR = Path("bm")
DMSP_OUTPUT_DIR = Path("dmsp")
PLOTS_DIR = Path("plots")
DEFAULT_LOCATIONS_CSV = Path("sampled_locations.csv")
DEFAULT_MANIFEST = Path("bm_dmsp_pairs.csv")
NOMINAL_DEG_PER_PX = 15.0 / 3600.0
BM_DATASET_PATH = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"


class TileMetadataMissingError(RuntimeError):
    """Raised when a Black Marble granule cannot be located in the tile index."""


class DownloadError(RuntimeError):
    """Raised when an S3 object cannot be downloaded after multiple attempts."""

    def __init__(self, bucket: str, key: str, attempts: int, last_exception: Exception | None):
        message = f"Failed to download s3://{bucket}/{key} after {attempts} attempts"
        if last_exception is not None:
            message = f"{message}: {last_exception}"
        super().__init__(message)
        self.bucket = bucket
        self.key = key
        self.attempts = attempts
        self.last_exception = last_exception


@dataclass(frozen=True)
class BMPatchInfo:
    tile_id: str
    path: Path
    date: str
    longitude: float
    latitude: float


@dataclass(frozen=True)
class DMSPMatchedPatch:
    tile_id: str
    path: Path
    f_number: str
    correlation: float
    source_key: str


@dataclass(frozen=True)
class DownloadFailure:
    tile_id: str
    key: str
    error: DownloadError


class _ProgressReporter:
    def __init__(self, total: int | None, desc: str):
        self._bar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=False,
        )

    def update(self, advance: int) -> None:
        try:
            self._bar.update(advance)
        except Exception:  # pragma: no cover - tqdm internal failures
            pass

    def close(self) -> None:
        try:
            self._bar.close()
        except Exception:  # pragma: no cover - tqdm internal failures
            pass


def _create_progress(total: int | None, desc: str) -> _ProgressReporter:
    normalized_total = total if total and total > 0 else None
    return _ProgressReporter(normalized_total, desc)


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


def get_patch_bbox(
    lon: float,
    lat: float,
    patch_size_pix: int,
    pixel_size_deg: float = NOMINAL_DEG_PER_PX,
) -> List[float]:
    half_deg = (patch_size_pix * pixel_size_deg) / 2
    return [lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg]


def _normalize_bbox(bbox: Sequence[float]) -> List[List[float]]:
    """Split a bbox into dateline-safe segments for CMR queries."""

    if len(bbox) != 4:
        raise ValueError("Expected 4 values for bounding box")

    lon1, lat1, lon2, lat2 = bbox
    min_lon, max_lon = sorted((lon1, lon2))
    min_lat, max_lat = sorted((lat1, lat2))
    min_lat = max(-90.0, min_lat)
    max_lat = min(90.0, max_lat)

    if max_lon - min_lon >= 360:
        return [[-180.0, min_lat, 180.0, max_lat]]

    def wrap(lon: float) -> float:
        wrapped = ((lon + 180.0) % 360.0) - 180.0
        if math.isclose(wrapped, -180.0) and lon > 0:
            return 180.0
        return wrapped

    wrapped_min = wrap(min_lon)
    wrapped_max = wrap(max_lon)

    if wrapped_min <= wrapped_max and -180.0 <= wrapped_min <= 180.0 and -180.0 <= wrapped_max <= 180.0:
        return [[wrapped_min, min_lat, wrapped_max, max_lat]]

    first = [wrapped_min, min_lat, 180.0, max_lat]
    second = [-180.0, min_lat, wrapped_max, max_lat]
    return [first, second]


def search_nasa_cmr(collection_id: str, date_str: str, bbox: Sequence[float]) -> List[str]:
    segments = _normalize_bbox(bbox)
    seen: set[str] = set()
    results: List[str] = []
    for segment in segments:
        params = {
            "collection_concept_id": collection_id,
            "temporal": f"{date_str}T00:00:00Z,{date_str}T23:59:59Z",
            "bounding_box": ",".join(f"{value:.6f}" for value in segment),
            "page_size": 50,
        }
        try:
            response = requests.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            LOGGER.warning(
                "CMR query failed for %s with bbox %s: %s", date_str, segment, exc
            )
            continue
        except requests.RequestException as exc:
            LOGGER.warning(
                "CMR request error for %s with bbox %s: %s", date_str, segment, exc
            )
            continue

        granules = response.json().get("feed", {}).get("entry", [])
        for granule in granules:
            for link in granule.get("links", []):
                href = link.get("href", "")
                if href.startswith("https") and href.endswith(".h5") and href not in seen:
                    seen.add(href)
                    results.append(href)
    return results


def h5_to_geotiff(h5_path: Path, tile_shapefile_gdf: gpd.GeoDataFrame) -> Path:
    """Convert a Black Marble granule HDF5 file into a temporary GeoTIFF."""

    with h5py.File(h5_path, "r") as h5_file:
        if BM_DATASET_PATH not in h5_file:
            raise RuntimeError(f"Dataset not found in {h5_path.name}")
        dataset = h5_file[BM_DATASET_PATH]
        data = dataset[...].astype(np.float32)

        for attr in ("_FillValue", "missing_value", "MissingValue"):
            value = dataset.attrs.get(attr)
            if value is None:
                continue
            value_arr = np.asarray(value, dtype=np.float32)
            if value_arr.size == 0:
                continue
            data = np.where(np.isin(data, value_arr), np.nan, data)

        data[data < 0] = np.nan

    tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
    if not tile_match:
        raise RuntimeError(f"Could not determine tile ID for {h5_path.name}")
    tile_id = tile_match.group()
    bounds_row = tile_shapefile_gdf[tile_shapefile_gdf["TileID"] == tile_id]
    if bounds_row.empty:
        raise TileMetadataMissingError(f"Tile ID {tile_id} not found in shapefile")
    left, bottom, right, top = bounds_row.total_bounds

    tif_path = h5_path.with_suffix(".tif")
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=rio_from_bounds(left, bottom, right, top, data.shape[1], data.shape[0]),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)

    return tif_path


def build_bm_mosaic_for_bbox(
    h5_paths: List[Path],
    tile_shapefile_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """Mosaic all tiles intersecting the target bbox into a single raster."""

    tif_paths: List[Path] = []
    try:
        for h5_path in h5_paths:
            try:
                tif_paths.append(h5_to_geotiff(h5_path, tile_shapefile_gdf))
            except TileMetadataMissingError as exc:
                LOGGER.warning("Skipping %s: %s", h5_path.name, exc)
            except RuntimeError:
                raise

        if not tif_paths:
            raise RuntimeError("No valid Black Marble tiles available for mosaic")

        datasets: List[rasterio.io.DatasetReader] = []
        try:
            for tif_path in tif_paths:
                datasets.append(rasterio.open(tif_path))
            mosaic, transform = rio_merge(datasets, nodata=np.nan)
            if mosaic.shape[0] != 1:
                raise RuntimeError("Expected a single-band mosaic")
            mosaic_array = mosaic[0].astype(np.float32, copy=False)
            profile = datasets[0].profile.copy()
            profile.update(
                {
                    "height": mosaic_array.shape[0],
                    "width": mosaic_array.shape[1],
                    "transform": transform,
                    "nodata": np.nan,
                    "count": 1,
                    "dtype": "float32",
                    "crs": "EPSG:4326",
                }
            )
        finally:
            for dataset in datasets:
                dataset.close()
    finally:
        for tif_path in tif_paths:
            tif_path.unlink(missing_ok=True)

    return mosaic_array, transform, profile


def crop_mosaic_to_bbox(
    mosaic_array: np.ndarray,
    mosaic_transform: rasterio.Affine,
    bbox: Sequence[float],
    patch_size_pix: int,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Crop the mosaic to the requested square bounding box."""

    profile = {
        "driver": "GTiff",
        "height": mosaic_array.shape[0],
        "width": mosaic_array.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": mosaic_transform,
        "nodata": np.nan,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(mosaic_array, 1)
            window = rasterio.windows.from_bounds(*bbox, transform=dataset.transform)
            window = window.round_offsets().round_lengths()
            desired_window = rasterio.windows.Window(
                col_off=int(round(window.col_off)),
                row_off=int(round(window.row_off)),
                width=patch_size_pix,
                height=patch_size_pix,
            )
            full_window = rasterio.windows.Window(0, 0, dataset.width, dataset.height)
            intersection = desired_window.intersection(full_window)
            if intersection.width <= 0 or intersection.height <= 0:
                raise RuntimeError(f"Mosaic does not cover bbox {bbox}")
            patch = dataset.read(1, window=intersection, boundless=False).astype(np.float32, copy=False)

    cropped = np.full((patch_size_pix, patch_size_pix), np.nan, dtype=np.float32)
    row_offset = int(round(intersection.row_off - desired_window.row_off))
    col_offset = int(round(intersection.col_off - desired_window.col_off))
    cropped[
        row_offset : row_offset + patch.shape[0],
        col_offset : col_offset + patch.shape[1],
    ] = patch

    patch_transform = rasterio.windows.transform(desired_window, profile["transform"])
    return cropped, patch_transform


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

    bbox = get_patch_bbox(lon, lat, patch_size_pix)
    search_bbox = list(bbox)
    if search_bbox[1] < -60:
        search_bbox[1] = -60
    if search_bbox[3] < -60:
        return (f"Skipping search below -60°S for bbox {bbox}", None)

    urls = search_nasa_cmr(collection_id, date_str, search_bbox)
    if not urls:
        return (f"No Black Marble granules found for {date_str} at ({lon:.3f}, {lat:.3f})", None)

    sample_temp = temp_folder / f"{lon:.3f}_{lat:.3f}_{date_str.replace('-', '')}"
    sample_temp.mkdir(parents=True, exist_ok=True)

    h5_paths: List[Path] = []
    try:
        headers = {"Authorization": f"Bearer {token}"}
        for url in urls:
            h5_path = sample_temp / os.path.basename(url)
            if not h5_path.exists():
                download_file_streaming(url, headers, h5_path)
            h5_paths.append(h5_path)

        mosaic_array, mosaic_transform, mosaic_profile = build_bm_mosaic_for_bbox(
            h5_paths, tile_shapefile
        )
        patch, patch_transform = crop_mosaic_to_bbox(
            mosaic_array, mosaic_transform, bbox, patch_size_pix
        )

        out_path = output_folder / f"BM_patch_{date_str}_{lon:.3f}_{lat:.3f}.tif"
        profile = mosaic_profile.copy()
        profile.update(
            {
                "height": patch.shape[0],
                "width": patch.shape[1],
                "transform": patch_transform,
                "dtype": "float32",
                "nodata": np.nan,
            }
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(patch, 1)
    except Exception as exc:  # pragma: no cover - network/file errors
        LOGGER.error(
            "Error processing sample %s (%s, %s): %s", date_str, lon, lat, exc, exc_info=True
        )
        return (f"Failed to save patch for {date_str} at ({lon:.3f}, {lat:.3f}): {exc}", None)
    finally:
        for path in h5_paths:
            path.unlink(missing_ok=True)
        shutil.rmtree(sample_temp, ignore_errors=True)

    return (f"Saved mosaic patch: {out_path}", out_path)


def process_samples_parallel(
    sample_list: Sequence[dict],
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapefile_path: Path,
    output_folder: Path,
    temp_folder: Path,
    max_workers: int = 4,
) -> List[BMPatchInfo]:
    output_folder.mkdir(parents=True, exist_ok=True)
    temp_folder.mkdir(parents=True, exist_ok=True)
    tile_shapefile = gpd.read_file(tile_shapefile_path)

    results: List[BMPatchInfo] = []
    failures: list[tuple[dict, Exception]] = []
    successes: list[tuple[int, dict, Path]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(
                process_single_sample,
                sample,
                patch_size_pix,
                collection_id,
                token,
                tile_shapefile,
                output_folder,
                temp_folder,
            ): (index, sample)
            for index, sample in enumerate(sample_list, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_sample):
            index, sample = future_to_sample[future]
            try:
                message, path = future.result()
            except Exception as exc:  # pragma: no cover - unexpected worker error
                failures.append((sample, exc))
                LOGGER.error(
                    "Worker failed for sample %s: %s", sample, exc, exc_info=True
                )
                continue
            LOGGER.info(message)
            if path is not None:
                successes.append((index, sample, path))

    if failures:
        LOGGER.warning("Encountered %d failed samples", len(failures))

    shutil.rmtree(temp_folder, ignore_errors=True)

    successes.sort(key=lambda item: item[0])
    for tile_number, (_, sample, path) in enumerate(successes, start=1):
        tile_id = f"tile_{tile_number:03d}"
        tile_path = path.with_name(f"{tile_id}{path.suffix}")
        if tile_path.exists():
            tile_path.unlink()
        path.rename(tile_path)
        LOGGER.info("Assigned %s to %s", tile_id, tile_path)
        results.append(
            BMPatchInfo(
                tile_id=tile_id,
                path=tile_path,
                date=sample["date"],
                longitude=sample["Longitude"],
                latitude=sample["Latitude"],
            )
        )

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


def download_file_streaming(url: str, headers: dict[str, str], destination: Path) -> None:
    desc = destination.name
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        total_length = response.headers.get("Content-Length")
        total = int(total_length) if total_length and total_length.isdigit() else None
        progress = _create_progress(total, desc)
        try:
            with destination.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 512):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    progress.update(len(chunk))
        finally:
            progress.close()


def safe_download(s3, bucket: str, key: str, outpath: Path, max_retries: int = 5) -> None:
    import time

    size: int | None = None
    try:
        metadata = s3.head_object(Bucket=bucket, Key=key)
        if metadata is not None:
            size = metadata.get("ContentLength")
    except Exception as exc:  # pragma: no cover - boto3 optional failure path
        LOGGER.debug("Unable to determine size for %s: %s", key, exc)

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        progress = _create_progress(size, Path(key).name)
        try:
            s3.download_file(bucket, key, str(outpath), Callback=progress.update)
            wait_for_file_release(outpath)
            return
        except botocore.exceptions.EndpointConnectionError as exc:  # pragma: no cover - network
            last_exc = exc
            LOGGER.warning(
                "EndpointConnectionError on %s (attempt %s/%s): %s",
                key,
                attempt,
                max_retries,
                exc,
            )
        except botocore.exceptions.ClientError as exc:  # pragma: no cover - network
            last_exc = exc
            LOGGER.warning(
                "ClientError on %s (attempt %s/%s): %s", key, attempt, max_retries, exc
            )
        except Exception as exc:  # pragma: no cover - network
            last_exc = exc
            LOGGER.warning(
                "Other error on %s (attempt %s/%s): %s", key, attempt, max_retries, exc
            )
        finally:
            progress.close()
            outpath.unlink(missing_ok=True)
        time.sleep(2)

    LOGGER.error("Failed to download %s after %s attempts", key, max_retries)
    raise DownloadError(bucket, key, max_retries, last_exc)


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
    bm_patch: BMPatchInfo,
    file_keys: Sequence[str],
    s3,
    bucket_name: str,
    download_dir: Path,
    dmsp_out_dir: Path,
    min_valid_fraction: float = 0.10,
) -> tuple[list[DMSPMatchedPatch], list[DownloadFailure]]:
    bm_patch_path = bm_patch.path
    bm_patch_name = bm_patch_path.name
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)
        bm_patch_array = bm_src.read(1).astype(np.float32)
        bm_patch_array[bm_patch_array < 0] = np.nan

    total_pixels = bm_shape[0] * bm_shape[1]
    best_corr = None
    best_vis_patch: np.ndarray | None = None
    best_vis_file: Path | None = None
    best_valid_fraction = 0.0
    best_f_number = ""
    best_source_key = ""

    download_failures: list[DownloadFailure] = []

    for vis_key in file_keys:
        base = os.path.basename(vis_key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        vis_file = download_dir / base
        if not vis_file.exists():
            LOGGER.info("Downloading %s", vis_key)
            try:
                safe_download(s3, bucket_name, vis_key, vis_file)
            except DownloadError as exc:
                LOGGER.warning(
                    "Failed to download %s for tile %s (%s): %s",
                    vis_key,
                    bm_patch.tile_id,
                    bm_patch_name,
                    exc,
                )
                download_failures.append(
                    DownloadFailure(tile_id=bm_patch.tile_id, key=vis_key, error=exc)
                )
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
        corr = compute_patch_correlation(bm_patch_array, vis_patch)
        if corr is None:
            LOGGER.debug("Unable to compute correlation for %s", base)
            continue
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_vis_patch = vis_patch
            best_vis_file = vis_file
            best_valid_fraction = valid_fraction
            best_f_number = f_number
            best_source_key = vis_key

    if best_vis_file is None or best_vis_patch is None or best_corr is None:
        LOGGER.info("No DMSP scene correlated well with %s", bm_patch_name)
        return ([], download_failures)

    out_path = dmsp_out_dir / f"{bm_patch.tile_id}{bm_patch_path.suffix}"
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
    match = DMSPMatchedPatch(
        tile_id=bm_patch.tile_id,
        path=out_path,
        f_number=best_f_number,
        correlation=float(best_corr),
        source_key=best_source_key,
    )
    return ([match], download_failures)


def parallel_process_bm_patch(
    bm_patch: BMPatchInfo,
    s3,
    bucket_name: str,
    temp_dir: Path,
    dmsp_out_dir: Path,
) -> tuple[list[DMSPMatchedPatch], list[DownloadFailure]]:
    dmsp_date_str = bm_patch.date.replace("-", "")
    satellites = [f"F{n}" for n in range(10, 19)]
    file_keys: List[str] = []
    for sat in satellites:
        prefix = f"{sat}{dmsp_date_str[:4]}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if dmsp_date_str in key and key.endswith(".vis.co.tif"):
                    file_keys.append(key)
    if not file_keys:
        LOGGER.info(
            "No DMSP scenes found for tile %s (%s)",
            bm_patch.tile_id,
            bm_patch.path.name,
        )
        return ([], [])
    matches, failures = select_best_dmsp_match(
        bm_patch,
        file_keys,
        s3,
        bucket_name,
        temp_dir,
        dmsp_out_dir,
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    return (matches, failures)


def download_dmsp_matches(
    bm_patches: Sequence[BMPatchInfo],
    dmsp_out_dir: Path,
    temp_dir: Path,
    max_workers: int = 4,
) -> tuple[list[DMSPMatchedPatch], list[DownloadFailure]]:
    dmsp_out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket_name = "globalnightlight"
    saved_matches: list[DMSPMatchedPatch] = []
    download_failures: list[DownloadFailure] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for patch in bm_patches:
            patch_temp = temp_dir / patch.tile_id
            patch_temp.mkdir(parents=True, exist_ok=True)
            futures.append(
                executor.submit(
                    parallel_process_bm_patch,
                    patch,
                    s3,
                    bucket_name,
                    patch_temp,
                    dmsp_out_dir,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            matches, failures = future.result()
            saved_matches.extend(matches)
            download_failures.extend(failures)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return (saved_matches, download_failures)


def create_pair_manifest(
    bm_patches: Sequence[BMPatchInfo],
    dmsp_matches: Sequence[DMSPMatchedPatch],
    manifest_path: Path,
) -> pd.DataFrame:
    bm_index = {patch.tile_id: patch for patch in bm_patches}
    rows = []
    for match in dmsp_matches:
        bm_patch = bm_index.get(match.tile_id)
        if bm_patch is None:
            continue
        rows.append(
            {
                "tile_id": match.tile_id,
                "bm_patch": str(bm_patch.path),
                "dmsp_patch": str(match.path),
                "f_number": match.f_number,
                "correlation": match.correlation,
                "source_key": match.source_key,
                "date": bm_patch.date,
                "longitude": bm_patch.longitude,
                "latitude": bm_patch.latitude,
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        LOGGER.info("Wrote manifest to %s", manifest_path)
    elif manifest_path.exists():
        manifest_path.unlink()
    return manifest


def resolve_cli_path(
    output_root: Path | None,
    candidate: Path,
    default_value: Path,
    default_name: str,
) -> Path:
    """Resolve a CLI-provided path relative to the optional output root."""

    candidate = candidate.expanduser()
    if output_root is None or candidate.is_absolute():
        return candidate
    if candidate == default_value:
        return output_root / default_name
    return output_root / candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download paired BM/DMSP GeoTIFFs")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size in pixels")
    parser.add_argument("--samples-per-bin", type=int, default=200, help="Number of LandScan samples per population bin")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for downloads")
    parser.add_argument("--locations-csv", type=Path, default=DEFAULT_LOCATIONS_CSV, help="CSV of locations to process")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output CSV manifest for BM/DMSP pairs")
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Root directory where BM, DMSP, plots, and CSV artifacts will be written",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Use the existing locations CSV instead of regenerating samples",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=2024,
        help="Random seed used when drawing LandScan samples",
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
    parser.add_argument(
        "--date-seed",
        type=int,
        default=13492,
        help="Random seed used when assigning DMSP acquisition dates",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    output_root = args.output_folder.expanduser() if args.output_folder else None

    bm_dir = output_root / BM_OUTPUT_DIR if output_root else BM_OUTPUT_DIR
    dmsp_dir = output_root / DMSP_OUTPUT_DIR if output_root else DMSP_OUTPUT_DIR
    plots_dir = output_root / PLOTS_DIR if output_root else PLOTS_DIR
    temp_dir = output_root / Path("temp_dl") if output_root else Path("temp_dl")
    dmsp_temp_dir = (
        output_root / Path("dmsp_raw_temp") if output_root else Path("dmsp_raw_temp")
    )

    locations_csv = resolve_cli_path(
        output_root,
        args.locations_csv,
        DEFAULT_LOCATIONS_CSV,
        DEFAULT_LOCATIONS_CSV.name,
    )
    manifest_path = resolve_cli_path(
        output_root,
        args.manifest,
        DEFAULT_MANIFEST,
        DEFAULT_MANIFEST.name,
    )

    if args.skip_sampling and not locations_csv.exists():
        raise FileNotFoundError(f"--skip-sampling provided but {locations_csv} does not exist")

    if args.skip_sampling:
        if args.countries:
            LOGGER.warning("--countries ignored because --skip-sampling was provided")
        df = pd.read_csv(locations_csv)
    else:
        df = sample_landscan_population(
            samples_per_bin=args.samples_per_bin,
            random_seed=args.sampling_seed,
            plots_dir=plots_dir,
            output_csv=locations_csv,
            countries=args.countries,
            country_column=args.country_column,
        )

    dmsp_dates = list_dmsp_dates()
    df = assign_random_dates(df, dmsp_dates, seed=args.date_seed)
    sample_list = df[["Longitude", "Latitude", "date"]].to_dict(orient="records")

    token = load_nasa_token()
    bm_patches = process_samples_parallel(
        sample_list=sample_list,
        patch_size_pix=args.patch_size,
        collection_id=args.collection_id,
        token=token,
        tile_shapefile_path=DEFAULT_TILE_SHAPEFILE,
        output_folder=bm_dir,
        temp_folder=temp_dir,
        max_workers=args.max_workers,
    )

    dmsp_matches, download_failures = download_dmsp_matches(
        bm_patches=bm_patches,
        dmsp_out_dir=dmsp_dir,
        temp_dir=dmsp_temp_dir,
        max_workers=args.max_workers,
    )

    LOGGER.info(
        "Downloaded %s BM patches and %s DMSP patches",
        len(bm_patches),
        len(dmsp_matches),
    )
    if download_failures:
        LOGGER.warning(
            "Encountered %s DMSP download failures", len(download_failures)
        )
        for failure in download_failures:
            LOGGER.warning(
                "Tile %s failed to download %s: %s",
                failure.tile_id,
                failure.key,
                failure.error,
            )

    manifest = create_pair_manifest(bm_patches, dmsp_matches, manifest_path)

    expected_bm = {patch.path.name for patch in bm_patches}
    expected_dmsp = {match.path.name for match in dmsp_matches}

    def iter_rasters(directory: Path) -> Iterable[Path]:
        seen: set[Path] = set()
        for pattern in ("*.tif", "*.TIF"):
            for path in directory.glob(pattern):
                if path not in seen:
                    seen.add(path)
                    yield path

    bm_removed = 0
    if bm_dir.exists():
        for bm_path in iter_rasters(bm_dir):
            if bm_path.name not in expected_bm:
                bm_path.unlink(missing_ok=True)
                bm_removed += 1

    dmsp_removed = 0
    if dmsp_dir.exists():
        for dmsp_path in iter_rasters(dmsp_dir):
            if dmsp_path.name not in expected_dmsp:
                dmsp_path.unlink(missing_ok=True)
                dmsp_removed += 1

    if bm_removed or dmsp_removed:
        LOGGER.info(
            "Removed %s unmatched BM rasters and %s unmatched DMSP rasters", bm_removed, dmsp_removed
        )
    else:
        LOGGER.info("No unmatched BM or DMSP rasters were removed")


if __name__ == "__main__":
    main()
