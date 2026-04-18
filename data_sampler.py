"""Simplified pipeline for sampling LandScan and downloading paired BM/DMSP GeoTIFFs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
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
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_bounds as rio_from_bounds
from rasterio.warp import reproject, Resampling as WarpResampling
import rasterio.windows
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

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
DEFAULT_DMSP_DATES_CACHE = Path(".dmsp_dates_cache.txt")
DEFAULT_CACHE_ROOT = Path(".cache")
BM_GRANULE_CACHE_DIRNAME = "bm_granules"
DMSP_RAW_CACHE_DIRNAME = "dmsp_raw"
BM_CMR_CACHE_DIRNAME = "bm_cmr"
BM_GEOTIFF_CACHE_DIRNAME = "bm_geotiff"
DMSP_SCENE_INDEX_DIRNAME = "dmsp_scene_index"
PAIR_OUTPUT_ROOT = Path("Raw_NL_Data")
BM_OUTPUT_DIR = PAIR_OUTPUT_ROOT / "bm"
DMSP_OUTPUT_DIR = PAIR_OUTPUT_ROOT / "dmsp"
PLOTS_DIR = Path("plots")
DEFAULT_LOCATIONS_CSV = Path("sampled_locations.csv")
DEFAULT_MANIFEST = Path("Raw_NL_Data/bm_dmsp_pairs.csv")
TIMINGS_FILENAME = "timings.json"
NOMINAL_DEG_PER_PX = 15.0 / 3600.0
BM_DATASET_PATH = "/HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
HTTP_POOL_SIZE = 16
DMSP_BUCKET_NAME = "globalnightlight"
DMSP_SATELLITES = tuple(f"F{n}" for n in range(10, 19))
_HTTP_SESSION_LOCAL = threading.local()


class TileMetadataMissingError(RuntimeError):
    """Raised when a Black Marble granule cannot be located in the tile index."""


class DownloadError(RuntimeError):
    """Raised when an object cannot be downloaded after multiple attempts."""

    def __init__(self, bucket: str, key: str, attempts: int, error: Exception):
        super().__init__(
            f"Failed to download s3://{bucket}/{key} after {attempts} attempts: {error}"
        )
        self.bucket = bucket
        self.key = key
        self.attempts = attempts
        self.error = error


@dataclass
class BMPatch:
    tile_id: str
    path: Path
    longitude: float
    latitude: float
    date: str
    population_bin: str | None = None


@dataclass
class DMSPMatch:
    tile_id: str
    bm_path: Path
    dmsp_path: Path
    f_number: str
    correlation: float
    valid_fraction: float
    source_key: str


@dataclass
class DownloadFailure:
    tile_id: str
    bm_path: Path
    key: str
    error: Exception


@dataclass
class RunMetrics:
    counts: dict[str, int] = field(default_factory=dict)
    bytes_downloaded: dict[str, int] = field(default_factory=dict)
    cache_hits: dict[str, int] = field(default_factory=dict)
    cache_misses: dict[str, int] = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def set_metadata(self, key: str, value: object) -> None:
        with self._lock:
            self.metadata[key] = value

    def set_count(self, name: str, value: int) -> None:
        with self._lock:
            self.counts[name] = int(value)

    def increment_count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + int(amount)

    def add_bytes(self, name: str, amount: int) -> None:
        with self._lock:
            self.bytes_downloaded[name] = self.bytes_downloaded.get(name, 0) + int(amount)

    def record_cache_hit(self, name: str) -> None:
        with self._lock:
            self.cache_hits[name] = self.cache_hits.get(name, 0) + 1

    def record_cache_miss(self, name: str) -> None:
        with self._lock:
            self.cache_misses[name] = self.cache_misses.get(name, 0) + 1

    def add_stage_time(self, stage: str, elapsed_seconds: float) -> None:
        with self._lock:
            self.stage_seconds[stage] = self.stage_seconds.get(stage, 0.0) + float(
                elapsed_seconds
            )

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_stage_time(stage, time.perf_counter() - start)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "counts": dict(sorted(self.counts.items())),
                "bytes_downloaded": dict(sorted(self.bytes_downloaded.items())),
                "cache_hits": dict(sorted(self.cache_hits.items())),
                "cache_misses": dict(sorted(self.cache_misses.items())),
                "stage_seconds": {
                    key: round(value, 6) for key, value in sorted(self.stage_seconds.items())
                },
                "metadata": dict(sorted(self.metadata.items())),
            }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextmanager
def _progress_bar(total: int | None, description: str):
    if tqdm is None:
        yield None
        return
    bar = tqdm(total=total, unit="B", unit_scale=True, desc=description, leave=False)
    try:
        yield bar
    finally:
        bar.close()


def _write_response_to_file(response, destination: Path, description: str) -> int:
    chunk_size = 1024 * 128
    total_header = response.headers.get("Content-Length") if hasattr(response, "headers") else None
    total: int | None = None
    if total_header is not None:
        try:
            total = int(total_header)
        except (TypeError, ValueError):  # pragma: no cover - malformed header
            total = None
    bytes_written = 0
    with destination.open("wb") as fh:
        with _progress_bar(total, description) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                bytes_written += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
    return bytes_written


def build_http_session():
    session_factory = getattr(requests, "Session", None)
    if session_factory is None:  # pragma: no cover - exercised only in lightweight tests
        return None

    session = session_factory()
    try:  # pragma: no cover - adapters may be unavailable in lightweight tests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=HTTP_POOL_SIZE,
            pool_maxsize=HTTP_POOL_SIZE,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    except Exception:
        pass
    return session


def get_http_session():
    session = getattr(_HTTP_SESSION_LOCAL, "session", None)
    if session is None:
        session = build_http_session()
        _HTTP_SESSION_LOCAL.session = session
    return session


def materialize_cached_file(
    target_path: Path,
    writer,
    wait_timeout: float = 1800.0,
    poll_interval: float = 0.25,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    lock_path = target_path.with_name(f"{target_path.name}.lock")
    start = time.time()
    while True:
        if target_path.exists():
            return target_path
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - start > wait_timeout:
                raise TimeoutError(f"Timed out waiting for cached file {target_path}")
            time.sleep(poll_interval)

    temp_path = target_path.with_name(
        f"{target_path.name}.{os.getpid()}.{threading.get_ident()}.part"
    )
    try:
        if target_path.exists():
            return target_path
        writer(temp_path)
        os.replace(temp_path, target_path)
        return target_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def build_s3_client(max_workers: int = 4):
    pool_size = max(HTTP_POOL_SIZE, max_workers * 4)
    try:
        config = Config(signature_version=UNSIGNED, max_pool_connections=pool_size)
    except TypeError:  # pragma: no cover - lightweight tests may stub Config loosely
        try:
            config = Config(signature_version=UNSIGNED)
        except TypeError:
            config = None
    return boto3.client("s3", config=config)


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
    s3 = build_s3_client()
    prefix = "F"
    paginator = s3.get_paginator("list_objects_v2")
    all_dates: set[str] = set()
    for page in paginator.paginate(Bucket=DMSP_BUCKET_NAME, Prefix=prefix):
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


def _serialize_scene_entries(scene_entries: Sequence[tuple[str, int | None]]) -> List[dict[str, object]]:
    return [{"key": key, "size": size} for key, size in scene_entries]


def _deserialize_scene_entries(entries: Sequence[dict[str, object]]) -> List[tuple[str, int | None]]:
    scene_entries: List[tuple[str, int | None]] = []
    for entry in entries:
        key = str(entry.get("key", ""))
        if not key:
            continue
        size = entry.get("size")
        scene_entries.append((key, int(size) if size is not None else None))
    return scene_entries


def load_dmsp_year_scene_index(
    year: str,
    s3,
    bucket_name: str,
    cache_dir: Path,
    metrics: RunMetrics | None = None,
) -> dict[str, List[tuple[str, int | None]]]:
    cache_path = cache_dir / f"{year}.json"
    if cache_path.exists():
        if metrics is not None:
            metrics.record_cache_hit("dmsp_scene_index")
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            dates = payload.get("dates", {})
            if isinstance(dates, dict):
                return {
                    str(date_key): _deserialize_scene_entries(scene_entries)
                    for date_key, scene_entries in dates.items()
                }
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            LOGGER.warning("Failed to read cached DMSP scene index %s; rebuilding", cache_path)

    if metrics is not None:
        metrics.record_cache_miss("dmsp_scene_index")

    scenes_by_date: defaultdict[str, List[tuple[str, int | None]]] = defaultdict(list)
    for sat in DMSP_SATELLITES:
        prefix = f"{sat}{year}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".vis.co.tif"):
                    continue
                fname = os.path.basename(key)
                if len(fname) < 11:
                    continue
                date_str = fname[3:11]
                if not date_str.startswith(year):
                    continue
                scenes_by_date[date_str].append((key, obj.get("Size")))

    payload = {
        "year": year,
        "dates": {
            date_key: _serialize_scene_entries(sorted(scene_entries))
            for date_key, scene_entries in sorted(scenes_by_date.items())
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {date_key: list(scene_entries) for date_key, scene_entries in scenes_by_date.items()}


def list_dmsp_scene_keys_for_dates(
    dmsp_date_strs: Sequence[str],
    s3,
    bucket_name: str,
    cache_dir: Path,
    metrics: RunMetrics | None = None,
) -> dict[str, List[tuple[str, int | None]]]:
    unique_dates = sorted({date_str for date_str in dmsp_date_strs if date_str})
    indices_by_year = {
        year: load_dmsp_year_scene_index(year, s3, bucket_name, cache_dir, metrics=metrics)
        for year in sorted({date_str[:4] for date_str in unique_dates})
    }
    return {
        date_str: list(indices_by_year.get(date_str[:4], {}).get(date_str, []))
        for date_str in unique_dates
    }


def list_dmsp_scene_keys_for_date(
    dmsp_date_str: str,
    s3,
    bucket_name: str,
    cache_dir: Path | None = None,
    metrics: RunMetrics | None = None,
) -> List[tuple[str, int | None]]:
    resolved_cache_dir = cache_dir or (DEFAULT_CACHE_ROOT / DMSP_SCENE_INDEX_DIRNAME)
    return list(
        list_dmsp_scene_keys_for_dates(
            [dmsp_date_str],
            s3,
            bucket_name,
            resolved_cache_dir,
            metrics=metrics,
        ).get(dmsp_date_str, [])
    )


def get_dmsp_dates(
    cache_path: Path = DEFAULT_DMSP_DATES_CACHE,
    min_date: pd.Timestamp | None = None,
    metrics: RunMetrics | None = None,
) -> List[str]:
    if cache_path.exists():
        if metrics is not None:
            metrics.record_cache_hit("dmsp_dates")
        cached_dates = [
            line.strip()
            for line in cache_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if cached_dates:
            LOGGER.info("Loaded %s cached DMSP dates from %s", len(cached_dates), cache_path)
            return cached_dates

    if metrics is not None:
        metrics.record_cache_miss("dmsp_dates")
    dates = list_dmsp_dates(min_date=min_date)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(dates) + "\n", encoding="utf-8")
    LOGGER.info("Cached %s DMSP dates to %s", len(dates), cache_path)
    return dates


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


def search_nasa_cmr(
    collection_id: str,
    date_str: str,
    bbox: Sequence[float],
    session=None,
) -> List[str]:
    session = session or get_http_session()
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
            response = session.get(
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


def bounds_intersect(
    bounds_a: Sequence[float],
    bounds_b: Sequence[float],
) -> bool:
    left_a, bottom_a, right_a, top_a = bounds_a
    left_b, bottom_b, right_b, top_b = bounds_b
    return not (
        right_a < left_b
        or right_b < left_a
        or top_a < bottom_b
        or top_b < bottom_a
    )


def find_bm_tile_ids_for_bbox(
    bbox: Sequence[float],
    tile_bounds_lookup: dict[str, tuple[float, float, float, float]],
) -> List[str]:
    segments = _normalize_bbox(bbox)
    return sorted(
        tile_id
        for tile_id, tile_bounds in tile_bounds_lookup.items()
        if any(bounds_intersect(tile_bounds, segment) for segment in segments)
    )


def filter_cmr_urls_for_tile_ids(
    urls: Sequence[str],
    required_tile_ids: Sequence[str],
) -> List[str]:
    if not required_tile_ids:
        return list(urls)

    required = set(required_tile_ids)
    filtered = [
        url
        for url in urls
        if (match := re.search(r"h\d{2}v\d{2}", url)) is not None and match.group() in required
    ]
    if filtered:
        return filtered
    LOGGER.warning(
        "CMR results did not match the required BM tile IDs %s; falling back to all returned URLs",
        ", ".join(required_tile_ids),
    )
    return list(urls)


def get_cached_bm_granule_urls(
    collection_id: str,
    date_str: str,
    bbox: Sequence[float],
    required_tile_ids: Sequence[str],
    session,
    cache_dir: Path,
    metrics: RunMetrics | None = None,
) -> List[str]:
    cache_payload = {
        "collection_id": collection_id,
        "date": date_str,
        "required_tile_ids": list(required_tile_ids),
    }
    digest = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = cache_dir / f"{date_str.replace('-', '')}_{digest}.json"
    if cache_path.exists():
        if metrics is not None:
            metrics.record_cache_hit("bm_cmr")
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            urls = payload.get("urls", [])
            if isinstance(urls, list):
                return [str(url) for url in urls]
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            LOGGER.warning("Failed to read cached CMR response %s; rebuilding", cache_path)

    if metrics is not None:
        metrics.record_cache_miss("bm_cmr")
        metrics.increment_count("bm_cmr_queries")
    urls = search_nasa_cmr(collection_id, date_str, bbox, session=session)
    filtered_urls = filter_cmr_urls_for_tile_ids(urls, required_tile_ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "collection_id": collection_id,
                "date": date_str,
                "required_tile_ids": list(required_tile_ids),
                "urls": filtered_urls,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return filtered_urls


def build_tile_bounds_lookup(
    tile_shapefile_gdf: gpd.GeoDataFrame,
) -> dict[str, tuple[float, float, float, float]]:
    if "TileID" not in tile_shapefile_gdf.columns:
        raise ValueError("Black Marble tile shapefile is missing the TileID column")
    lookup: dict[str, tuple[float, float, float, float]] = {}
    for row in tile_shapefile_gdf[["TileID", "geometry"]].itertuples(index=False):
        geometry = getattr(row, "geometry", None)
        if geometry is None:
            continue
        lookup[str(row.TileID)] = tuple(float(value) for value in geometry.bounds)
    return lookup


def h5_to_geotiff(
    h5_path: Path,
    tile_bounds_lookup: dict[str, tuple[float, float, float, float]],
    out_path: Path | None = None,
) -> Path:
    """Convert a Black Marble granule HDF5 file into a temporary GeoTIFF."""

    bounds: tuple[float, float, float, float] | None = None
    with h5py.File(h5_path, "r") as h5_file:
        if BM_DATASET_PATH not in h5_file:
            raise RuntimeError(f"Dataset not found in {h5_path.name}")
        dataset = h5_file[BM_DATASET_PATH]
        data = dataset[...].astype(np.float32)
        west = h5_file.attrs.get("WestBoundingCoord")
        east = h5_file.attrs.get("EastBoundingCoord")
        south = h5_file.attrs.get("SouthBoundingCoord")
        north = h5_file.attrs.get("NorthBoundingCoord")
        if all(value is not None for value in (west, east, south, north)):
            bounds = (
                float(np.asarray(west).reshape(-1)[0]),
                float(np.asarray(south).reshape(-1)[0]),
                float(np.asarray(east).reshape(-1)[0]),
                float(np.asarray(north).reshape(-1)[0]),
            )

        for attr in ("_FillValue", "missing_value", "MissingValue"):
            value = dataset.attrs.get(attr)
            if value is None:
                continue
            value_arr = np.asarray(value, dtype=np.float32)
            if value_arr.size == 0:
                continue
            data = np.where(np.isin(data, value_arr), np.nan, data)

        data[data < 0] = np.nan

    if bounds is None:
        tile_match = re.search(r"h\d{2}v\d{2}", h5_path.name)
        if not tile_match:
            raise RuntimeError(f"Could not determine tile ID for {h5_path.name}")
        tile_id = tile_match.group()
        bounds = tile_bounds_lookup.get(tile_id)
        if bounds is None:
            raise TileMetadataMissingError(f"Tile ID {tile_id} not found in shapefile")
    left, bottom, right, top = bounds

    tif_path = out_path if out_path is not None else h5_path.with_suffix(".tif")
    tif_path.parent.mkdir(parents=True, exist_ok=True)
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
    tile_bounds_lookup: dict[str, tuple[float, float, float, float]],
    bbox: Sequence[float] | None = None,
    geotiff_cache_dir: Path | None = None,
    metrics: RunMetrics | None = None,
) -> tuple[np.ndarray, rasterio.Affine, dict]:
    """Mosaic all tiles intersecting the target bbox into a single raster."""

    tif_paths: List[Path] = []
    try:
        for h5_path in h5_paths:
            try:
                if geotiff_cache_dir is None:
                    tif_paths.append(h5_to_geotiff(h5_path, tile_bounds_lookup))
                else:
                    tif_path = geotiff_cache_dir / f"{h5_path.stem}.geo.tif"
                    if tif_path.exists():
                        if metrics is not None:
                            metrics.record_cache_hit("bm_geotiff")
                    else:
                        if metrics is not None:
                            metrics.record_cache_miss("bm_geotiff")
                        tif_path = materialize_cached_file(
                            tif_path,
                            lambda temp_path, source_path=h5_path: h5_to_geotiff(
                                source_path,
                                tile_bounds_lookup,
                                out_path=temp_path,
                            ),
                        )
                    tif_paths.append(tif_path)
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
            merge_kwargs = {"nodata": np.nan}
            if bbox is not None and bbox[0] <= bbox[2]:
                merge_kwargs["bounds"] = tuple(float(value) for value in bbox)
            mosaic, transform = rio_merge(datasets, **merge_kwargs)
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
        if geotiff_cache_dir is None:
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

    window = rasterio.windows.from_bounds(*bbox, transform=mosaic_transform)
    window = window.round_offsets().round_lengths()
    desired_window = rasterio.windows.Window(
        col_off=int(round(window.col_off)),
        row_off=int(round(window.row_off)),
        width=patch_size_pix,
        height=patch_size_pix,
    )
    full_window = rasterio.windows.Window(0, 0, mosaic_array.shape[1], mosaic_array.shape[0])
    intersection = desired_window.intersection(full_window)
    if intersection.width <= 0 or intersection.height <= 0:
        raise RuntimeError(f"Mosaic does not cover bbox {bbox}")

    row_start = int(round(intersection.row_off))
    row_stop = row_start + int(round(intersection.height))
    col_start = int(round(intersection.col_off))
    col_stop = col_start + int(round(intersection.width))
    patch = mosaic_array[row_start:row_stop, col_start:col_stop].astype(np.float32, copy=False)

    cropped = np.full((patch_size_pix, patch_size_pix), np.nan, dtype=np.float32)
    row_offset = int(round(intersection.row_off - desired_window.row_off))
    col_offset = int(round(intersection.col_off - desired_window.col_off))
    cropped[
        row_offset : row_offset + patch.shape[0],
        col_offset : col_offset + patch.shape[1],
    ] = patch

    patch_transform = rasterio.windows.transform(desired_window, mosaic_transform)
    return cropped, patch_transform


def process_single_sample(
    sample: dict,
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_bounds_lookup: dict[str, tuple[float, float, float, float]],
    output_folder: Path,
    bm_cache_dir: Path,
    bm_cmr_cache_dir: Path,
    bm_geotiff_cache_dir: Path,
    metrics: RunMetrics | None = None,
) -> tuple[str, Path | None]:
    lon, lat, date_str = sample["Longitude"], sample["Latitude"], sample["date"]
    tile_id = sample["tile_id"]
    out_path = output_folder / f"{tile_id}.tif"

    if out_path.exists():
        return (
            f"Reusing existing BM patch: {out_path}",
            BMPatch(
                tile_id=tile_id,
                path=out_path,
                longitude=lon,
                latitude=lat,
                date=date_str,
                population_bin=sample.get("Bin"),
            ),
        )

    if lat < -60:
        return (f"Skipping Antarctica sample at ({lon:.3f}, {lat:.3f})", None)

    bbox = get_patch_bbox(lon, lat, patch_size_pix)
    search_bbox = list(bbox)
    if search_bbox[1] < -60:
        search_bbox[1] = -60
    if search_bbox[3] < -60:
        return (f"Skipping search below -60°S for bbox {bbox}", None)

    session = get_http_session()
    required_tile_ids = find_bm_tile_ids_for_bbox(search_bbox, tile_bounds_lookup)
    urls = get_cached_bm_granule_urls(
        collection_id,
        date_str,
        search_bbox,
        required_tile_ids,
        session,
        bm_cmr_cache_dir,
        metrics=metrics,
    )
    if not urls:
        return (f"No Black Marble granules found for {date_str} at ({lon:.3f}, {lat:.3f})", None)

    h5_paths: List[Path] = []
    try:
        headers = {"Authorization": f"Bearer {token}"}
        for url in urls:
            h5_path = bm_cache_dir / os.path.basename(url)
            if h5_path.exists():
                if metrics is not None:
                    metrics.record_cache_hit("bm_granules")
            else:
                if metrics is not None:
                    metrics.record_cache_miss("bm_granules")
                LOGGER.info("Downloading Black Marble granule %s for %s", url, tile_id)

                def writer(temp_path: Path, source_url=url):
                    response = session.get(source_url, headers=headers, stream=True, timeout=120)
                    response.raise_for_status()
                    try:
                        bytes_written = _write_response_to_file(
                            response,
                            temp_path,
                            f"BM {tile_id}: {os.path.basename(source_url)}",
                        )
                        if metrics is not None:
                            metrics.add_bytes("bm", bytes_written)
                            metrics.increment_count("bm_granule_downloads")
                    finally:
                        close = getattr(response, "close", None)
                        if callable(close):
                            close()

                h5_path = materialize_cached_file(h5_path, writer)
            h5_paths.append(h5_path)

        mosaic_array, mosaic_transform, mosaic_profile = build_bm_mosaic_for_bbox(
            h5_paths,
            tile_bounds_lookup,
            bbox=bbox,
            geotiff_cache_dir=bm_geotiff_cache_dir,
            metrics=metrics,
        )
        patch, patch_transform = crop_mosaic_to_bbox(
            mosaic_array, mosaic_transform, bbox, patch_size_pix
        )

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

    return (
        f"Saved mosaic patch: {out_path}",
        BMPatch(
            tile_id=tile_id,
            path=out_path,
            longitude=lon,
            latitude=lat,
            date=date_str,
            population_bin=sample.get("Bin"),
        ),
    )


def process_samples_parallel(
    sample_list: Sequence[dict],
    patch_size_pix: int,
    collection_id: str,
    token: str,
    tile_shapefile_path: Path,
    output_folder: Path,
    bm_cache_dir: Path,
    bm_cmr_cache_dir: Path,
    bm_geotiff_cache_dir: Path,
    max_workers: int = 4,
    metrics: RunMetrics | None = None,
) -> List[BMPatch]:
    output_folder.mkdir(parents=True, exist_ok=True)
    bm_cache_dir.mkdir(parents=True, exist_ok=True)
    bm_cmr_cache_dir.mkdir(parents=True, exist_ok=True)
    bm_geotiff_cache_dir.mkdir(parents=True, exist_ok=True)
    tile_shapefile = gpd.read_file(tile_shapefile_path)
    tile_bounds_lookup = build_tile_bounds_lookup(tile_shapefile)

    results: List[BMPatch] = []
    failures: list[tuple[dict, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(
                process_single_sample,
                sample,
                patch_size_pix,
                collection_id,
                token,
                tile_bounds_lookup,
                output_folder,
                bm_cache_dir,
                bm_cmr_cache_dir,
                bm_geotiff_cache_dir,
                metrics,
            ): sample
            for sample in sample_list
        }
        for future in concurrent.futures.as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                message, patch = future.result()
            except Exception as exc:  # pragma: no cover - unexpected worker error
                failures.append((sample, exc))
                LOGGER.error(
                    "Worker failed for sample %s: %s", sample, exc, exc_info=True
                )
                continue
            LOGGER.info(message)
            if patch is not None:
                results.append(patch)

    if failures:
        LOGGER.warning("Encountered %d failed samples", len(failures))

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


def safe_download(
    s3,
    bucket: str,
    key: str,
    outpath: Path,
    max_retries: int = 5,
    object_size: int | None = None,
    metrics: RunMetrics | None = None,
) -> Path:
    if tqdm is not None and object_size is None:
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            object_size = head.get("ContentLength")
        except Exception as exc:  # pragma: no cover - optional diagnostics
            LOGGER.debug("Unable to determine size for %s: %s", key, exc)

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        progress_bar = None
        try:
            if tqdm is not None:
                desc = f"DMSP {os.path.basename(key)}"
                progress_bar = tqdm(
                    total=object_size,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                    leave=False,
                )

                def _callback(bytes_amount, bar=progress_bar):
                    bar.update(bytes_amount)

                callback = _callback
            else:
                callback = None

            if callback is not None:
                s3.download_file(bucket, key, str(outpath), Callback=callback)
            else:
                s3.download_file(bucket, key, str(outpath))
            wait_for_file_release(outpath)
            if metrics is not None:
                downloaded_size = object_size
                if downloaded_size is None and outpath.exists():
                    downloaded_size = outpath.stat().st_size
                if downloaded_size is not None:
                    metrics.add_bytes("dmsp", downloaded_size)
                metrics.increment_count("dmsp_scene_downloads")
            return outpath
        except botocore.exceptions.EndpointConnectionError as exc:  # pragma: no cover - network
            last_error = exc
            LOGGER.warning(
                "EndpointConnectionError on %s (attempt %s/%s): %s",
                key,
                attempt,
                max_retries,
                exc,
            )
        except botocore.exceptions.ClientError as exc:  # pragma: no cover - network
            last_error = exc
            LOGGER.warning(
                "ClientError on %s (attempt %s/%s): %s",
                key,
                attempt,
                max_retries,
                exc,
            )
        except Exception as exc:  # pragma: no cover - network
            last_error = exc
            LOGGER.warning(
                "Other error on %s (attempt %s/%s): %s",
                key,
                attempt,
                max_retries,
                exc,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()
        time.sleep(2)

    error = last_error or RuntimeError("Unknown download failure")
    LOGGER.error("Failed to download %s after %s attempts", key, max_retries)
    raise DownloadError(bucket, key, max_retries, error)


def reproject_to_bm_grid(src_path: Path, bm_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata if src.nodata is not None else 255
        dst = np.full((bm_profile["height"], bm_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=bm_profile["transform"],
            dst_crs=bm_profile["crs"],
            resampling=WarpResampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    return dst


def compute_patch_correlation_stats(
    bm_patch: np.ndarray,
    dmsp_patch: np.ndarray,
) -> tuple[float | None, float]:
    mask = (~np.isnan(bm_patch)) & (~np.isnan(dmsp_patch))
    total_pixels = int(mask.size)
    valid_pixels = int(mask.sum())
    valid_fraction = (valid_pixels / total_pixels) if total_pixels else 0.0
    if valid_pixels < 2:
        return (None, valid_fraction)

    bm_vals = bm_patch[mask].astype(np.float64, copy=False)
    dmsp_vals = dmsp_patch[mask].astype(np.float64, copy=False)
    bm_centered = bm_vals - bm_vals.mean()
    dmsp_centered = dmsp_vals - dmsp_vals.mean()
    denominator = math.sqrt(
        float(np.dot(bm_centered, bm_centered)) * float(np.dot(dmsp_centered, dmsp_centered))
    )
    if denominator == 0:
        return (None, valid_fraction)
    corr = float(np.dot(bm_centered, dmsp_centered) / denominator)
    if np.isnan(corr):
        return (None, valid_fraction)
    return (corr, valid_fraction)


def compute_patch_correlation(bm_patch: np.ndarray, dmsp_patch: np.ndarray) -> float | None:
    corr, _ = compute_patch_correlation_stats(bm_patch, dmsp_patch)
    return corr


def select_best_dmsp_match(
    bm_patch: BMPatch,
    file_keys: Sequence[tuple[str, int | None]],
    s3,
    bucket_name: str,
    raw_cache_dir: Path,
    dmsp_out_dir: Path,
    min_valid_fraction: float = 0.10,
    metrics: RunMetrics | None = None,
) -> tuple[DMSPMatch | None, List[DownloadFailure]]:
    bm_patch_path = bm_patch.path
    bm_patch_name = bm_patch_path.name
    with rasterio.open(bm_patch_path) as bm_src:
        bm_profile = bm_src.profile.copy()
        bm_shape = (bm_src.height, bm_src.width)
        bm_patch_array = bm_src.read(1).astype(np.float32)
        bm_patch_array[bm_patch_array < 0] = np.nan

    best_corr = None
    best_vis_patch: np.ndarray | None = None
    best_vis_file: Path | None = None
    best_valid_fraction = 0.0
    best_f_number = ""
    best_source_key = ""
    failures: List[DownloadFailure] = []

    for vis_key, object_size in file_keys:
        base = os.path.basename(vis_key)
        f_number = base.split("_")[0] if "_" in base else base[:3]
        vis_file = raw_cache_dir.joinpath(*vis_key.split("/"))
        if vis_file.exists():
            if metrics is not None:
                metrics.record_cache_hit("dmsp_raw")
        else:
            if metrics is not None:
                metrics.record_cache_miss("dmsp_raw")
            LOGGER.info("Downloading %s", vis_key)
            try:
                vis_file = materialize_cached_file(
                    vis_file,
                    lambda temp_path, key=vis_key, size=object_size: safe_download(
                        s3,
                        bucket_name,
                        key,
                        temp_path,
                        object_size=size,
                        metrics=metrics,
                    ),
                )
            except DownloadError as exc:
                LOGGER.warning(
                    "Failed to download %s for tile %s: %s",
                    vis_key,
                    bm_patch.tile_id,
                    exc,
                )
                failures.append(
                    DownloadFailure(
                        tile_id=bm_patch.tile_id,
                        bm_path=bm_patch_path,
                        key=vis_key,
                        error=exc,
                    )
                )
                continue
        try:
            vis_patch = reproject_to_bm_grid(vis_file, bm_profile)
        except Exception as exc:  # pragma: no cover - reprojection failure
            LOGGER.warning("Error processing %s: %s", vis_file, exc)
            continue
        corr, valid_fraction = compute_patch_correlation_stats(bm_patch_array, vis_patch)
        if valid_fraction < min_valid_fraction:
            LOGGER.debug(
                "Skipping %s for %s due to low coverage (%.2f%%)",
                base,
                bm_patch_name,
                valid_fraction * 100,
            )
            continue
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
        return (None, failures)

    out_path = dmsp_out_dir / f"{bm_patch.tile_id}.tif"
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
    return (
        DMSPMatch(
            tile_id=bm_patch.tile_id,
            bm_path=bm_patch_path,
            dmsp_path=out_path,
            f_number=best_f_number,
            correlation=best_corr,
            valid_fraction=best_valid_fraction,
            source_key=best_source_key,
        ),
        failures,
    )


def parallel_process_bm_patch(
    bm_patch: BMPatch,
    file_keys: Sequence[tuple[str, int | None]],
    s3,
    bucket_name: str,
    raw_cache_dir: Path,
    dmsp_out_dir: Path,
    metrics: RunMetrics | None = None,
) -> tuple[List[DMSPMatch], List[DownloadFailure]]:
    if not file_keys:
        LOGGER.info("No DMSP scenes found for %s", bm_patch.path.name)
        return ([], [])
    match, failures = select_best_dmsp_match(
        bm_patch,
        file_keys,
        s3,
        bucket_name,
        raw_cache_dir,
        dmsp_out_dir,
        metrics=metrics,
    )
    return (([match] if match else []), failures)


def load_existing_dmsp_matches(manifest_path: Path) -> dict[str, DMSPMatch]:
    if not manifest_path.exists():
        return {}
    try:
        manifest = pd.read_csv(manifest_path)
    except pd.errors.EmptyDataError:
        return {}

    required_columns = {
        "tile_id",
        "bm_patch",
        "dmsp_patch",
        "f_number",
        "correlation",
        "valid_fraction",
        "dmsp_source_key",
    }
    if not required_columns.issubset(manifest.columns):
        return {}

    existing: dict[str, DMSPMatch] = {}
    for row in manifest.itertuples(index=False):
        dmsp_path = Path(str(row.dmsp_patch))
        if not dmsp_path.exists():
            continue
        try:
            correlation = float(row.correlation)
            valid_fraction = float(row.valid_fraction)
        except (TypeError, ValueError):
            continue
        existing[str(row.tile_id)] = DMSPMatch(
            tile_id=str(row.tile_id),
            bm_path=Path(str(row.bm_patch)),
            dmsp_path=dmsp_path,
            f_number=str(row.f_number),
            correlation=correlation,
            valid_fraction=valid_fraction,
            source_key=str(row.dmsp_source_key),
        )
    return existing


def download_dmsp_matches(
    bm_patches: Sequence[BMPatch],
    dmsp_out_dir: Path,
    raw_cache_dir: Path,
    max_workers: int = 4,
    existing_matches: dict[str, DMSPMatch] | None = None,
    scene_index_cache_dir: Path | None = None,
    metrics: RunMetrics | None = None,
) -> tuple[List[DMSPMatch], List[DownloadFailure]]:
    dmsp_out_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_scene_index_cache_dir = scene_index_cache_dir or (
        DEFAULT_CACHE_ROOT / DMSP_SCENE_INDEX_DIRNAME
    )
    resolved_scene_index_cache_dir.mkdir(parents=True, exist_ok=True)
    s3 = build_s3_client(max_workers=max_workers)
    bucket_name = DMSP_BUCKET_NAME
    saved_matches: List[DMSPMatch] = []
    failures: List[DownloadFailure] = []

    pending_patches: List[BMPatch] = []
    for patch in bm_patches:
        existing = existing_matches.get(patch.tile_id) if existing_matches else None
        if existing is None or not existing.dmsp_path.exists():
            pending_patches.append(patch)
            continue
        saved_matches.append(
            DMSPMatch(
                tile_id=patch.tile_id,
                bm_path=patch.path,
                dmsp_path=existing.dmsp_path,
                f_number=existing.f_number,
                correlation=existing.correlation,
                valid_fraction=existing.valid_fraction,
                source_key=existing.source_key,
            )
        )
    if saved_matches:
        LOGGER.info("Reusing %s existing DMSP matches from %s", len(saved_matches), dmsp_out_dir)
    if not pending_patches:
        return saved_matches, failures

    unique_dates = sorted({patch.date.replace("-", "") for patch in pending_patches})
    if metrics is not None:
        metrics.set_count("dmsp_unique_dates", len(unique_dates))
        metrics.set_count("dmsp_unique_years", len({date_str[:4] for date_str in unique_dates}))
    file_keys_by_date = list_dmsp_scene_keys_for_dates(
        unique_dates,
        s3,
        bucket_name,
        resolved_scene_index_cache_dir,
        metrics=metrics,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for patch in pending_patches:
            futures.append(
                executor.submit(
                    parallel_process_bm_patch,
                    patch,
                    file_keys_by_date.get(patch.date.replace("-", ""), []),
                    s3,
                    bucket_name,
                    raw_cache_dir,
                    dmsp_out_dir,
                    metrics,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            matches, download_failures = future.result()
            saved_matches.extend(matches)
            failures.extend(download_failures)
    return saved_matches, failures


def create_pair_manifest(
    bm_patches: Sequence[BMPatch],
    dmsp_matches: Sequence[DMSPMatch],
    manifest_path: Path,
) -> pd.DataFrame:
    bm_lookup = {patch.tile_id: patch for patch in bm_patches}
    rows = []
    for match in dmsp_matches:
        bm_patch = bm_lookup.get(match.tile_id)
        if bm_patch is None:
            LOGGER.debug(
                "Skipping manifest entry for %s because BM patch is missing",
                match.tile_id,
            )
            continue
        rows.append(
            {
                "tile_id": match.tile_id,
                "bm_patch": str(bm_patch.path),
                "dmsp_patch": str(match.dmsp_path),
                "population_bin": bm_patch.population_bin,
                "longitude": bm_patch.longitude,
                "latitude": bm_patch.latitude,
                "date": bm_patch.date,
                "f_number": match.f_number,
                "correlation": match.correlation,
                "valid_fraction": match.valid_fraction,
                "dmsp_source_key": match.source_key,
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        LOGGER.info("Wrote manifest to %s", manifest_path)
    else:
        LOGGER.info("No paired BM/DMSP tiles to record in manifest")
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
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Shared cache directory for reusable BM and DMSP source files",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Use the existing locations CSV instead of regenerating samples",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Generate or normalize the locations CSV and dates, then exit before BM/DMSP downloads",
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
    cache_root = args.cache_root.expanduser()

    bm_dir = output_root / "bm" if output_root else BM_OUTPUT_DIR
    dmsp_dir = output_root / "dmsp" if output_root else DMSP_OUTPUT_DIR
    plots_dir = output_root / "plots" if output_root else PLOTS_DIR
    bm_cache_dir = cache_root / BM_GRANULE_CACHE_DIRNAME
    bm_cmr_cache_dir = cache_root / BM_CMR_CACHE_DIRNAME
    bm_geotiff_cache_dir = cache_root / BM_GEOTIFF_CACHE_DIRNAME
    dmsp_raw_cache_dir = cache_root / DMSP_RAW_CACHE_DIRNAME
    dmsp_scene_index_dir = cache_root / DMSP_SCENE_INDEX_DIRNAME
    dmsp_dates_cache = cache_root / DEFAULT_DMSP_DATES_CACHE.name
    timings_root = output_root if output_root else PAIR_OUTPUT_ROOT
    timings_path = timings_root / TIMINGS_FILENAME

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

    metrics = RunMetrics()
    metrics.set_metadata(
        "parameters",
        {
            "cache_root": str(cache_root),
            "collection_id": args.collection_id,
            "locations_csv": str(locations_csv),
            "manifest": str(manifest_path),
            "max_workers": args.max_workers,
            "output_folder": str(output_root) if output_root is not None else None,
            "patch_size": args.patch_size,
            "sample_only": args.sample_only,
            "samples_per_bin": args.samples_per_bin,
            "sampling_seed": args.sampling_seed,
            "date_seed": args.date_seed,
        },
    )
    total_start = time.perf_counter()

    try:
        if args.skip_sampling and not locations_csv.exists():
            raise FileNotFoundError(f"--skip-sampling provided but {locations_csv} does not exist")

        with metrics.measure("sampling"):
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

        if "date" not in df.columns:
            with metrics.measure("date_assignment"):
                dmsp_dates = get_dmsp_dates(cache_path=dmsp_dates_cache, metrics=metrics)
                df = assign_random_dates(df, dmsp_dates, seed=args.date_seed)
                locations_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(locations_csv, index=False)
                LOGGER.info("Wrote sampled locations with dates to %s", locations_csv)
        else:
            df = df.copy()

        metrics.set_count("sample_rows", len(df))
        if "date" in df.columns:
            metrics.set_count("dated_rows", int(df["date"].notna().sum()))
            metrics.set_count("unique_dates", len({str(value) for value in df["date"].dropna()}))
        else:
            metrics.set_count("dated_rows", 0)
            metrics.set_count("unique_dates", 0)

        if args.sample_only:
            LOGGER.info("Sample-only mode enabled; skipping BM and DMSP downloads")
            return

        sample_columns = ["Longitude", "Latitude", "date"]
        if "Bin" in df.columns:
            sample_columns.append("Bin")
        base_samples = df[sample_columns].to_dict(orient="records")
        sample_list = []
        for idx, sample in enumerate(base_samples, start=1):
            tile_id = f"tile_{idx:03d}"
            enriched = dict(sample)
            enriched["tile_id"] = tile_id
            sample_list.append(enriched)
        metrics.set_count("sample_tiles_requested", len(sample_list))

        token = load_nasa_token()
        with metrics.measure("bm_processing"):
            bm_patches = process_samples_parallel(
                sample_list=sample_list,
                patch_size_pix=args.patch_size,
                collection_id=args.collection_id,
                token=token,
                tile_shapefile_path=DEFAULT_TILE_SHAPEFILE,
                output_folder=bm_dir,
                bm_cache_dir=bm_cache_dir,
                bm_cmr_cache_dir=bm_cmr_cache_dir,
                bm_geotiff_cache_dir=bm_geotiff_cache_dir,
                max_workers=args.max_workers,
                metrics=metrics,
            )
        metrics.set_count("bm_patches", len(bm_patches))

        existing_dmsp_matches = load_existing_dmsp_matches(manifest_path)
        metrics.set_count("existing_dmsp_matches", len(existing_dmsp_matches))
        with metrics.measure("dmsp_processing"):
            dmsp_matches, download_failures = download_dmsp_matches(
                bm_patches=bm_patches,
                dmsp_out_dir=dmsp_dir,
                raw_cache_dir=dmsp_raw_cache_dir,
                max_workers=args.max_workers,
                existing_matches=existing_dmsp_matches,
                scene_index_cache_dir=dmsp_scene_index_dir,
                metrics=metrics,
            )
        metrics.set_count("dmsp_matches", len(dmsp_matches))
        metrics.set_count("download_failures", len(download_failures))

        LOGGER.info(
            "Downloaded %s BM patches and %s DMSP patches",
            len(bm_patches),
            len(dmsp_matches),
        )
        if download_failures:
            failed_tiles = {failure.tile_id for failure in download_failures}
            LOGGER.warning(
                "Encountered %s download failures across %s tiles",
                len(download_failures),
                len(failed_tiles),
            )
            for failure in download_failures:
                LOGGER.warning(
                    "Tile %s (%s) failed to download %s: %s",
                    failure.tile_id,
                    failure.bm_path.name,
                    failure.key,
                    failure.error,
                )

        with metrics.measure("manifest_write"):
            manifest = create_pair_manifest(bm_patches, dmsp_matches, manifest_path)
        metrics.set_count("manifest_rows", len(manifest))

        with metrics.measure("cleanup"):
            manifest_df = manifest
            if manifest_path.exists():
                try:
                    manifest_df = pd.read_csv(manifest_path)
                except pd.errors.EmptyDataError:
                    manifest_df = pd.DataFrame()

            expected_bm: set[str] = set()
            expected_dmsp: set[str] = {Path(match.dmsp_path).name for match in dmsp_matches}
            if "bm_patch" in manifest_df.columns:
                expected_bm.update(
                    Path(path).name
                    for path in manifest_df["bm_patch"].dropna().astype(str)
                )
            if "dmsp_patch" in manifest_df.columns:
                expected_dmsp.update(
                    Path(path).name
                    for path in manifest_df["dmsp_patch"].dropna().astype(str)
                )

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
                    "Removed %s unmatched BM rasters and %s unmatched DMSP rasters",
                    bm_removed,
                    dmsp_removed,
                )
            else:
                LOGGER.info("No unmatched BM or DMSP rasters were removed")

        metrics.set_count("removed_bm_rasters", bm_removed)
        metrics.set_count("removed_dmsp_rasters", dmsp_removed)
    finally:
        metrics.add_stage_time("total", time.perf_counter() - total_start)
        metrics.write(timings_path)
        LOGGER.info("Wrote timings to %s", timings_path)


if __name__ == "__main__":
    main()
