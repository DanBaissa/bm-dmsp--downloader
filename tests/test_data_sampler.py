import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dummy_botocore = types.ModuleType("botocore")
dummy_botocore.UNSIGNED = object()
dummy_botocore.exceptions = types.ModuleType("botocore.exceptions")
dummy_botocore.exceptions.EndpointConnectionError = Exception
dummy_botocore.exceptions.ClientError = Exception
dummy_botocore.config = types.ModuleType("botocore.config")
dummy_botocore.config.Config = type("Config", (), {})

sys.modules.setdefault("botocore", dummy_botocore)
sys.modules.setdefault("botocore.exceptions", dummy_botocore.exceptions)
sys.modules.setdefault("botocore.config", dummy_botocore.config)

dummy_boto3 = types.ModuleType("boto3")
dummy_boto3.client = lambda *args, **kwargs: None
dummy_boto3.resource = lambda *args, **kwargs: None
sys.modules.setdefault("boto3", dummy_boto3)

dummy_matplotlib = types.ModuleType("matplotlib")
dummy_matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
dummy_matplotlib.pyplot.figure = lambda *args, **kwargs: None
dummy_matplotlib.pyplot.savefig = lambda *args, **kwargs: None
dummy_matplotlib.pyplot.close = lambda *args, **kwargs: None
sys.modules.setdefault("matplotlib", dummy_matplotlib)
sys.modules.setdefault("matplotlib.pyplot", dummy_matplotlib.pyplot)

dummy_geopandas = types.ModuleType("geopandas")
dummy_geopandas.GeoDataFrame = type("GeoDataFrame", (), {})
dummy_geopandas.read_file = lambda *args, **kwargs: None
sys.modules.setdefault("geopandas", dummy_geopandas)

dummy_h5py = types.ModuleType("h5py")
class _UnavailableFile:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("h5py is not available in tests")

dummy_h5py.File = _UnavailableFile
sys.modules.setdefault("h5py", dummy_h5py)

try:  # pragma: no cover - exercised when numpy is available
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency absent
    dummy_numpy = types.ModuleType("numpy")
    dummy_numpy.float32 = float
    dummy_numpy.nan = float("nan")
    sys.modules.setdefault("numpy", dummy_numpy)

try:  # pragma: no cover - exercised when pandas is available
    import pandas  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency absent
    dummy_pandas = types.ModuleType("pandas")
    dummy_pandas.DataFrame = type("DataFrame", (), {})
    dummy_pandas.Series = type("Series", (), {})
    sys.modules.setdefault("pandas", dummy_pandas)

dummy_rasterio = types.ModuleType("rasterio")


class _DummyArray:
    def __init__(self, shape=(1, 1), fill_value=0.0):
        self.shape = shape
        self.fill_value = fill_value

    def astype(self, *args, **kwargs):
        return self

    def copy(self):
        return self

    def __getitem__(self, item):
        return self

class _DummyWindow:
    def __init__(self, *args, **kwargs):
        self.col_off = kwargs.get("col_off", 0)
        self.row_off = kwargs.get("row_off", 0)
        self.width = kwargs.get("width", 0)
        self.height = kwargs.get("height", 0)

    def round_offsets(self):
        return self

    def round_lengths(self):
        return self

    def intersection(self, other):
        return self

dummy_rasterio.windows = types.ModuleType("rasterio.windows")
dummy_rasterio.windows.from_bounds = lambda *args, **kwargs: _DummyWindow()
dummy_rasterio.windows.Window = _DummyWindow
dummy_rasterio.windows.transform = lambda window, transform: transform

dummy_rasterio.transform = types.ModuleType("rasterio.transform")
dummy_rasterio.transform.from_bounds = lambda *args, **kwargs: None
dummy_rasterio.transform.from_origin = lambda *args, **kwargs: None

dummy_rasterio.merge = types.ModuleType("rasterio.merge")
dummy_rasterio.merge.merge = lambda datasets, nodata=None: (None, None)

dummy_rasterio.io = types.ModuleType("rasterio.io")

class _DummyMemoryFile:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def open(self, **profile):
        return types.SimpleNamespace(
            write=lambda *a, **k: None,
            read=lambda *a, **k: _DummyArray(),
            width=profile.get("width", 0),
            height=profile.get("height", 0),
            transform=profile.get("transform"),
        )

dummy_rasterio.io.MemoryFile = _DummyMemoryFile

dummy_rasterio.enums = types.ModuleType("rasterio.enums")
dummy_rasterio.enums.Resampling = type("Resampling", (), {})

dummy_rasterio.features = types.ModuleType("rasterio.features")
dummy_rasterio.features.rasterize = lambda *args, **kwargs: None

dummy_rasterio.warp = types.ModuleType("rasterio.warp")
dummy_rasterio.warp.reproject = lambda *args, **kwargs: None
dummy_rasterio.warp.Resampling = type("Resampling", (), {"bilinear": "bilinear"})
dummy_rasterio.band = lambda src, index: (src, index)

dummy_rasterio.open = lambda *args, **kwargs: types.SimpleNamespace(
    profile={},
    close=lambda: None,
    read=lambda *a, **k: _DummyArray(),
    height=1,
    width=1,
    transform=None,
    crs=None,
    nodata=None,
)

sys.modules.setdefault("rasterio", dummy_rasterio)
sys.modules.setdefault("rasterio.windows", dummy_rasterio.windows)
sys.modules.setdefault("rasterio.transform", dummy_rasterio.transform)
sys.modules.setdefault("rasterio.merge", dummy_rasterio.merge)
sys.modules.setdefault("rasterio.io", dummy_rasterio.io)
sys.modules.setdefault("rasterio.enums", dummy_rasterio.enums)
sys.modules.setdefault("rasterio.features", dummy_rasterio.features)
sys.modules.setdefault("rasterio.warp", dummy_rasterio.warp)

dummy_requests = types.ModuleType("requests")
dummy_requests.HTTPError = Exception
dummy_requests.RequestException = Exception
dummy_requests.get = lambda *args, **kwargs: None
dummy_requests.Session = lambda: types.SimpleNamespace(
    get=dummy_requests.get,
    mount=lambda *args, **kwargs: None,
)
sys.modules.setdefault("requests", dummy_requests)

import data_sampler

rasterio = data_sampler.rasterio


class DummyResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict | None = None,
        raise_error: bool = False,
        content: bytes | None = None,
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self._raise_error = raise_error
        self._content = content or b""
        self.headers = {"Content-Length": str(len(self._content))}

    def raise_for_status(self):
        if self._raise_error or self.status_code >= 400:
            raise data_sampler.requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=8192):
        if not self._content:
            yield from []
            return
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def close(self):  # pragma: no cover - no resources to release
        return None


def test_search_nasa_cmr_handles_dateline_split(monkeypatch):
    calls: list[list[float]] = []

    def fake_get(url, params, timeout=30):
        bbox_vals = [float(v) for v in params["bounding_box"].split(",")]
        calls.append(bbox_vals)
        if bbox_vals[0] > 0:
            return DummyResponse(400, raise_error=True)
        payload = {
            "feed": {
                "entry": [
                    {"links": [{"href": "https://example.com/file1.h5"}]},
                    {
                        "links": [
                            {"href": "https://example.com/file1.h5"},
                            {"href": "https://example.com/file2.txt"},
                        ]
                    },
                    {"links": [{"href": "https://example.com/file2.h5"}]},
                ]
            }
        }
        return DummyResponse(200, payload)

    monkeypatch.setattr(
        data_sampler,
        "get_http_session",
        lambda: types.SimpleNamespace(get=fake_get),
    )

    links = data_sampler.search_nasa_cmr(
        "collection",
        "2014-08-28",
        [179.5, -10.0, 181.5, 10.0],
    )

    assert links == ["https://example.com/file1.h5", "https://example.com/file2.h5"]
    assert len(calls) == 2


def test_process_samples_parallel_continues_after_failure(monkeypatch, tmp_path):
    def fake_read_file(_):
        return types.SimpleNamespace(
            columns=["TileID", "geometry"],
            __getitem__=lambda self, key: self,
            itertuples=lambda index=False: [],
        )

    monkeypatch.setattr(data_sampler.gpd, "read_file", fake_read_file)
    monkeypatch.setattr(data_sampler, "build_tile_bounds_lookup", lambda *_: {})

    def fake_worker(sample, *args, **kwargs):
        if sample["id"] == "fail":
            raise RuntimeError("boom")
        patch_path = tmp_path / f"{sample['tile_id']}.tif"
        return (
            f"ok {sample['id']}",
            data_sampler.BMPatch(
                tile_id=sample["tile_id"],
                path=patch_path,
                longitude=0.0,
                latitude=0.0,
                date="2020-01-01",
            ),
        )

    monkeypatch.setattr(data_sampler, "process_single_sample", fake_worker)

    samples = [
        {"id": "a", "tile_id": "tile_001"},
        {"id": "fail", "tile_id": "tile_002"},
        {"id": "b", "tile_id": "tile_003"},
    ]
    results = data_sampler.process_samples_parallel(
        samples,
        patch_size_pix=10,
        collection_id="cid",
        token="token",
        tile_shapefile_path=Path("dummy.shp"),
        output_folder=tmp_path / "out",
        bm_cache_dir=tmp_path / "cache",
        bm_cmr_cache_dir=tmp_path / "cmr_cache",
        bm_geotiff_cache_dir=tmp_path / "geotiff_cache",
        max_workers=2,
    )

    assert sorted(patch.tile_id for patch in results) == ["tile_001", "tile_003"]


def test_safe_download_raises_download_error(monkeypatch, tmp_path):
    class FakeS3:
        def __init__(self):
            self.calls = 0

        def head_object(self, Bucket, Key):
            return {"ContentLength": 1024}

        def download_file(self, bucket, key, filename, Callback=None):
            self.calls += 1
            raise ValueError("network down")

    fake_s3 = FakeS3()
    monkeypatch.setattr(data_sampler, "tqdm", None)
    monkeypatch.setattr(data_sampler.time, "sleep", lambda *_: None)

    with pytest.raises(data_sampler.DownloadError) as excinfo:
        data_sampler.safe_download(
            fake_s3,
            "bucket",
            "object",
            tmp_path / "out.bin",
            max_retries=2,
        )

    assert "Failed to download" in str(excinfo.value)
    assert fake_s3.calls == 2


def test_select_best_dmsp_match_collects_download_failures(monkeypatch, tmp_path):
    bm_path = tmp_path / "bm" / "tile_001.tif"
    bm_path.parent.mkdir(parents=True, exist_ok=True)
    bm_path.touch()

    bm_patch = data_sampler.BMPatch(
        tile_id="tile_001",
        path=bm_path,
        longitude=0.0,
        latitude=0.0,
        date="2020-01-01",
    )

    class FakeDataset:
        def __init__(self):
            self.profile = {
                "height": 1,
                "width": 1,
                "transform": None,
                "crs": None,
            }
            self.height = 1
            self.width = 1

        def read(self, *args, **kwargs):
            import numpy as np

            return np.ones((1, 1), dtype=np.float32)

        def write(self, data, index):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(path, mode="r", **kwargs):
        if mode == "w":
            dataset = FakeDataset()

            def write(data, index):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"fake")

            dataset.write = write  # type: ignore[attr-defined]
            return dataset
        return FakeDataset()

    monkeypatch.setattr(data_sampler.rasterio, "open", fake_open)

    def fake_safe_download(s3, bucket, key, outpath, max_retries=5, object_size=None, metrics=None):
        raise data_sampler.DownloadError(bucket, key, max_retries, RuntimeError("oops"))

    monkeypatch.setattr(data_sampler, "safe_download", fake_safe_download)

    match, failures = data_sampler.select_best_dmsp_match(
        bm_patch,
        [("F101_example.vis.co.tif", None)],
        s3=None,
        bucket_name="bucket",
        raw_cache_dir=tmp_path / "dl",
        dmsp_out_dir=tmp_path / "dmsp",
    )

    assert match is None
    assert len(failures) == 1
    failure = failures[0]
    assert failure.tile_id == "tile_001"
    assert failure.key == "F101_example.vis.co.tif"


def test_select_best_dmsp_match_uses_tile_id_for_outputs(monkeypatch, tmp_path):
    bm_path = tmp_path / "bm" / "tile_002.tif"
    bm_path.parent.mkdir(parents=True, exist_ok=True)
    bm_path.touch()

    bm_patch = data_sampler.BMPatch(
        tile_id="tile_002",
        path=bm_path,
        longitude=0.0,
        latitude=0.0,
        date="2020-01-01",
    )

    class FakeDataset:
        def __init__(self):
            self.profile = {
                "height": 1,
                "width": 1,
                "transform": None,
                "crs": None,
            }
            self.height = 1
            self.width = 1

        def read(self, *args, **kwargs):
            import numpy as np

            return np.ones((1, 1), dtype=np.float32)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(path, mode="r", **kwargs):
        dataset = FakeDataset()
        if mode == "w":
            def write(data, index):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"fake")

            dataset.write = write  # type: ignore[attr-defined]
        return dataset

    monkeypatch.setattr(data_sampler.rasterio, "open", fake_open)
    monkeypatch.setattr(
        data_sampler,
        "reproject_to_bm_grid",
        lambda *a, **k: __import__("numpy").ones((1, 1), dtype=float),
    )
    monkeypatch.setattr(
        data_sampler,
        "compute_patch_correlation_stats",
        lambda *a, **k: (0.9, 1.0),
    )

    def fake_safe_download(s3, bucket, key, outpath, max_retries=5, object_size=None, metrics=None):
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_bytes(b"data")
        return outpath

    monkeypatch.setattr(data_sampler, "safe_download", fake_safe_download)

    download_dir = tmp_path / "dl"
    dmsp_dir = tmp_path / "dmsp"

    match, failures = data_sampler.select_best_dmsp_match(
        bm_patch,
        [("F141_example.vis.co.tif", None)],
        s3=None,
        bucket_name="bucket",
        raw_cache_dir=download_dir,
        dmsp_out_dir=dmsp_dir,
    )

    assert not failures
    assert match is not None
    assert match.dmsp_path.name == "tile_002.tif"
    assert match.f_number == "F141"
    assert match.source_key == "F141_example.vis.co.tif"
    assert match.dmsp_path.exists()


def test_build_bm_mosaic_skips_missing_tiles(monkeypatch, tmp_path):
    merge_calls = []

    class FakeBand:
        shape = (1, 1)

        def astype(self, *args, **kwargs):
            return self

    class FakeMosaic:
        shape = (1, 1, 1)

        def __getitem__(self, item):
            return FakeBand()

    class FakeDataset:
        def __init__(self):
            self.profile = {"dummy": True}

        def close(self):
            pass

    def fake_open(path, *args, **kwargs):
        return FakeDataset()

    def fake_merge(datasets, nodata=None, bounds=None):
        merge_calls.append({"nodata": nodata, "bounds": bounds})
        return FakeMosaic(), "transform"

    monkeypatch.setattr(data_sampler, "rio_merge", fake_merge)
    monkeypatch.setattr(data_sampler.rasterio, "open", fake_open)

    def fake_h5_to_geotiff(path, tile_bounds_lookup):
        if path.stem == "missing":
            raise data_sampler.TileMetadataMissingError("no tile metadata")
        tif_path = tmp_path / f"{path.stem}.tif"
        tif_path.touch()
        return tif_path

    monkeypatch.setattr(data_sampler, "h5_to_geotiff", fake_h5_to_geotiff)

    mosaic, transform, profile = data_sampler.build_bm_mosaic_for_bbox(
        [Path("missing.h5"), Path("valid.h5")],
        tile_bounds_lookup={},
        bbox=[1.0, 2.0, 3.0, 4.0],
    )

    assert mosaic.shape == (1, 1)
    assert profile["dtype"] == "float32"
    assert len(merge_calls) == 1
    assert merge_calls[0]["bounds"] == (1.0, 2.0, 3.0, 4.0)
    assert merge_calls[0]["nodata"] != merge_calls[0]["nodata"]

    with pytest.raises(RuntimeError, match="No valid Black Marble tiles"):
        data_sampler.build_bm_mosaic_for_bbox(
            [Path("missing.h5")],
            tile_bounds_lookup={},
        )


def test_resolve_cli_path_defaults_with_output_root(tmp_path):
    output_root = tmp_path / "outputs"
    default = data_sampler.DEFAULT_MANIFEST
    resolved = data_sampler.resolve_cli_path(output_root, default, default, default.name)
    assert resolved == output_root / default.name


def test_get_dmsp_dates_prefers_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "dmsp_dates.txt"
    cache_path.write_text("20120120\n20120121\n", encoding="utf-8")
    monkeypatch.setattr(
        data_sampler,
        "list_dmsp_dates",
        lambda *args, **kwargs: pytest.fail("cache hit should avoid listing the bucket"),
    )

    assert data_sampler.get_dmsp_dates(cache_path=cache_path) == ["20120120", "20120121"]


def test_main_skips_dmsp_date_scan_when_locations_already_have_dates(monkeypatch, tmp_path):
    locations_csv = tmp_path / "locations.csv"
    locations_csv.write_text("Longitude,Latitude,date\n1.0,2.0,2012-01-20\n", encoding="utf-8")
    output_root = tmp_path / "out"

    monkeypatch.setattr(data_sampler, "load_nasa_token", lambda: "token")
    monkeypatch.setattr(
        data_sampler,
        "process_samples_parallel",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        data_sampler,
        "download_dmsp_matches",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(
        data_sampler,
        "create_pair_manifest",
        lambda *args, **kwargs: __import__("pandas").DataFrame(),
    )
    monkeypatch.setattr(
        data_sampler,
        "list_dmsp_dates",
        lambda: pytest.fail("list_dmsp_dates should not be called when dates already exist"),
    )

    data_sampler.main(
        [
            "--skip-sampling",
            "--locations-csv",
            str(locations_csv),
            "--output-folder",
            str(output_root),
        ]
    )


def test_main_persists_assigned_dates_to_locations_csv(monkeypatch, tmp_path):
    locations_csv = tmp_path / "locations.csv"
    locations_csv.write_text("Longitude,Latitude\n1.0,2.0\n", encoding="utf-8")
    output_root = tmp_path / "out"

    monkeypatch.setattr(data_sampler, "load_nasa_token", lambda: "token")
    monkeypatch.setattr(data_sampler, "list_dmsp_dates", lambda *args, **kwargs: ["20120120"])
    monkeypatch.setattr(
        data_sampler,
        "process_samples_parallel",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        data_sampler,
        "download_dmsp_matches",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(
        data_sampler,
        "create_pair_manifest",
        lambda *args, **kwargs: __import__("pandas").DataFrame(),
    )

    data_sampler.main(
        [
            "--skip-sampling",
            "--locations-csv",
            str(locations_csv),
            "--output-folder",
            str(output_root),
        ]
    )

    persisted = __import__("pandas").read_csv(locations_csv)
    assert list(persisted["date"]) == ["2012-01-20"]


def test_download_dmsp_matches_reuses_existing_outputs_and_lists_per_unique_date(monkeypatch, tmp_path):
    bm_dir = tmp_path / "bm"
    bm_dir.mkdir()
    patches = []
    for tile_id, date_str in [
        ("tile_001", "2012-01-20"),
        ("tile_002", "2012-01-20"),
        ("tile_003", "2012-01-21"),
    ]:
        bm_path = bm_dir / f"{tile_id}.tif"
        bm_path.write_bytes(b"bm")
        patches.append(
            data_sampler.BMPatch(
                tile_id=tile_id,
                path=bm_path,
                longitude=0.0,
                latitude=0.0,
                date=date_str,
            )
        )

    existing_dmsp = tmp_path / "dmsp" / "tile_001.tif"
    existing_dmsp.parent.mkdir(parents=True, exist_ok=True)
    existing_dmsp.write_bytes(b"dmsp")
    existing_matches = {
        "tile_001": data_sampler.DMSPMatch(
            tile_id="tile_001",
            bm_path=patches[0].path,
            dmsp_path=existing_dmsp,
            f_number="F12",
            correlation=0.8,
            valid_fraction=0.5,
            source_key="F122012/F12201201200000.night.OIS.vis.co.tif",
        )
    }

    list_calls = []

    class FakeS3:
        pass

    monkeypatch.setattr(data_sampler, "Config", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_sampler.boto3, "client", lambda *args, **kwargs: FakeS3())

    def fake_list_dmsp_scene_keys_for_dates(date_strs, s3, bucket_name, cache_dir, metrics=None):
        result = {}
        for date_str in sorted(date_strs):
            list_calls.append(date_str)
            result[date_str] = [(f"{date_str}.vis.co.tif", 123)]
        return result

    monkeypatch.setattr(
        data_sampler,
        "list_dmsp_scene_keys_for_dates",
        fake_list_dmsp_scene_keys_for_dates,
    )

    def fake_parallel_process_bm_patch(
        bm_patch,
        file_keys,
        s3,
        bucket_name,
        raw_cache_dir,
        dmsp_out_dir,
        metrics=None,
    ):
        out_path = dmsp_out_dir / f"{bm_patch.tile_id}.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"dmsp")
        return (
            [
                data_sampler.DMSPMatch(
                    tile_id=bm_patch.tile_id,
                    bm_path=bm_patch.path,
                    dmsp_path=out_path,
                    f_number="F13",
                    correlation=0.9,
                    valid_fraction=0.6,
                    source_key=file_keys[0][0],
                )
            ],
            [],
        )

    monkeypatch.setattr(data_sampler, "parallel_process_bm_patch", fake_parallel_process_bm_patch)

    matches, failures = data_sampler.download_dmsp_matches(
        bm_patches=patches,
        dmsp_out_dir=tmp_path / "dmsp",
        raw_cache_dir=tmp_path / "raw_cache",
        max_workers=1,
        existing_matches=existing_matches,
        scene_index_cache_dir=tmp_path / "scene_index",
    )

    assert not failures
    assert sorted(match.tile_id for match in matches) == ["tile_001", "tile_002", "tile_003"]
    assert sorted(list_calls) == ["20120120", "20120121"]


def test_list_dmsp_scene_keys_for_dates_reuses_year_cache(tmp_path):
    objects = [
        ("F102012/F10201201200000.night.OIS.vis.co.tif", 11),
        ("F112012/F11201201210000.night.OIS.vis.co.tif", 12),
        ("F102013/F10201301010000.night.OIS.vis.co.tif", 13),
    ]

    class FakePaginator:
        def paginate(self, Bucket, Prefix):
            matching = [
                {"Key": key, "Size": size}
                for key, size in objects
                if key.startswith(Prefix)
            ]
            return [{"Contents": matching}]

    class FakeS3:
        def __init__(self):
            self.calls = 0

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            self.calls += 1
            return FakePaginator()

    s3 = FakeS3()
    cache_dir = tmp_path / "scene_index"

    first = data_sampler.list_dmsp_scene_keys_for_dates(
        ["20120120", "20120121"],
        s3,
        "bucket",
        cache_dir,
    )
    assert first["20120120"] == [("F102012/F10201201200000.night.OIS.vis.co.tif", 11)]
    assert first["20120121"] == [("F112012/F11201201210000.night.OIS.vis.co.tif", 12)]
    assert s3.calls == len(data_sampler.DMSP_SATELLITES)

    second = data_sampler.list_dmsp_scene_keys_for_dates(
        ["20120121"],
        s3,
        "bucket",
        cache_dir,
    )
    assert second["20120121"] == [("F112012/F11201201210000.night.OIS.vis.co.tif", 12)]
    assert s3.calls == len(data_sampler.DMSP_SATELLITES)


def test_get_cached_bm_granule_urls_reuses_cache(monkeypatch, tmp_path):
    calls = []

    def fake_search(collection_id, date_str, bbox, session=None):
        calls.append((collection_id, date_str, tuple(bbox)))
        return [
            "https://example.com/VNP_fake_h11v04.h5",
            "https://example.com/VNP_fake_h99v99.h5",
        ]

    monkeypatch.setattr(data_sampler, "search_nasa_cmr", fake_search)

    cache_dir = tmp_path / "bm_cmr"
    required_tile_ids = ["h11v04"]
    first = data_sampler.get_cached_bm_granule_urls(
        "collection",
        "2012-01-20",
        [-1.0, -1.0, 1.0, 1.0],
        required_tile_ids,
        session=object(),
        cache_dir=cache_dir,
    )
    second = data_sampler.get_cached_bm_granule_urls(
        "collection",
        "2012-01-20",
        [-1.0, -1.0, 1.0, 1.0],
        required_tile_ids,
        session=object(),
        cache_dir=cache_dir,
    )

    assert first == ["https://example.com/VNP_fake_h11v04.h5"]
    assert second == first
    assert len(calls) == 1


def test_reproject_to_bm_grid_uses_raster_band_source(monkeypatch):
    captured = {}
    band_token = object()

    class FakeDataset:
        def __init__(self):
            self.transform = "src_transform"
            self.crs = "EPSG:4326"
            self.nodata = 255

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(data_sampler.rasterio, "open", lambda path: FakeDataset())
    monkeypatch.setattr(data_sampler.rasterio, "band", lambda src, index: band_token)

    def fake_reproject(**kwargs):
        captured.update(kwargs)
        kwargs["destination"][:] = 7.0

    monkeypatch.setattr(data_sampler, "reproject", fake_reproject)

    bm_profile = {"height": 2, "width": 3, "transform": "dst_transform", "crs": "EPSG:4326"}
    result = data_sampler.reproject_to_bm_grid(Path("source.tif"), bm_profile)

    assert result.shape == (2, 3)
    assert captured["source"] is band_token
    assert captured["src_nodata"] == 255
    assert (result == 7.0).all()


def test_compute_patch_correlation_stats_matches_np_corrcoef():
    import numpy as np

    bm_patch = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32)
    dmsp_patch = np.array([[2.0, 4.0, np.nan], [3.0, 9.0, 12.0]], dtype=np.float32)

    mask = (~np.isnan(bm_patch)) & (~np.isnan(dmsp_patch))
    expected_corr = float(np.corrcoef(bm_patch[mask], dmsp_patch[mask])[0, 1])

    corr, valid_fraction = data_sampler.compute_patch_correlation_stats(bm_patch, dmsp_patch)

    assert corr == pytest.approx(expected_corr)
    assert valid_fraction == pytest.approx(mask.sum() / mask.size)


def test_main_sample_only_benchmark_fixture_writes_timings(monkeypatch, tmp_path):
    fixture_path = ROOT / "tests" / "fixtures" / "benchmark_locations.csv"
    locations_csv = tmp_path / "benchmark_locations.csv"
    locations_csv.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")
    output_root = tmp_path / "benchmark_run"

    monkeypatch.setattr(
        data_sampler,
        "load_nasa_token",
        lambda: pytest.fail("sample-only mode should not request the NASA token"),
    )
    monkeypatch.setattr(
        data_sampler,
        "process_samples_parallel",
        lambda **kwargs: pytest.fail("sample-only mode should not download BM patches"),
    )
    monkeypatch.setattr(
        data_sampler,
        "download_dmsp_matches",
        lambda **kwargs: pytest.fail("sample-only mode should not download DMSP patches"),
    )

    data_sampler.main(
        [
            "--skip-sampling",
            "--sample-only",
            "--locations-csv",
            str(locations_csv),
            "--output-folder",
            str(output_root),
        ]
    )

    timings = json.loads((output_root / data_sampler.TIMINGS_FILENAME).read_text(encoding="utf-8"))
    persisted = __import__("pandas").read_csv(locations_csv)

    assert list(persisted["date"]) == ["2012-01-20"]
    assert timings["counts"]["sample_rows"] == 1
    assert timings["metadata"]["parameters"]["sample_only"] is True
    assert "sampling" in timings["stage_seconds"]


def test_resolve_cli_path_respects_relative_overrides(tmp_path):
    output_root = tmp_path / "outputs"
    custom = Path("nested/output.csv")
    resolved = data_sampler.resolve_cli_path(
        output_root, custom, data_sampler.DEFAULT_MANIFEST, data_sampler.DEFAULT_MANIFEST.name
    )
    assert resolved == output_root / custom


def test_resolve_cli_path_keeps_absolute_paths(tmp_path):
    output_root = tmp_path / "outputs"
    absolute = (tmp_path / "explicit.csv").resolve()
    resolved = data_sampler.resolve_cli_path(
        output_root, absolute, data_sampler.DEFAULT_MANIFEST, data_sampler.DEFAULT_MANIFEST.name
    )
    assert resolved == absolute


def test_materialize_cached_file_reuses_existing_artifact(tmp_path):
    target = tmp_path / "cache" / "artifact.bin"
    calls = []

    def writer(temp_path):
        calls.append(temp_path)
        temp_path.write_bytes(b"payload")

    first = data_sampler.materialize_cached_file(target, writer)
    second = data_sampler.materialize_cached_file(target, writer)

    assert first == target
    assert second == target
    assert target.read_bytes() == b"payload"
    assert len(calls) == 1
