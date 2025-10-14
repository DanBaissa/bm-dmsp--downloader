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


class _DummyPaginator:
    def paginate(self, **kwargs):
        return []


class _DummyS3:
    def get_paginator(self, *_args, **_kwargs):
        return _DummyPaginator()

    def head_object(self, **_kwargs):
        return {"ContentLength": 0}

    def download_file(self, *_args, Callback=None, **_kwargs):
        if Callback is not None:
            Callback(0)


dummy_boto3.client = lambda *args, **kwargs: _DummyS3()
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

dummy_numpy = types.ModuleType("numpy")
dummy_numpy.float32 = float
dummy_numpy.nan = float("nan")
sys.modules.setdefault("numpy", dummy_numpy)

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
dummy_rasterio.warp.Resampling = type("Resampling", (), {})

dummy_rasterio.open = lambda *args, **kwargs: types.SimpleNamespace(
    profile={},
    close=lambda: None,
    read=lambda *a, **k: _DummyArray(),
    height=1,
    width=1,
    transform=None,
    crs=None,
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
sys.modules.setdefault("requests", dummy_requests)

import data_sampler

rasterio = data_sampler.rasterio


class DummyResponse:
    def __init__(self, status_code: int, payload: dict | None = None, raise_error: bool = False):
        self.status_code = status_code
        self._payload = payload or {}
        self._raise_error = raise_error
        self.headers = self._payload.get("headers", {})

    def raise_for_status(self):
        if self._raise_error or self.status_code >= 400:
            raise data_sampler.requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=8192):
        content = self._payload.get("content", b"")
        if isinstance(content, (bytes, bytearray)):
            yield content
        else:
            yield b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


dummy_requests.get = lambda *args, **kwargs: DummyResponse(200)


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

    monkeypatch.setattr(data_sampler.requests, "get", fake_get)

    links = data_sampler.search_nasa_cmr(
        "collection",
        "2014-08-28",
        [179.5, -10.0, 181.5, 10.0],
    )

    assert links == ["https://example.com/file1.h5", "https://example.com/file2.h5"]
    assert len(calls) == 2


def test_process_samples_parallel_continues_after_failure(monkeypatch, tmp_path):
    def fake_read_file(_):
        return object()

    monkeypatch.setattr(data_sampler.gpd, "read_file", fake_read_file)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def fake_worker(sample, *args, **kwargs):
        if sample["id"] == "fail":
            raise RuntimeError("boom")
        path = output_dir / f"{sample['id']}.tif"
        path.write_bytes(b"data")
        return (f"ok {sample['id']}", path)

    monkeypatch.setattr(data_sampler, "process_single_sample", fake_worker)

    samples = [
        {"id": "a", "Longitude": 1.0, "Latitude": 2.0, "date": "2014-01-01"},
        {"id": "fail", "Longitude": 3.0, "Latitude": 4.0, "date": "2014-01-02"},
        {"id": "b", "Longitude": 5.0, "Latitude": 6.0, "date": "2014-01-03"},
    ]
    results = data_sampler.process_samples_parallel(
        samples,
        patch_size_pix=10,
        collection_id="cid",
        token="token",
        tile_shapefile_path=Path("dummy.shp"),
        output_folder=tmp_path / "out",
        temp_folder=tmp_path / "tmp",
        max_workers=2,
    )

    assert [patch.tile_id for patch in results] == ["tile_001", "tile_002"]
    assert all(patch.path.name == f"{patch.tile_id}.tif" for patch in results)


def test_build_bm_mosaic_skips_missing_tiles(monkeypatch, tmp_path):
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

    def fake_merge(datasets, nodata=None):
        return FakeMosaic(), "transform"

    monkeypatch.setattr(data_sampler, "rio_merge", fake_merge)
    monkeypatch.setattr(data_sampler.rasterio, "open", fake_open)

    def fake_h5_to_geotiff(path, gdf):
        if path.stem == "missing":
            raise data_sampler.TileMetadataMissingError("no tile metadata")
        tif_path = tmp_path / f"{path.stem}.tif"
        tif_path.touch()
        return tif_path

    monkeypatch.setattr(data_sampler, "h5_to_geotiff", fake_h5_to_geotiff)

    mosaic, transform, profile = data_sampler.build_bm_mosaic_for_bbox(
        [Path("missing.h5"), Path("valid.h5")],
        tile_shapefile_gdf=object(),
    )

    assert mosaic.shape == (1, 1)
    assert profile["dtype"] == "float32"

    with pytest.raises(RuntimeError, match="No valid Black Marble tiles"):
        data_sampler.build_bm_mosaic_for_bbox(
            [Path("missing.h5")],
            tile_shapefile_gdf=object(),
        )


def test_resolve_cli_path_defaults_with_output_root(tmp_path):
    output_root = tmp_path / "outputs"
    default = data_sampler.DEFAULT_MANIFEST
    resolved = data_sampler.resolve_cli_path(output_root, default, default, default.name)
    assert resolved == output_root / default.name


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
