# bm-dmsp--downloader

This repository contains tooling for downloading and preparing Black Marble DMSP datasets.

## Prerequisites

1. Create a NASA Earthdata account and generate a [personal API token](https://urs.earthdata.nasa.gov/).
2. Copy the example environment file and add your token:

   ```bash
   cp .env.example .env
   echo "NASA_TOKEN=your_actual_token" >> .env
   ```

   You can also set `NASA_TOKEN` in your shell environment instead of using a `.env` file.

3. (Optional) Install [`python-dotenv`](https://pypi.org/project/python-dotenv/) if you want the downloader to automatically read from the `.env` file:

   ```bash
   pip install python-dotenv
   ```

## Running the downloader

The refactored pipeline lives in `data_sampler.py`. It keeps the LandScan-driven sampling so every run produces a population-balanced CSV of candidate locations **and** downloads the paired Black Marble (BM) and DMSP GeoTIFFs at a 1000×1000 pixel resolution.

```bash
python data_sampler.py --patch-size 1000 --samples-per-bin 200 --max-workers 4 --output-folder Raw_NL_Data --sampling-seed 13492
```

Use `--output-folder` to keep every generated artifact—Black Marble rasters, DMSP rasters, plots, CSVs, and manifests—under a
single directory. Any other relative output paths you supply (for example, a custom `--locations-csv` or `--manifest`) are
resolved beneath that root.

### What the script does

1. Downsamples LandScan 2012, masks Antarctica / < -60° latitudes, and samples `--samples-per-bin` points per integer log-population bin.
2. Writes the sampled table (including the assigned acquisition dates) to `sampled_locations.csv` (or the path you choose) so you can inspect or reuse it.
3. Uses the CSV to download BM tiles via NASA CMR, extracts 1000×1000 GeoTIFF patches, and saves them under `Raw_NL_Data/BM data/` by default (or inside your chosen output folder).
4. Finds matching DMSP scenes on the public S3 bucket, reprojects them onto each BM grid (so the extents align exactly), and stores the best-correlated patch in `Raw_NL_Data/DMSP data/` (again, relative to your output folder when provided).
5. Emits `Raw_NL_Data/bm_dmsp_pairs.csv`, a manifest describing every BM/DMSP pair, including the DMSP satellite (`F-number`) and the correlation score used to pick the best match (again defaulting to that path unless you relocate it with `--output-folder`).

If you already have a CSV of locations (with `Longitude`, `Latitude`, and `date` columns), skip the sampling step and feed it to the downloader:

```bash
python data_sampler.py --skip-sampling --locations-csv my_locations.csv
```

To focus sampling on one or two countries instead of the entire world, supply the `--countries` flag:

```bash
python data_sampler.py --countries "United States" Canada
```

The script will infer the appropriate attribute column from `Data/World_Countries/World_Countries_Generalized.shp`, but you can override it with `--country-column` if needed.

Set `--sampling-seed` to reproduce the LandScan sample selection and `--date-seed` to control how DMSP acquisition dates are drawn. Both defaults mirror the values used in the original pipeline, so your historical runs stay consistent unless you override them.

All command-line options can be viewed with `python data_sampler.py --help`.

## Testing

The regression checks live in `tests/test_data_sampler.py`. The module stubs out heavyweight dependencies (e.g., `rasterio`, `geopandas`, `boto3`) so the suite can exercise the downloader’s control flow—dateline-aware CMR queries, worker failure handling, and missing tile metadata—without needing the full geospatial stack. Run the tests with:

```bash
pytest
```

Keeping this file up to date ensures future changes preserve the downloader’s resilience characteristics, even in lightweight CI environments.
