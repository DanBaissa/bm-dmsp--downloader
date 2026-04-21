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
python data_sampler.py --patch-size 1000 --samples-per-bin 2 --max-workers 4 --output-folder Raw_NL_Data --sampling-seed 13492
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

## Running on AWS EC2

For a first pass, use EC2 rather than Lambda. This pipeline depends on native geospatial libraries (`rasterio`, `geopandas`, `GDAL`) and works more predictably on a normal Linux VM.

### 1. Launch an Ubuntu EC2 instance

An instance in the `t3.large` or `m6i.large` range is a reasonable starting point for a smoke test. For a large run, you will likely want a larger EBS volume than the default because the script writes:

- sampled CSVs
- Black Marble GeoTIFF patches
- DMSP GeoTIFF patches
- manifest files
- temporary download artifacts

### 2. Copy the repo, the `Data/` folder, and your `.env`

From your local machine, copy the project to the instance. The important part is that the EC2 working tree keeps the same relative layout, because the script expects:

- `Data/Global_2012/landscan-global-2012.tif`
- `Data/Black_Marble_IDs/Black_Marble_World_tiles.shp`
- `Data/World_Countries/World_Countries_Generalized.shp`
- `.env` containing `NASA_TOKEN=...`

Example with `scp`:

```bash
scp -i /path/to/key.pem -r bm-dmsp--downloader ubuntu@YOUR_EC2_PUBLIC_DNS:~/
```

### 3. Bootstrap the instance

Once connected to EC2:

```bash
cd ~/bm-dmsp--downloader
bash aws/bootstrap_ubuntu.sh
source .venv/bin/activate
```

### 4. Run a small validation download

This repo now includes a test runner that uses your existing `.env` and sets `samples per bin = 2`:

```bash
bash aws/run_test_download.sh
```

That writes outputs under `ec2_test_run/`.

If you want to kick off the EC2 test from your Windows machine and then pull the results back automatically, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\aws\run_ec2_test_and_pull.ps1 -HostName YOUR_EC2_PUBLIC_DNS -KeyPath C:\path\to\key.pem
```

That script will:

- upload `data_sampler.py`, `requirements.txt`, `.env`, `aws/`, and `Data/`
- bootstrap the Ubuntu instance
- run the test download with `--samples-per-bin 2 --patch-size 1000 --max-workers 2 --output-folder ec2_test_run`
- copy `ec2_test_run/` back to your local repo root

### 5. Run the larger job after the test passes

```bash
bash aws/run_full_download.sh
```

That writes outputs under `ec2_full_run/`.

The full-run helper is configured for `--samples-per-bin 2000`.

From Windows, you can launch that job on EC2 and pull the results back with:

```powershell
powershell -ExecutionPolicy Bypass -File .\aws\run_ec2_full_and_pull.ps1 -HostName YOUR_EC2_PUBLIC_DNS -KeyPath C:\path\to\key.pem
```

If you want a different output directory, worker count, or seeds, edit the shell script or run `python data_sampler.py` directly.

### Running one fixed sample across many EC2 instances

For a large run, the reliable pattern is:

1. Generate one fixed `sampled_locations.csv`.
2. Split it into disjoint shard CSVs.
3. Launch one EC2 instance per shard with a large EBS-backed root volume.
4. Warm each shard on that same instance and cache root.
5. Run the full shard on that same instance and merge the outputs afterward.

This repo now includes an AWS CLI-based launcher for that flow:

```powershell
powershell -ExecutionPolicy Bypass -File .\aws\launch_fixed_sample_fleet.ps1 `
  -ShardCount 10 `
  -SamplesPerBin 2000 `
  -WarmupRowsPerShard 25 `
  -InstanceType m6i.2xlarge `
  -VolumeSizeGiB 2000 `
  -Profile coauthor-project `
  -KeyName YOUR_EC2_KEY_NAME `
  -KeyPath C:\path\to\key.pem `
  -SubnetId subnet-... `
  -SecurityGroupIds sg-...
```

What it does:

- runs `python data_sampler.py --sample-only ...` locally to create one fixed sample CSV
- splits that CSV into `shard_01.csv` ... `shard_10.csv` plus matching warm-up subset CSVs
- launches one EC2 instance per shard via `aws ec2 run-instances`
- provisions a large EBS-backed root volume on each instance
- uploads the repo, `Data/`, `.env`, and the shard CSVs
- bootstraps Ubuntu dependencies
- runs `aws/run_fixed_shard_download.sh` remotely so each instance:
  - warms its cache with the small shard subset
  - runs the full shard against the same `--cache-root`

Each instance keeps its cache under `/data/bm_dmsp_cache/<shard_id>` and outputs under `/data/bm_dmsp_runs/<shard_id>/`.

To pull the results back and merge them:

```powershell
powershell -ExecutionPolicy Bypass -File .\aws\pull_and_merge_fixed_sample_fleet.ps1 `
  -InstancesCsv .\ec2_fixed_sample_fleet_YYYYMMDD_HHMMSS\instances.csv `
  -KeyPath C:\path\to\key.pem
```

That downloads each shard's `full/` output and writes a merged dataset locally.

Storage guidance:

- `--samples-per-bin 2000` is a large run. Plan for multi-terabyte EBS, not the default EC2 disk size.
- The launcher defaults to a `2000 GiB` `gp3` root volume per instance. Increase it if you expect broad date/tile coverage or want to retain warm caches for reruns.
- The warm cache only helps when the same instance and same `--cache-root` are reused for later work.

## Testing

The regression checks live in `tests/test_data_sampler.py`. The module stubs out heavyweight dependencies (e.g., `rasterio`, `geopandas`, `boto3`) so the suite can exercise the downloader’s control flow—dateline-aware CMR queries, worker failure handling, and missing tile metadata—without needing the full geospatial stack. Run the tests with:

```bash
pytest
```

Keeping this file up to date ensures future changes preserve the downloader’s resilience characteristics, even in lightweight CI environments.
