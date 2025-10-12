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
python data_sampler.py \
  --patch-size 1000 \
  --samples-per-bin 200 \
  --max-workers 4
```

### What the script does

1. Downsamples LandScan 2012, masks Antarctica / < -60° latitudes, and samples `--samples-per-bin` points per integer log-population bin.
2. Writes the sampled table (including the assigned acquisition dates) to `sampled_locations.csv` so you can inspect or reuse it.
3. Uses the CSV to download BM tiles via NASA CMR, extracts 1000×1000 GeoTIFF patches, and saves them under `Raw_NL_Data/BM data/`.
4. Finds matching DMSP scenes on the public S3 bucket, reprojects them onto each BM grid, and stores the best-quality patches in `Raw_NL_Data/DMSP data/`.
5. Emits `Raw_NL_Data/bm_dmsp_pairs.csv`, a manifest describing every BM/DMSP pair that was successfully created.

If you already have a CSV of locations (with `Longitude`, `Latitude`, and `date` columns), skip the sampling step and feed it to the downloader:

```bash
python data_sampler.py --skip-sampling --locations-csv my_locations.csv
```

All command-line options can be viewed with `python data_sampler.py --help`.
