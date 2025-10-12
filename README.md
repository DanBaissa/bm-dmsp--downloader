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

The new workflow is driven by `data_sampler.py`, which keeps the LandScan population-balanced sampling step and then downloads paired Black Marble and DMSP GeoTIFFs.

```bash
python data_sampler.py --patch-size 1000 --samples-per-bin 200
```

Key behaviours:

* A population-balanced sample table is generated from the LandScan 2012 raster. The sampled locations (with their assigned acquisition dates) are written to `Raw_NL_Data/sample_locations.csv`.
* Black Marble patches are downloaded at the requested size (default `1000x1000` pixels) into `Raw_NL_Data/BM data/`.
* Matching DMSP scenes are reprojected onto each Black Marble patch footprint and written to `Raw_NL_Data/DMSP data/`.
* A manifest of matched file pairs is saved to `Raw_NL_Data/bm_dmsp_manifest.csv`.

Use `python data_sampler.py --help` to see all configuration options, including worker counts for the BM and DMSP steps.
