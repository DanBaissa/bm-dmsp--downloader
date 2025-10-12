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

The refactored downloader reads the NASA token from the `NASA_TOKEN` environment variable. If the variable is missing, the script will exit with a helpful message. Ensure the token is available before running download commands.
