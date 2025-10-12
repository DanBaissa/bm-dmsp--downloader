"""Compatibility entry point for the reorganised BM/DMSP downloader.

This script simply proxies to :mod:`bm_dmsp_downloader` so existing workflows
that executed ``data_sampler.py`` keep working. See the module docstring in
``bm_dmsp_downloader.py`` or run ``python bm_dmsp_downloader.py --help`` for
usage instructions.
"""

from bm_dmsp_downloader import main


if __name__ == "__main__":
    main()
