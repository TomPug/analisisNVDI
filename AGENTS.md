# AI Agent Instructions for analisisNVDI

These instructions help agents be productive in this repo. Keep changes focused on configuration and scripts, not on data outputs.

## What this project does
- Downloads Sentinel-2 L2A and Sentinel-1 data from Copernicus Data Space (CDSE) STAC.
- Computes spectral indices (NDVI/NDWI/NDBI/SAVI/EVI, Tasseled Cap) and exports time-series CSV/PNG and GeoTIFF stacks.
- Uses stackstac for temporal stacking and masking (SCL for clouds).

## Primary entry points
- Sentinel-2 pipeline: [s2_cdse_stackstac.py](s2_cdse_stackstac.py)
- Sentinel-1 pipeline: [s1_cdse_stackstac.py](s1_cdse_stackstac.py)
- Shared helpers (auth, AOI, I/O): [cdse_stackstac_common.py](cdse_stackstac_common.py)
- Index definitions: [s2_index_definitions.toml](s2_index_definitions.toml)
- Project overview and usage notes: [README.md](README.md)

## Run commands (from repo root)
- `python s2_cdse_stackstac.py`
- `python s1_cdse_stackstac.py`

## Configuration expectations
- A `.env` file is required for runtime configuration (paths, dates, credentials). See [README.md](README.md) for required variables.
- Index formulas live in [s2_index_definitions.toml](s2_index_definitions.toml). Add new indices there rather than in code.

## Common pitfalls
- CDSE auth is required: `CDSE_ACCESS_TOKEN` or `CDSE_USERNAME`/`CDSE_PASSWORD`.
- Cloud mask defaults to SCL classes 3, 8, 9, 10, 11; confirm if a different mask is required.
- Windows GDAL driver plugins may cause issues; follow the Windows troubleshooting steps in [README.md](README.md).

## When asked to download indices and bands
- Prefer configuring `.env` variables and running [s2_cdse_stackstac.py](s2_cdse_stackstac.py) instead of adding new code.
- If user requests a specific AOI path or band list, map it to existing env variables described in [README.md](README.md).
