# analisisNVDI

Script para descargar y calcular una serie temporal de indices espectrales (Sentinel-2 L2A) desde el STAC de Copernicus Data Space para un AOI definido en `Shapefile` o `GeoPackage`.

## Requisitos

```bash
pip install pystac-client requests rasterio matplotlib numpy geopandas fiona
```

> Si ya tienes `geopandas` funcionando, normalmente `fiona` ya viene instalado.

## Configuracion

1. Edita el archivo `.env`.
2. Define al menos:
- `AOI_VECTOR_PATH` (ruta a `.shp` o `.gpkg`)
- `START_DATE` y `END_DATE`
- `INDEX_NAME` (`NDVI`, `NDWI`, `NDBI`, `SAVI`, `EVI`)
- Credenciales CDSE (`CDSE_ACCESS_TOKEN` o `CDSE_USERNAME`/`CDSE_PASSWORD`)
- `EXPORT_INDEX_TIFF=true` para guardar GeoTIFF por escena

## Ejecucion

```bash
python NDVI_Copernicus.py
```

## Salidas

En `OUTPUT_DIR` se generan:
- `*_timeseries.csv` con una fila por escena (fecha, nubosidad, estadisticos del indice en AOI)
- `*_timeseries.png` con la serie temporal del valor medio del indice
- GeoTIFF por escena en `INDEX_TIFF_DIR` (o `OUTPUT_DIR/tiffs` si no se define)

## Troubleshooting (Windows)

Si aparece `UnicodeDecodeError` relacionado con `rasterio._env.log_error`:

```powershell
$env:PYTHONUTF8="1"
python NDVI_Copernicus.py
```

Si persiste, actualiza `rasterio/GDAL` en tu entorno virtual.

## Scripts Nuevos (Copernicus STAC + stackstac)

Se añadieron dos scripts nuevos, equivalentes a tu flujo S1/S2 pero sobre STAC de Copernicus y usando `stackstac`:

- `s2_cdse_stackstac.py`: Sentinel-2, calculo de indice (NDVI/NDWI/NDBI/SAVI/EVI), mascara de nubes por pixel (SCL), stack temporal y compuestos por intervalo.
- `s1_cdse_stackstac.py`: Sentinel-1, filtros de órbita/modo, opción dB y export GeoTIFF multibanda.

Base compartida:
- `cdse_stackstac_common.py`

Plantilla de variables:
- `.env.stackstac.example`

Ejecución:

```bash
python s2_cdse_stackstac.py
python s1_cdse_stackstac.py
```

Variables nuevas relevantes para S2:

- `S2_INDEX_NAME=NDVI` (nombre del indice en `s2_index_definitions.toml`, o `NONE`)
- `S2_INDEX_DEFINITIONS_FILE=s2_index_definitions.toml`
- `S2_APPLY_CLOUD_MASK=true`
- `S2_CLOUD_MASK_ASSET=SCL`
- `S2_CLOUD_MASK_SCL_CLASSES=3,8,9,10,11`
- `S2_MAX_CLOUD_COVER=None` (recomendado si ya usas mascara SCL por pixel)
- `S2_EXPORT_TEMPORAL_STACK=true`
- `S2_EXPORT_COMPOSITE_STACK=true`
- `S2_DELETE_INTERVAL_COMPOSITES_AFTER_STACK=true` (borra TIFF intermedios de intervalo y deja los 2 stacks)
- `S2_LOCAL_ASSET_CACHE=false` (modo S3 directo recomendado)
- `S2_CLEANUP_INTERMEDIATE_FILES=true`

Definiciones de indices en archivo externo (`s2_index_definitions.toml`):

```toml
[indices.NDRE]
required_assets = ["B08", "B05"]
formula = "nd(B08, B05)"
clip_range = [-1.0, 1.0]
```

Variables comunes para modo S3 directo (S1/S2):

- `AWS_ACCESS_KEY_ID=...`
- `AWS_SECRET_ACCESS_KEY=...`
- `STACKSTAC_DASK_WORKERS=4`
- `STACKSTAC_LOCAL_ASSET_CACHE=false`
