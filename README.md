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

- `s2_cdse_stackstac.py`: Sentinel-2, compuestos temporales por intervalo y export GeoTIFF multibanda.
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
