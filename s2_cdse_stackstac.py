"""Pipeline Sentinel-2 sobre Copernicus STAC usando stackstac.

Este script:
1. Lee configuracion desde entorno/.env.
2. Busca escenas S2 para AOI y rango de fechas.
3. Aplica filtro de nubosidad a nivel item.
4. Carga bandas con stackstac sobre una grilla comun.
5. Opcionalmente enmascara por AOI.
6. Compone intervalos temporales y exporta un GeoTIFF multibanda por intervalo.
7. Genera un CSV manifiesto con metadatos de salida.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import numpy as np
from pystac_client import Client
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
import stackstac

from cdse_stackstac_common import (
    AOIInfo,
    CDSE_CLIENT_ID_DEFAULT,
    CDSE_OIDC_TOKEN_URL_DEFAULT,
    STAC_API_URL_DEFAULT,
    apply_aoi_mask,
    build_intervals,
    build_stackstac_gdal_env,
    cache_assets_locally,
    get_cdse_access_token,
    get_cloud_cover,
    get_item_datetime,
    guess_epsg,
    load_aoi_info,
    load_env_file,
    normalize_item_https,
    parse_bool_env,
    parse_csv_env,
    parse_int_env,
    parse_optional_float_env,
    parse_optional_int_env,
    reduce_time_window,
    resolve_requested_assets,
    search_items_with_retry,
    select_time_window,
    validate_date,
    write_csv,
    write_multiband_geotiff,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"


@dataclass(frozen=True)
class S2Config:
    """Configuracion completa de ejecucion para el flujo Sentinel-2."""

    stac_api_url: str
    collection: str
    start_date: str
    end_date: str
    max_items: int
    max_cloud_cover: float | None
    requested_assets: list[str]
    interval_days: int
    composite_method: str
    output_epsg: int | None
    output_resolution: float
    chunksize: int
    stackstac_rescale: bool
    use_local_asset_cache: bool
    asset_cache_dir: Path
    asset_cache_force_refresh: bool
    output_dir: Path
    output_prefix: str
    tiff_compress: str
    apply_aoi_mask: bool
    aoi_vector_path: Path
    aoi_layer: str | None
    cdse_access_token: str | None
    cdse_username: str | None
    cdse_password: str | None
    ask_credentials_in_terminal: bool
    cdse_oidc_token_url: str
    cdse_client_id: str


def _to_abs_path(path_text: str) -> Path:
    """Convierte una ruta en absoluta respecto al directorio del script."""
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def build_config() -> S2Config:
    """Construye la configuracion S2 leyendo variables de entorno.

    Returns:
        Configuracion inmutable ``S2Config``.

    Raises:
        ValueError: Si faltan variables obligatorias o fechas invalidas.
    """
    load_env_file(DEFAULT_ENV_PATH)

    start_date = validate_date(os.getenv("START_DATE", "2025-01-01").strip())
    end_date = validate_date(os.getenv("END_DATE", "2025-12-31").strip())
    if start_date > end_date:
        raise ValueError("START_DATE no puede ser mayor que END_DATE.")

    aoi_path_raw = os.getenv("AOI_VECTOR_PATH", "").strip()
    if not aoi_path_raw:
        raise ValueError("AOI_VECTOR_PATH es obligatorio.")
    aoi_path = _to_abs_path(aoi_path_raw)

    output_dir = _to_abs_path(os.getenv("S2_OUTPUT_DIR", "outputs/s2_stackstac").strip())

    return S2Config(
        stac_api_url=os.getenv("STAC_API_URL", STAC_API_URL_DEFAULT).strip(),
        collection=os.getenv("S2_COLLECTION", "sentinel-2-l2a").strip(),
        start_date=start_date,
        end_date=end_date,
        max_items=parse_int_env("MAX_ITEMS", default=500, min_value=1),
        max_cloud_cover=parse_optional_float_env("S2_MAX_CLOUD_COVER", default=20.0),
        requested_assets=parse_csv_env("S2_BANDS", "B02,B03,B04,B08"),
        interval_days=parse_int_env("S2_INTERVAL_DAYS", default=5, min_value=1),
        composite_method=os.getenv("S2_COMPOSITE_METHOD", "median").strip().lower(),
        output_epsg=parse_optional_int_env("S2_OUTPUT_EPSG"),
        output_resolution=float(os.getenv("S2_OUTPUT_RESOLUTION_M", "20").strip()),
        chunksize=parse_int_env("S2_CHUNKSIZE", default=1024, min_value=128),
        stackstac_rescale=parse_bool_env("S2_STACKSTAC_RESCALE", False),
        use_local_asset_cache=parse_bool_env("STACKSTAC_LOCAL_ASSET_CACHE", True),
        asset_cache_dir=_to_abs_path(
            os.getenv("STACKSTAC_ASSET_CACHE_DIR", "outputs/asset_cache").strip()
        ),
        asset_cache_force_refresh=parse_bool_env(
            "STACKSTAC_ASSET_CACHE_FORCE_REFRESH", False
        ),
        output_dir=output_dir,
        output_prefix=(os.getenv("S2_OUTPUT_PREFIX", "s2").strip() or "s2"),
        tiff_compress=(os.getenv("S2_TIFF_COMPRESS", "DEFLATE").strip() or "DEFLATE").upper(),
        apply_aoi_mask=parse_bool_env("S2_APPLY_AOI_MASK", True),
        aoi_vector_path=aoi_path,
        aoi_layer=(os.getenv("AOI_LAYER", "").strip() or None),
        cdse_access_token=(os.getenv("CDSE_ACCESS_TOKEN", "").strip() or None),
        cdse_username=(os.getenv("CDSE_USERNAME", "").strip() or None),
        cdse_password=(os.getenv("CDSE_PASSWORD", "").strip() or None),
        ask_credentials_in_terminal=parse_bool_env("ASK_CREDENTIALS_IN_TERMINAL", True),
        cdse_oidc_token_url=os.getenv("CDSE_OIDC_TOKEN_URL", CDSE_OIDC_TOKEN_URL_DEFAULT).strip(),
        cdse_client_id=os.getenv("CDSE_CLIENT_ID", CDSE_CLIENT_ID_DEFAULT).strip(),
    )


def _dt_to_utc_naive(value: datetime | None) -> datetime | None:
    """Convierte datetime timezone-aware a UTC naive para comparaciones."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _count_items_in_window(item_datetimes: list[datetime], start: datetime, end: datetime) -> int:
    """Cuenta datetimes incluidos en el intervalo semiabierto [start, end)."""
    return sum(1 for dt in item_datetimes if start <= dt < end)


def _sanitize_filename(text: str) -> str:
    """Normaliza texto para uso en nombres de archivo."""
    allowed = {"-", "_"}
    chars = [char if (char.isalnum() or char in allowed) else "_" for char in text]
    return "".join(chars).strip("_") or "output"


def _build_output_tiff_path(
    output_dir: Path,
    prefix: str,
    start: datetime,
    end: datetime,
) -> Path:
    """Construye ruta de salida para el GeoTIFF de un intervalo temporal."""
    end_dt = end - timedelta(days=1)
    return output_dir / (
        f"{_sanitize_filename(prefix)}_"
        f"{start.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}.tif"
    )


def main() -> None:
    """Punto de entrada del flujo Sentinel-2.

    Flujo resumido:
    1. Cargar config, token y AOI.
    2. Buscar escenas STAC y filtrar por nubosidad.
    3. Resolver assets de bandas y preparar stack stackstac.
    4. Componer por intervalos temporales.
    5. Exportar GeoTIFF y CSV manifiesto.
    """
    config = build_config()

    if config.max_cloud_cover is not None and not (0 <= config.max_cloud_cover <= 100):
        raise ValueError("S2_MAX_CLOUD_COVER debe estar entre 0 y 100 o ser None.")
    if config.output_resolution <= 0:
        raise ValueError("S2_OUTPUT_RESOLUTION_M debe ser > 0.")

    access_token = get_cdse_access_token(
        access_token=config.cdse_access_token,
        username=config.cdse_username,
        password=config.cdse_password,
        ask_credentials_in_terminal=config.ask_credentials_in_terminal,
        oidc_token_url=config.cdse_oidc_token_url,
        client_id=config.cdse_client_id,
    )
    aoi: AOIInfo = load_aoi_info(config.aoi_vector_path, config.aoi_layer)

    print(f"AOI: {aoi.vector_path}")
    if aoi.layer:
        print(f"Capa AOI: {aoi.layer}")
    print(f"BBox lat/lon: {aoi.bbox_latlon}")
    print(f"Coleccion S2: {config.collection}")
    print(f"Bandas pedidas: {config.requested_assets}")
    print(f"stackstac rescale: {config.stackstac_rescale}")

    catalog = Client.open(config.stac_api_url)
    items = search_items_with_retry(
        catalog=catalog,
        collections=[config.collection],
        start_date=config.start_date,
        end_date=config.end_date,
        bbox_latlon=aoi.bbox_latlon,
        max_items=config.max_items,
    )
    print(f"Escenas encontradas (sin filtrar): {len(items)}")
    if not items:
        raise RuntimeError("No se encontraron escenas S2 en el rango solicitado.")

    if config.max_cloud_cover is not None:
        items = [item for item in items if get_cloud_cover(item) <= config.max_cloud_cover]
        print(f"Escenas con nubosidad <= {config.max_cloud_cover}%: {len(items)}")
    if not items:
        raise RuntimeError("No hay escenas S2 tras aplicar el filtro de nubosidad.")

    items = sorted(items, key=lambda it: (_dt_to_utc_naive(get_item_datetime(it)) or datetime.max, it.id))
    item_datetimes = [
        dt
        for dt in (_dt_to_utc_naive(get_item_datetime(item)) for item in items)
        if dt is not None
    ]

    prepared_items = [normalize_item_https(item) for item in items]
    requested_assets = [asset.upper() for asset in config.requested_assets]
    asset_mapping = resolve_requested_assets(prepared_items, requested_assets)
    asset_keys = [asset_mapping[name] for name in requested_assets]
    print(f"Assets resueltos: {asset_mapping}")

    if config.use_local_asset_cache:
        cache_dir = config.asset_cache_dir / "s2"
        print(f"Cache local assets habilitado: {cache_dir}")
        prepared_items = cache_assets_locally(
            items=prepared_items,
            asset_keys=asset_keys,
            access_token=access_token,
            cache_dir=cache_dir,
            force_refresh=config.asset_cache_force_refresh,
        )

    output_epsg = config.output_epsg or guess_epsg(prepared_items)
    if output_epsg is None:
        raise RuntimeError(
            "No se pudo inferir EPSG desde STAC. Define S2_OUTPUT_EPSG en .env."
        )
    print(f"EPSG salida: {output_epsg}")

    gdal_env = build_stackstac_gdal_env(access_token)
    stack_dtype = np.dtype("float64" if config.stackstac_rescale else "float32")
    stack_fill_value = stack_dtype.type(np.nan)
    if not np.can_cast(type(stack_fill_value), stack_dtype):
        stack_dtype = np.dtype("float64")
        stack_fill_value = np.float64(np.nan)
    stack = stackstac.stack(
        prepared_items,
        assets=asset_keys,
        epsg=output_epsg,
        resolution=config.output_resolution,
        bounds_latlon=aoi.bbox_latlon,
        snap_bounds=True,
        chunksize=config.chunksize,
        dtype=stack_dtype,
        fill_value=stack_fill_value,
        rescale=config.stackstac_rescale,
        sortby_date="asc",
        xy_coords="center",
        resampling=Resampling.bilinear,
        properties=False,
        band_coords=False,
        gdal_env=gdal_env,
        errors_as_nodata=(
            RasterioIOError("HTTP response code: 404"),
            RasterioIOError(r"HTTP response code: (429|5\\d\\d)"),
            RasterioIOError(r"Range downloading not supported by this server"),
        ),
    ).assign_coords(band=("band", requested_assets))

    if config.apply_aoi_mask:
        stack = apply_aoi_mask(stack, aoi, output_epsg)

    intervals = build_intervals(config.start_date, config.end_date, config.interval_days)
    print(f"Intervalos temporales: {len(intervals)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for idx, (start, end) in enumerate(intervals, start=1):
        window = select_time_window(stack, start, end)
        if window is None:
            continue

        composite = reduce_time_window(window, config.composite_method)
        output_tiff = _build_output_tiff_path(
            output_dir=config.output_dir,
            prefix=config.output_prefix,
            start=start,
            end=end,
        )
        write_multiband_geotiff(
            output_tiff=output_tiff,
            data=composite,
            output_epsg=output_epsg,
            compress=config.tiff_compress,
        )

        n_scenes = int(window.sizes.get("time", 0))
        n_items_by_date = _count_items_in_window(item_datetimes, start, end)
        end_inclusive = end - timedelta(days=1)
        print(
            f"[{idx}/{len(intervals)}] {start.date()} -> {end_inclusive.date()} | "
            f"escenas stack={n_scenes} | items={n_items_by_date} | {output_tiff.name}"
        )

        manifest_rows.append(
            {
                "interval_start": start.strftime("%Y-%m-%d"),
                "interval_end_exclusive": end.strftime("%Y-%m-%d"),
                "stack_scene_count": n_scenes,
                "item_count": n_items_by_date,
                "output_tiff": str(output_tiff.resolve()),
                "bands": ",".join(requested_assets),
                "method": config.composite_method,
                "epsg": output_epsg,
                "resolution_m": config.output_resolution,
            }
        )

    manifest_csv = config.output_dir / f"{_sanitize_filename(config.output_prefix)}_manifest.csv"
    write_csv(manifest_rows, manifest_csv)

    print("\nProceso S2 completado.")
    print(f"- GeoTIFF generados: {len(manifest_rows)}")
    print(f"- Carpeta salida: {config.output_dir.resolve()}")
    print(f"- Manifest CSV: {manifest_csv.resolve()}")


if __name__ == "__main__":
    main()
