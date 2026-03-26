"""Pipeline Sentinel-2 sobre Copernicus STAC usando stackstac.

Este script:
1. Lee configuracion desde entorno/.env.
2. Busca escenas S2 para AOI y rango de fechas.
3. Aplica filtro de nubosidad a nivel item.
4. Carga bandas requeridas para calculo de indices.
5. Aplica mascara de nubes por pixel sobre cada escena (SCL).
6. Exporta stack temporal del indice/bandas.
7. Opcionalmente compone intervalos temporales y exporta GeoTIFF.
8. Genera un CSV manifiesto con metadatos de salida.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from typing import Callable

import numpy as np
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
import stackstac
import xarray as xr

from cdse_stackstac_common import (
    AOIInfo,
    CDSE_CLIENT_ID_DEFAULT,
    CDSE_OIDC_TOKEN_URL_DEFAULT,
    CDSE_S3_ENDPOINT,
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
    normalize_item_s3,
    parse_bool_env,
    parse_csv_env,
    parse_int_env,
    parse_optional_float_env,
    parse_optional_int_env,
    reduce_time_window,
    remove_directory,
    resolve_requested_assets,
    search_items_with_retry,
    select_time_window,
    validate_date,
    write_csv,
    write_multiband_geotiff,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"
DEFAULT_SCL_MASK_VALUES = (3, 8, 9, 10, 11)


@dataclass(frozen=True)
class IndexDefinition:
    """Definicion de un indice espectral para Sentinel-2."""

    name: str
    required_assets: tuple[str, ...]
    compute: Callable[[dict[str, xr.DataArray]], xr.DataArray]
    clip_range: tuple[float, float] | None = None


def _normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    denom = a + b
    return xr.where(denom != 0, (a - b) / denom, np.nan).astype("float32")


def _compute_ndvi(bands: dict[str, xr.DataArray]) -> xr.DataArray:
    return _normalized_difference(bands["B08"], bands["B04"])


def _compute_ndwi(bands: dict[str, xr.DataArray]) -> xr.DataArray:
    return _normalized_difference(bands["B03"], bands["B08"])


def _compute_ndbi(bands: dict[str, xr.DataArray]) -> xr.DataArray:
    return _normalized_difference(bands["B11"], bands["B08"])


def _compute_savi(bands: dict[str, xr.DataArray]) -> xr.DataArray:
    nir = bands["B08"]
    red = bands["B04"]
    l_factor = 0.5
    denom = nir + red + l_factor
    return xr.where(
        denom != 0,
        ((nir - red) / denom) * (1 + l_factor),
        np.nan,
    ).astype("float32")


def _compute_evi(bands: dict[str, xr.DataArray]) -> xr.DataArray:
    nir = bands["B08"]
    red = bands["B04"]
    blue = bands["B02"]
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    return xr.where(
        denom != 0,
        2.5 * (nir - red) / denom,
        np.nan,
    ).astype("float32")


INDEX_DEFINITIONS: dict[str, IndexDefinition] = {
    "NDVI": IndexDefinition(
        name="NDVI",
        required_assets=("B08", "B04"),
        compute=_compute_ndvi,
        clip_range=(-1.0, 1.0),
    ),
    "NDWI": IndexDefinition(
        name="NDWI",
        required_assets=("B03", "B08"),
        compute=_compute_ndwi,
        clip_range=(-1.0, 1.0),
    ),
    "NDBI": IndexDefinition(
        name="NDBI",
        required_assets=("B11", "B08"),
        compute=_compute_ndbi,
        clip_range=(-1.0, 1.0),
    ),
    "SAVI": IndexDefinition(
        name="SAVI",
        required_assets=("B08", "B04"),
        compute=_compute_savi,
        clip_range=(-1.0, 1.0),
    ),
    "EVI": IndexDefinition(
        name="EVI",
        required_assets=("B08", "B04", "B02"),
        compute=_compute_evi,
        clip_range=None,
    ),
}


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
    index_name: str
    apply_cloud_mask: bool
    cloud_mask_asset: str
    cloud_mask_scl_values: tuple[int, ...]
    interval_days: int
    composite_method: str
    export_interval_composites: bool
    export_temporal_stack: bool
    output_epsg: int | None
    output_resolution: float
    chunksize: int
    stackstac_rescale: bool
    use_local_asset_cache: bool
    asset_cache_dir: Path
    asset_cache_force_refresh: bool
    cleanup_intermediate_files: bool
    output_dir: Path
    output_prefix: str
    tiff_compress: str
    apply_aoi_mask: bool
    aoi_vector_path: Path
    aoi_layer: str | None
    cdse_access_token: str | None
    cdse_username: str | None
    cdse_password: str | None
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    ask_credentials_in_terminal: bool
    cdse_oidc_token_url: str
    cdse_client_id: str


def _to_abs_path(path_text: str) -> Path:
    """Convierte una ruta en absoluta respecto al directorio del script."""
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def _parse_int_csv_env(name: str, default_csv: str) -> tuple[int, ...]:
    values = parse_csv_env(name, default_csv)
    parsed: list[int] = []
    for value in values:
        parsed.append(int(value))
    unique = tuple(sorted(set(parsed)))
    if not unique:
        raise ValueError(f"Variable {name} no puede quedar vacia.")
    return unique


def _normalize_index_name(value: str) -> str:
    text = value.strip().upper()
    if not text:
        return "NDVI"
    if text == "NONE":
        return "NONE"
    if text not in INDEX_DEFINITIONS:
        valid = ", ".join(sorted(INDEX_DEFINITIONS.keys()))
        raise ValueError(f"S2_INDEX_NAME invalido: {text}. Valores validos: {valid}, NONE.")
    return text


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
    requested_assets = [asset.upper() for asset in parse_csv_env("S2_BANDS", "B02,B03,B04,B08")]
    index_name = _normalize_index_name(os.getenv("S2_INDEX_NAME", "NDVI"))
    use_local_asset_cache = parse_bool_env(
        "S2_LOCAL_ASSET_CACHE",
        parse_bool_env("STACKSTAC_LOCAL_ASSET_CACHE", False),
    )

    return S2Config(
        stac_api_url=os.getenv("STAC_API_URL", STAC_API_URL_DEFAULT).strip(),
        collection=os.getenv("S2_COLLECTION", "sentinel-2-l2a").strip(),
        start_date=start_date,
        end_date=end_date,
        max_items=parse_int_env("MAX_ITEMS", default=500, min_value=1),
        max_cloud_cover=parse_optional_float_env("S2_MAX_CLOUD_COVER", default=20.0),
        requested_assets=requested_assets,
        index_name=index_name,
        apply_cloud_mask=parse_bool_env("S2_APPLY_CLOUD_MASK", True),
        cloud_mask_asset=(os.getenv("S2_CLOUD_MASK_ASSET", "SCL").strip() or "SCL").upper(),
        cloud_mask_scl_values=_parse_int_csv_env(
            "S2_CLOUD_MASK_SCL_CLASSES",
            ",".join(str(v) for v in DEFAULT_SCL_MASK_VALUES),
        ),
        interval_days=parse_int_env("S2_INTERVAL_DAYS", default=5, min_value=1),
        composite_method=os.getenv("S2_COMPOSITE_METHOD", "median").strip().lower(),
        export_interval_composites=parse_bool_env("S2_EXPORT_INTERVAL_COMPOSITES", True),
        export_temporal_stack=parse_bool_env("S2_EXPORT_TEMPORAL_STACK", True),
        output_epsg=parse_optional_int_env("S2_OUTPUT_EPSG"),
        output_resolution=float(os.getenv("S2_OUTPUT_RESOLUTION_M", "20").strip()),
        chunksize=parse_int_env("S2_CHUNKSIZE", default=1024, min_value=128),
        stackstac_rescale=parse_bool_env("S2_STACKSTAC_RESCALE", False),
        use_local_asset_cache=use_local_asset_cache,
        asset_cache_dir=_to_abs_path(
            os.getenv("STACKSTAC_ASSET_CACHE_DIR", "outputs/asset_cache").strip()
        ),
        asset_cache_force_refresh=parse_bool_env(
            "STACKSTAC_ASSET_CACHE_FORCE_REFRESH", False
        ),
        cleanup_intermediate_files=parse_bool_env("S2_CLEANUP_INTERMEDIATE_FILES", True),
        output_dir=output_dir,
        output_prefix=(os.getenv("S2_OUTPUT_PREFIX", "s2").strip() or "s2"),
        tiff_compress=(os.getenv("S2_TIFF_COMPRESS", "DEFLATE").strip() or "DEFLATE").upper(),
        apply_aoi_mask=parse_bool_env("S2_APPLY_AOI_MASK", True),
        aoi_vector_path=aoi_path,
        aoi_layer=(os.getenv("AOI_LAYER", "").strip() or None),
        cdse_access_token=(os.getenv("CDSE_ACCESS_TOKEN", "").strip() or None),
        cdse_username=(os.getenv("CDSE_USERNAME", "").strip() or None),
        cdse_password=(os.getenv("CDSE_PASSWORD", "").strip() or None),
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID", "").strip() or None),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY", "").strip() or None),
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
    suffix: str = "",
) -> Path:
    """Construye ruta de salida para el GeoTIFF de un intervalo temporal."""
    end_dt = end - timedelta(days=1)
    suffix_token = f"_{_sanitize_filename(suffix)}" if suffix else ""
    return output_dir / (
        f"{_sanitize_filename(prefix)}_"
        f"{start.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}{suffix_token}.tif"
    )


def _build_temporal_stack_path(output_dir: Path, output_prefix: str, token: str) -> Path:
    safe_prefix = _sanitize_filename(output_prefix)
    safe_token = _sanitize_filename(token.lower())
    return output_dir / f"{safe_prefix}_{safe_token}_temporal_stack.tif"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_processing_assets(config: S2Config) -> list[str]:
    if config.index_name == "NONE":
        assets = list(config.requested_assets)
    else:
        assets = list(INDEX_DEFINITIONS[config.index_name].required_assets)
    if config.apply_cloud_mask:
        assets.append(config.cloud_mask_asset)
    return _dedupe_keep_order([asset.upper() for asset in assets])


def _apply_scl_cloud_mask(
    data: xr.DataArray,
    scl_layer: xr.DataArray,
    masked_scl_values: tuple[int, ...],
) -> xr.DataArray:
    nodata_class = np.int16(-9999)
    rounded_int = xr.where(
        np.isfinite(scl_layer),
        np.rint(scl_layer),
        nodata_class,
    ).astype("int16")
    valid_mask = (rounded_int != nodata_class) & (~rounded_int.isin(list(masked_scl_values)))
    return data.where(valid_mask)


def _compute_index_series(
    stack: xr.DataArray,
    index_definition: IndexDefinition,
) -> xr.DataArray:
    bands = {
        asset: stack.sel(band=asset).astype("float32")
        for asset in index_definition.required_assets
    }
    index = index_definition.compute(bands)
    if index_definition.clip_range is not None:
        min_value, max_value = index_definition.clip_range
        index = index.clip(min=min_value, max=max_value)
    return index.astype("float32")


def _to_temporal_multiband(data: xr.DataArray) -> xr.DataArray:
    if "time" not in data.dims:
        raise RuntimeError("Se esperaba dimension 'time' para exportar stack temporal.")

    if "band" in data.dims:
        stacked = data.stack(stack_band=("time", "band")).transpose("stack_band", "y", "x")
        labels: list[str] = []
        for time_value, band_value in stacked["stack_band"].values:
            timestamp = np.datetime_as_string(np.datetime64(time_value), unit="s")
            label = f"{timestamp}_{band_value}"
            labels.append(label)
        stacked = stacked.assign_coords(stack_band=("stack_band", labels))
        return stacked.rename({"stack_band": "band"})

    labels = [
        np.datetime_as_string(np.datetime64(raw), unit="s")
        for raw in data["time"].values
    ]
    stacked = data.transpose("time", "y", "x").rename({"time": "band"})
    return stacked.assign_coords(band=("band", labels))


def main() -> None:
    """Punto de entrada del flujo Sentinel-2.

    Flujo resumido:
    1. Cargar config, token y AOI.
    2. Buscar escenas STAC y filtrar por nubosidad.
    3. Resolver assets requeridos y preparar stack stackstac.
    4. Aplicar mascara AOI y nube por pixel.
    5. Calcular indice por escena (opcional) y exportar stack temporal.
    6. Componer intervalos temporales y exportar GeoTIFF + CSV manifiesto.
    """
    config = build_config()

    if config.max_cloud_cover is not None and not (0 <= config.max_cloud_cover <= 100):
        raise ValueError("S2_MAX_CLOUD_COVER debe estar entre 0 y 100 o ser None.")
    if config.output_resolution <= 0:
        raise ValueError("S2_OUTPUT_RESOLUTION_M debe ser > 0.")

    index_definition = None
    if config.index_name != "NONE":
        index_definition = INDEX_DEFINITIONS[config.index_name]

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
    print(f"Indice S2: {config.index_name}")
    if index_definition is None:
        print(f"Bandas pedidas: {config.requested_assets}")
    else:
        print(f"Bandas del indice: {list(index_definition.required_assets)}")
    print(f"Aplicar mascara de nubes: {config.apply_cloud_mask}")
    if config.apply_cloud_mask:
        print(
            f"Cloud mask SCL: asset={config.cloud_mask_asset} "
            f"clases={list(config.cloud_mask_scl_values)}"
        )
    print(f"stackstac rescale: {config.stackstac_rescale}")
    print(f"Exportar stack temporal: {config.export_temporal_stack}")
    print(f"Exportar compositos por intervalo: {config.export_interval_composites}")
    print(f"S3 endpoint: {CDSE_S3_ENDPOINT}")
    print(
        "Credenciales S3 configuradas: "
        f"{bool(config.aws_access_key_id and config.aws_secret_access_key)}"
    )

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

    prepared_items = [normalize_item_s3(item) for item in items]
    processing_assets = _resolve_processing_assets(config)
    asset_mapping = resolve_requested_assets(prepared_items, processing_assets)
    asset_key_by_name = {name: asset_mapping[name] for name in processing_assets}
    print(f"Assets logicos: {processing_assets}")
    print(f"Assets resueltos: {asset_mapping}")

    runtime_cache_dir: Path | None = None

    try:
        if config.use_local_asset_cache:
            base_cache_dir = config.asset_cache_dir / "s2"
            if config.cleanup_intermediate_files:
                run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                runtime_cache_dir = base_cache_dir / f"run_{run_stamp}"
                force_refresh = True
                print(
                    f"Descarga temporal assets (sin cache persistente): "
                    f"{runtime_cache_dir}"
                )
            else:
                runtime_cache_dir = base_cache_dir
                force_refresh = config.asset_cache_force_refresh
                print(f"Cache local assets habilitado: {runtime_cache_dir}")

            prepared_items = cache_assets_locally(
                items=prepared_items,
                asset_keys=[asset_key_by_name[name] for name in processing_assets],
                access_token=access_token,
                cache_dir=runtime_cache_dir,
                force_refresh=force_refresh,
            )
        else:
            print(
                "Cache local assets: deshabilitado (modo S3). "
                "Si falla acceso S3, revisa AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "o usa S2_LOCAL_ASSET_CACHE=true como fallback."
            )

        output_epsg = config.output_epsg or guess_epsg(prepared_items)
        if output_epsg is None:
            raise RuntimeError(
                "No se pudo inferir EPSG desde STAC. Define S2_OUTPUT_EPSG en .env."
            )
        print(f"EPSG salida: {output_epsg}")

        gdal_env = build_stackstac_gdal_env(
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )
        stack_dtype = np.dtype("float64" if config.stackstac_rescale else "float32")
        stack_fill_value = stack_dtype.type(np.nan)
        errors_as_nodata = (
            RasterioIOError("HTTP response code: 404"),
            RasterioIOError(r"HTTP response code: (429|5\d\d)"),
            RasterioIOError(r"Range downloading not supported by this server"),
        )

        def _build_stack_for_assets(
            logical_assets: list[str],
            resampling_mode: Resampling,
        ) -> xr.DataArray:
            stack_local = stackstac.stack(
                prepared_items,
                assets=[asset_key_by_name[name] for name in logical_assets],
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
                resampling=resampling_mode,
                properties=False,
                band_coords=False,
                gdal_env=gdal_env,
                errors_as_nodata=errors_as_nodata,
            ).assign_coords(band=("band", logical_assets))
            if config.apply_aoi_mask:
                stack_local = apply_aoi_mask(stack_local, aoi, output_epsg)
            return stack_local

        data_assets = list(processing_assets)
        if config.apply_cloud_mask:
            if config.cloud_mask_asset not in processing_assets:
                raise RuntimeError(
                    f"El asset de mascara de nubes {config.cloud_mask_asset} "
                    "no esta en los assets de trabajo."
                )
            data_assets = [asset for asset in processing_assets if asset != config.cloud_mask_asset]
            if not data_assets:
                raise RuntimeError("No quedan bandas tras separar el asset de mascara de nubes.")

        # Bandas espectrales continuas con bilinear.
        stack = _build_stack_for_assets(data_assets, Resampling.bilinear)

        if config.apply_cloud_mask:
            # SCL en stack separado con nearest para no mezclar clases.
            cloud_stack = _build_stack_for_assets(
                [config.cloud_mask_asset],
                Resampling.nearest,
            )
            cloud_layer = cloud_stack.sel(band=config.cloud_mask_asset)
            stack = _apply_scl_cloud_mask(
                data=stack,
                scl_layer=cloud_layer,
                masked_scl_values=config.cloud_mask_scl_values,
            )
            print("Mascara de nubes aplicada por pixel en cada escena.")

        if index_definition is None:
            analysis_data = stack
            analysis_token = "bands"
            export_suffix = "bands"
            report_bands = ",".join([str(v) for v in stack["band"].values])
        else:
            analysis_data = _compute_index_series(stack, index_definition)
            analysis_token = config.index_name
            export_suffix = config.index_name.lower()
            report_bands = ",".join(index_definition.required_assets)
            print(f"Indice {config.index_name} calculado para todas las escenas.")

        config.output_dir.mkdir(parents=True, exist_ok=True)
        temporal_stack_path: Path | None = None

        if config.export_temporal_stack:
            temporal_multiband = _to_temporal_multiband(analysis_data)
            temporal_stack_path = _build_temporal_stack_path(
                output_dir=config.output_dir,
                output_prefix=config.output_prefix,
                token=analysis_token,
            )
            write_multiband_geotiff(
                output_tiff=temporal_stack_path,
                data=temporal_multiband,
                output_epsg=output_epsg,
                compress=config.tiff_compress,
            )
            print(f"Stack temporal exportado: {temporal_stack_path.resolve()}")

        manifest_rows: list[dict] = []
        intervals = build_intervals(config.start_date, config.end_date, config.interval_days)
        print(f"Intervalos temporales: {len(intervals)}")

        if config.export_interval_composites:
            for idx, (start, end) in enumerate(intervals, start=1):
                window = select_time_window(analysis_data, start, end)
                if window is None:
                    continue

                composite = reduce_time_window(window, config.composite_method)
                if index_definition is None:
                    composite_to_export = composite
                else:
                    composite_to_export = composite.expand_dims(band=[config.index_name])

                output_tiff = _build_output_tiff_path(
                    output_dir=config.output_dir,
                    prefix=config.output_prefix,
                    start=start,
                    end=end,
                    suffix=export_suffix,
                )
                write_multiband_geotiff(
                    output_tiff=output_tiff,
                    data=composite_to_export,
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
                        "output_type": "index" if index_definition is not None else "bands",
                        "index_name": config.index_name if index_definition is not None else "",
                        "bands": report_bands,
                        "method": config.composite_method,
                        "cloud_mask_asset": config.cloud_mask_asset if config.apply_cloud_mask else "",
                        "cloud_mask_scl_values": (
                            ",".join(str(v) for v in config.cloud_mask_scl_values)
                            if config.apply_cloud_mask
                            else ""
                        ),
                        "temporal_stack_tiff": str(temporal_stack_path.resolve())
                        if temporal_stack_path is not None
                        else "",
                        "epsg": output_epsg,
                        "resolution_m": config.output_resolution,
                    }
                )

        manifest_csv = config.output_dir / f"{_sanitize_filename(config.output_prefix)}_manifest.csv"
        write_csv(manifest_rows, manifest_csv)

        print("\nProceso S2 completado.")
        print(f"- GeoTIFF por intervalo: {len(manifest_rows)}")
        if temporal_stack_path is not None:
            print(f"- Stack temporal: {temporal_stack_path.resolve()}")
        print(f"- Carpeta salida: {config.output_dir.resolve()}")
        print(f"- Manifest CSV: {manifest_csv.resolve()}")

    finally:
        if (
            runtime_cache_dir is not None
            and config.use_local_asset_cache
            and config.cleanup_intermediate_files
        ):
            remove_directory(runtime_cache_dir)
            print(f"Intermedios eliminados: {runtime_cache_dir}")


if __name__ == "__main__":
    main()
