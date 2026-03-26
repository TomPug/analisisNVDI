from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import getpass
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject, transform_geom
import requests
from pystac_client import Client
from pystac_client.exceptions import APIError


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / ".env"

STAC_API_URL_DEFAULT = "https://stac.dataspace.copernicus.eu/v1"
S2_COLLECTIONS_DEFAULT = ["sentinel-2-l2a"]
CDSE_OIDC_TOKEN_URL_DEFAULT = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_CLIENT_ID_DEFAULT = "cdse-public"
REQUEST_TIMEOUT_SECONDS = 120


@dataclass
class AppConfig:
    stac_api_url: str
    s2_collections: list[str]
    cdse_oidc_token_url: str
    cdse_client_id: str
    cdse_access_token: str | None
    cdse_username: str | None
    cdse_password: str | None
    ask_credentials_in_terminal: bool
    start_date: str
    end_date: str
    max_cloud_cover: float | None
    max_items: int
    aoi_vector_path: Path
    aoi_layer: str | None
    index_name: str
    output_dir: Path
    output_prefix: str | None
    export_index_tiff: bool
    index_tiff_dir: Path
    tiff_compress: str


@dataclass(frozen=True)
class IndexDefinition:
    name: str
    required_bands: tuple[str, ...]
    compute: Callable[[dict[str, np.ndarray]], np.ndarray]
    clip_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class SceneIndexResult:
    stats: dict[str, float | int]
    index_array: np.ndarray
    transform: rasterio.Affine
    crs: CRS


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = np.where(denom != 0, (a - b) / denom, np.nan)
    return nd.astype("float32")


def _compute_ndvi(bands: dict[str, np.ndarray]) -> np.ndarray:
    return _normalized_difference(bands["nir"], bands["red"])


def _compute_ndwi(bands: dict[str, np.ndarray]) -> np.ndarray:
    return _normalized_difference(bands["green"], bands["nir"])


def _compute_ndbi(bands: dict[str, np.ndarray]) -> np.ndarray:
    return _normalized_difference(bands["swir16"], bands["nir"])


def _compute_savi(bands: dict[str, np.ndarray]) -> np.ndarray:
    nir = bands["nir"]
    red = bands["red"]
    l_factor = 0.5
    denom = nir + red + l_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        savi = np.where(denom != 0, ((nir - red) / denom) * (1 + l_factor), np.nan)
    return savi.astype("float32")


def _compute_evi(bands: dict[str, np.ndarray]) -> np.ndarray:
    nir = bands["nir"]
    red = bands["red"]
    blue = bands["blue"]
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        evi = np.where(denom != 0, 2.5 * (nir - red) / denom, np.nan)
    return evi.astype("float32")


INDEX_DEFINITIONS: dict[str, IndexDefinition] = {
    "NDVI": IndexDefinition(
        name="NDVI",
        required_bands=("nir", "red"),
        compute=_compute_ndvi,
        clip_range=(-1.0, 1.0),
    ),
    "NDWI": IndexDefinition(
        name="NDWI",
        required_bands=("green", "nir"),
        compute=_compute_ndwi,
        clip_range=(-1.0, 1.0),
    ),
    "NDBI": IndexDefinition(
        name="NDBI",
        required_bands=("swir16", "nir"),
        compute=_compute_ndbi,
        clip_range=(-1.0, 1.0),
    ),
    "SAVI": IndexDefinition(
        name="SAVI",
        required_bands=("nir", "red"),
        compute=_compute_savi,
        clip_range=(-1.0, 1.0),
    ),
    "EVI": IndexDefinition(
        name="EVI",
        required_bands=("nir", "red", "blue"),
        compute=_compute_evi,
        clip_range=None,
    ),
}


BAND_ASSET_CANDIDATES: dict[str, tuple[str, ...]] = {
    "blue": ("B02", "B02_10m", "blue"),
    "green": ("B03", "B03_10m", "green"),
    "red": ("B04", "B04_10m", "red"),
    "nir": ("B08", "B08_10m", "nir"),
    "swir16": ("B11", "B11_20m", "swir16", "swir1"),
}


def load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file unless keys already exist in OS env."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Variable {name} invalida: {raw!r}. Usa true/false, 1/0, yes/no."
    )


def parse_optional_float_env(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def parse_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        value = int(raw.strip())

    if value < min_value:
        raise ValueError(f"Variable {name} debe ser >= {min_value}.")
    return value


def validate_date(date_str: str) -> str:
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def build_config_from_env() -> AppConfig:
    stac_api_url = os.getenv("STAC_API_URL", STAC_API_URL_DEFAULT).strip()

    collections_raw = os.getenv("S2_COLLECTIONS", ",".join(S2_COLLECTIONS_DEFAULT))
    collections = [part.strip() for part in collections_raw.split(",") if part.strip()]
    if not collections:
        raise ValueError("S2_COLLECTIONS no puede estar vacia.")

    start_date = validate_date(os.getenv("START_DATE", "2025-07-01").strip())
    end_date = validate_date(os.getenv("END_DATE", "2025-07-31").strip())
    if start_date > end_date:
        raise ValueError("START_DATE no puede ser mayor que END_DATE.")

    max_cloud_cover = parse_optional_float_env("MAX_CLOUD_COVER", 20.0)
    if max_cloud_cover is not None and not (0 <= max_cloud_cover <= 100):
        raise ValueError("MAX_CLOUD_COVER debe estar entre 0 y 100 o ser None.")

    max_items = parse_int_env("MAX_ITEMS", default=300, min_value=1)

    aoi_vector_raw = os.getenv("AOI_VECTOR_PATH", "").strip()
    if not aoi_vector_raw:
        raise ValueError("AOI_VECTOR_PATH es obligatorio.")

    aoi_vector_path = Path(aoi_vector_raw).expanduser()
    if not aoi_vector_path.is_absolute():
        aoi_vector_path = (SCRIPT_DIR / aoi_vector_path).resolve()

    aoi_layer = os.getenv("AOI_LAYER", "").strip() or None

    index_name = os.getenv("INDEX_NAME", "NDVI").strip().upper()
    if index_name not in INDEX_DEFINITIONS:
        valid = ", ".join(sorted(INDEX_DEFINITIONS.keys()))
        raise ValueError(f"INDEX_NAME invalido: {index_name}. Valores validos: {valid}.")

    output_dir_raw = os.getenv("OUTPUT_DIR", "outputs").strip() or "outputs"
    output_dir = Path(output_dir_raw).expanduser()
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()

    output_prefix = os.getenv("OUTPUT_PREFIX", "").strip() or None
    export_index_tiff = parse_bool_env("EXPORT_INDEX_TIFF", True)

    index_tiff_dir_raw = os.getenv("INDEX_TIFF_DIR", "").strip()
    if index_tiff_dir_raw:
        index_tiff_dir = Path(index_tiff_dir_raw).expanduser()
        if not index_tiff_dir.is_absolute():
            index_tiff_dir = (SCRIPT_DIR / index_tiff_dir).resolve()
    else:
        index_tiff_dir = (output_dir / "tiffs").resolve()

    tiff_compress = (os.getenv("TIFF_COMPRESS", "DEFLATE").strip() or "DEFLATE").upper()

    return AppConfig(
        stac_api_url=stac_api_url,
        s2_collections=collections,
        cdse_oidc_token_url=os.getenv("CDSE_OIDC_TOKEN_URL", CDSE_OIDC_TOKEN_URL_DEFAULT).strip(),
        cdse_client_id=os.getenv("CDSE_CLIENT_ID", CDSE_CLIENT_ID_DEFAULT).strip(),
        cdse_access_token=(os.getenv("CDSE_ACCESS_TOKEN", "").strip() or None),
        cdse_username=(os.getenv("CDSE_USERNAME", "").strip() or None),
        cdse_password=(os.getenv("CDSE_PASSWORD", "").strip() or None),
        ask_credentials_in_terminal=parse_bool_env("ASK_CREDENTIALS_IN_TERMINAL", True),
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        max_items=max_items,
        aoi_vector_path=aoi_vector_path,
        aoi_layer=aoi_layer,
        index_name=index_name,
        output_dir=output_dir,
        output_prefix=output_prefix,
        export_index_tiff=export_index_tiff,
        index_tiff_dir=index_tiff_dir,
        tiff_compress=tiff_compress,
    )


def validate_config(config: AppConfig) -> None:
    if not config.aoi_vector_path.exists():
        raise FileNotFoundError(f"AOI no existe: {config.aoi_vector_path}")

    suffix = config.aoi_vector_path.suffix.lower()
    if suffix not in {".shp", ".gpkg"}:
        raise ValueError("AOI_VECTOR_PATH debe ser .shp o .gpkg.")


def get_cdse_access_token(config: AppConfig) -> str | None:
    """Resolve CDSE token from env/config or request a new one with credentials."""
    if config.cdse_access_token:
        return config.cdse_access_token.strip()

    username = config.cdse_username
    password = config.cdse_password

    if (
        (not username or not password)
        and config.ask_credentials_in_terminal
        and sys.stdin
        and sys.stdin.isatty()
    ):
        print("No hay token ni credenciales CDSE configuradas.")
        username_in = input("CDSE username: ").strip()
        password_in = getpass.getpass("CDSE password: ").strip()
        if username_in and password_in:
            username = username_in
            password = password_in

    if not username or not password:
        return None

    response = requests.post(
        config.cdse_oidc_token_url,
        data={
            "client_id": config.cdse_client_id,
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            "No se pudo obtener token CDSE automaticamente. "
            f"HTTP {response.status_code}: {response.text[:300]}"
        )

    payload = response.json()
    token = payload.get("access_token")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("La respuesta OIDC no incluyo access_token valido.")

    print("Token CDSE generado automaticamente con usuario/password.")
    return token.strip()


def get_cloud_cover(item) -> float:
    """Return cloud cover using common STAC field variants."""
    props = item.properties
    return float(props.get("eo:cloud_cover", props.get("cloudCover", 100.0)))


def _api_error_status(err: APIError) -> int | None:
    response = getattr(err, "response", None)
    return getattr(response, "status_code", None)


def search_items_with_retry(
    catalog: Client,
    collections: list[str],
    start_date: str,
    end_date: str,
    bbox: tuple[float, float, float, float],
    max_items: int,
    retries: int = 4,
) -> list:
    """Search STAC items with bbox/datetime and retry on temporary gateway errors."""
    delay = 2.0
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            search = catalog.search(
                collections=collections,
                datetime=f"{start_date}/{end_date}",
                bbox=list(bbox),
                method="GET",
                limit=min(100, max_items),
                max_items=max_items,
            )
            return list(search.items())
        except APIError as err:
            status = _api_error_status(err)
            message = str(err)
            is_temporary = (
                status in {502, 503, 504}
                or "Gateway Time-out" in message
                or "timed out" in message.lower()
            )
            last_error = err
            if not is_temporary or attempt == retries:
                raise
            print(
                f"STAC temporalmente no disponible (intento {attempt}/{retries}, "
                f"status={status}). Reintentando en {delay:.0f}s..."
            )
            time.sleep(delay)
            delay *= 2

    if last_error is not None:
        raise last_error
    return []


def prefer_https_asset_hrefs(item_dict: dict) -> dict:
    """Use HTTPS alternate links when available (CDSE assets default to s3://)."""
    assets = item_dict.get("assets", {})
    for asset in assets.values():
        alternate = asset.get("alternate")
        if not isinstance(alternate, dict):
            continue
        https_alt = alternate.get("https")
        if isinstance(https_alt, dict):
            href = https_alt.get("href")
            if isinstance(href, str) and href:
                asset["href"] = href
    return item_dict


def get_https_asset_href(item_dict: dict, asset_key: str) -> str | None:
    asset = item_dict.get("assets", {}).get(asset_key, {})
    href = asset.get("href")
    return href if isinstance(href, str) and href else None


def resolve_item_assets(item, index_definition: IndexDefinition) -> dict[str, str]:
    """Resolve STAC asset keys needed for an index (red/nir/etc)."""
    assets = item.assets or {}
    available_keys = set(assets.keys())

    resolved: dict[str, str] = {}
    missing: list[str] = []

    for logical_band in index_definition.required_bands:
        candidates = BAND_ASSET_CANDIDATES.get(logical_band, ())
        match = next((key for key in candidates if key in available_keys), None)
        if match is None:
            missing.append(logical_band)
            continue
        resolved[logical_band] = match

    if missing:
        raise RuntimeError(
            f"Escena {item.id} no tiene bandas requeridas {missing}. "
            f"Assets disponibles: {sorted(available_keys)}"
        )

    return resolved


def assert_asset_accessible(sample_href: str, token: str | None) -> None:
    """Fail fast if Copernicus asset endpoint requires authentication."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with requests.get(sample_href, headers=headers, stream=True, timeout=25) as response:
        status_code = response.status_code

    if status_code == 401:
        raise RuntimeError(
            "CDSE devolvio 401 al acceder a assets raster. "
            "Configura CDSE_ACCESS_TOKEN o CDSE_USERNAME/CDSE_PASSWORD."
        )
    if status_code >= 400:
        raise RuntimeError(
            f"No se pudo acceder al asset raster (HTTP {status_code})."
        )


def download_asset_to_tempfile(href: str, token: str | None) -> Path:
    """Download a remote raster asset to a local temporary file."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    suffix = Path(href.split("?", maxsplit=1)[0]).suffix or ".dat"

    with requests.get(href, headers=headers, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
            return Path(tmp_file.name)


def load_aoi_geometries(
    vector_path: Path,
    layer: str | None,
) -> tuple[list[dict], tuple[float, float, float, float], CRS, str | None, str]:
    """Load AOI geometries from SHP/GPKG using geopandas first, then fiona fallback."""
    geopandas_error: Exception | None = None

    try:
        import geopandas as gpd

        read_kwargs = {}
        if layer:
            read_kwargs["layer"] = layer

        gdf = gpd.read_file(vector_path, **read_kwargs)
        if gdf.empty:
            raise RuntimeError("El AOI no contiene geometrias.")
        if gdf.crs is None:
            raise RuntimeError("El AOI no tiene CRS definido.")

        geometries = [
            geom.__geo_interface__
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        if not geometries:
            raise RuntimeError("No se encontraron geometrias validas en el AOI.")

        bounds = tuple(float(value) for value in gdf.total_bounds)
        return geometries, bounds, CRS.from_user_input(gdf.crs), layer, "geopandas"

    except Exception as err:
        geopandas_error = err

    try:
        import fiona

        selected_layer = layer
        if selected_layer is None and vector_path.suffix.lower() == ".gpkg":
            layers = list(fiona.listlayers(str(vector_path)))
            if not layers:
                raise RuntimeError("El geopackage no contiene capas.")
            selected_layer = layers[0]
            if len(layers) > 1:
                print(f"AOI_LAYER no definido. Se usa la primera capa: {selected_layer}")

        open_kwargs = {}
        if selected_layer:
            open_kwargs["layer"] = selected_layer

        with fiona.open(str(vector_path), **open_kwargs) as src:
            crs_raw = src.crs_wkt or src.crs
            if not crs_raw:
                raise RuntimeError("El AOI no tiene CRS definido.")

            geometries: list[dict] = []
            for feature in src:
                geometry = feature.get("geometry")
                if geometry:
                    geometries.append(geometry)

            if not geometries:
                raise RuntimeError("No se encontraron geometrias validas en el AOI.")

            bounds = tuple(float(value) for value in src.bounds)

        return geometries, bounds, CRS.from_user_input(crs_raw), selected_layer, "fiona"

    except Exception as fiona_error:
        raise RuntimeError(
            "No se pudo leer el AOI (.shp/.gpkg). "
            "Instala geopandas o fiona correctamente.\n"
            f"Error geopandas: {geopandas_error}\n"
            f"Error fiona: {fiona_error}"
        ) from fiona_error


def read_clipped_band(
    dataset: rasterio.io.DatasetReader,
    geometries: list[dict],
    geometries_crs: CRS,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Clip one raster band to AOI geometry and return float32 array + transform."""
    if dataset.crs is None:
        raise RuntimeError("Raster sin CRS definido.")

    projected_geometries = [
        transform_geom(geometries_crs, dataset.crs, geom) for geom in geometries
    ]

    data, out_transform = mask(dataset, projected_geometries, crop=True, filled=False)
    band = data[0].astype("float32")

    if np.ma.isMaskedArray(band):
        band = band.filled(np.nan).astype("float32")

    nodata = dataset.nodata
    if nodata is not None:
        try:
            nodata_value = float(nodata)
            if np.isfinite(nodata_value):
                band = np.where(np.isclose(band, nodata_value), np.nan, band)
        except (TypeError, ValueError):
            pass

    return band, out_transform


def align_band_to_reference(
    source_band: np.ndarray,
    source_transform: rasterio.Affine,
    source_crs: CRS,
    reference_shape: tuple[int, int],
    reference_transform: rasterio.Affine,
    reference_crs: CRS,
) -> np.ndarray:
    """Reproject one band onto a reference AOI grid so all arrays align."""
    destination = np.full(reference_shape, np.nan, dtype="float32")

    reproject(
        source=source_band,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=reference_transform,
        dst_crs=reference_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    return destination


def compute_scene_index_stats(
    item,
    index_definition: IndexDefinition,
    cdse_token: str | None,
    geometries: list[dict],
    geometries_crs: CRS,
) -> SceneIndexResult:
    """Download required bands, clip to AOI, compute index and descriptive stats."""
    resolved_assets = resolve_item_assets(item, index_definition)
    prepared_item = prefer_https_asset_hrefs(item.to_dict())

    href_by_band: dict[str, str] = {}
    for logical_band, asset_key in resolved_assets.items():
        href = get_https_asset_href(prepared_item, asset_key)
        if href is None:
            raise RuntimeError(
                f"Escena {item.id} no tiene href valido para asset {asset_key}."
            )
        href_by_band[logical_band] = href

    temp_paths: list[Path] = []
    try:
        local_paths: dict[str, Path] = {}
        for logical_band in index_definition.required_bands:
            local_path = download_asset_to_tempfile(href_by_band[logical_band], cdse_token)
            local_paths[logical_band] = local_path
            temp_paths.append(local_path)

        aligned_bands: dict[str, np.ndarray] = {}
        reference_transform = None
        reference_crs = None
        reference_shape: tuple[int, int] | None = None

        for position, logical_band in enumerate(index_definition.required_bands):
            with rasterio.open(local_paths[logical_band]) as dataset:
                clipped_band, clipped_transform = read_clipped_band(
                    dataset=dataset,
                    geometries=geometries,
                    geometries_crs=geometries_crs,
                )

                if position == 0:
                    aligned_bands[logical_band] = clipped_band
                    reference_transform = clipped_transform
                    reference_crs = dataset.crs
                    reference_shape = clipped_band.shape
                    continue

                if reference_shape is None or reference_transform is None or reference_crs is None:
                    raise RuntimeError("No se pudo construir la grilla de referencia.")

                aligned_bands[logical_band] = align_band_to_reference(
                    source_band=clipped_band,
                    source_transform=clipped_transform,
                    source_crs=dataset.crs,
                    reference_shape=reference_shape,
                    reference_transform=reference_transform,
                    reference_crs=reference_crs,
                )

        index_array = index_definition.compute(aligned_bands)
        if index_definition.clip_range is not None:
            min_val, max_val = index_definition.clip_range
            index_array = np.clip(index_array, min_val, max_val)

        valid = np.isfinite(index_array)
        if not np.any(valid):
            raise RuntimeError("No hay pixeles validos dentro del AOI para esta escena.")

        values = index_array[valid]
        if reference_transform is None or reference_crs is None:
            raise RuntimeError("No se pudo generar georreferenciacion del indice.")

        stats = {
            "valid_pixels": int(values.size),
            "index_min": float(np.min(values)),
            "index_p10": float(np.percentile(values, 10)),
            "index_median": float(np.percentile(values, 50)),
            "index_p90": float(np.percentile(values, 90)),
            "index_max": float(np.max(values)),
            "index_mean": float(np.mean(values)),
            "index_std": float(np.std(values)),
        }
        return SceneIndexResult(
            stats=stats,
            index_array=index_array,
            transform=reference_transform,
            crs=reference_crs,
        )

    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def get_item_datetime(item) -> datetime | None:
    for key in ("datetime", "start_datetime"):
        raw = item.properties.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            continue
    return None


def write_timeseries_csv(rows: list[dict], output_csv: Path) -> None:
    fieldnames = [
        "scene_id",
        "datetime",
        "date",
        "cloud_cover",
        "valid_pixels",
        "index_min",
        "index_p10",
        "index_median",
        "index_p90",
        "index_max",
        "index_mean",
        "index_std",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_timeseries_plot(
    rows: list[dict],
    output_png: Path,
    index_name: str,
    aoi_label: str,
) -> None:
    x_values: list[datetime] = []
    y_values: list[float] = []

    for row in rows:
        dt_raw = row.get("datetime") or row.get("date")
        if not dt_raw:
            continue
        x_values.append(datetime.fromisoformat(str(dt_raw).replace("Z", "+00:00")))
        y_values.append(float(row["index_mean"]))

    if not x_values:
        return

    plt.figure(figsize=(11, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=1.6, markersize=4, color="#1f7a3e")
    plt.grid(alpha=0.3)
    plt.title(f"Serie temporal {index_name} - AOI {aoi_label}")
    plt.xlabel("Fecha")
    plt.ylabel(f"{index_name} medio")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def sanitize_filename(text: str) -> str:
    allowed = {"-", "_"}
    chars = [char if (char.isalnum() or char in allowed) else "_" for char in text]
    cleaned = "".join(chars).strip("_")
    return cleaned or "timeseries"


def build_output_prefix(config: AppConfig) -> str:
    if config.output_prefix:
        return sanitize_filename(config.output_prefix)

    generated = f"{config.index_name.lower()}_{config.start_date}_{config.end_date}"
    return sanitize_filename(generated)


def process_candidate_items(
    candidate_items: list,
    index_definition: IndexDefinition,
    cdse_access_token: str | None,
    geometries: list[dict],
    aoi_crs: CRS,
) -> tuple[list[dict], int]:
    """Compute index stats for all candidate scenes."""
    rows: list[dict] = []
    skipped = 0

    for position, item in enumerate(candidate_items, start=1):
        item_dt = get_item_datetime(item)
        item_dt_text = item_dt.isoformat() if item_dt else ""
        cloud_cover = get_cloud_cover(item)

        print(
            f"[{position}/{len(candidate_items)}] {item.id} | "
            f"datetime={item_dt_text or 'N/A'} | cloud={cloud_cover:.2f}%"
        )

        try:
            stats = compute_scene_index_stats(
                item=item,
                index_definition=index_definition,
                cdse_token=cdse_access_token,
                geometries=geometries,
                geometries_crs=aoi_crs,
            )
        except requests.HTTPError as err:
            status = err.response.status_code if err.response is not None else None
            if status == 401:
                raise RuntimeError(
                    "CDSE devolvio 401 al descargar assets raster. "
                    "Configura CDSE_ACCESS_TOKEN o CDSE_USERNAME/CDSE_PASSWORD."
                ) from err
            skipped += 1
            print(f"  Escena saltada por error HTTP {status}: {err}")
            continue
        except RuntimeError as err:
            message = str(err)
            if "401" in message:
                raise
            skipped += 1
            print(f"  Escena saltada: {message}")
            continue
        except Exception as err:
            skipped += 1
            print(f"  Escena saltada por error no esperado: {err}")
            continue

        rows.append(
            {
                "scene_id": item.id,
                "datetime": item_dt_text,
                "date": item_dt.date().isoformat() if item_dt else "",
                "cloud_cover": round(cloud_cover, 4),
                **stats,
            }
        )

    return rows, skipped


def run_processing_with_rasterio_env(
    candidate_items: list,
    index_definition: IndexDefinition,
    cdse_access_token: str | None,
    geometries: list[dict],
    aoi_crs: CRS,
) -> tuple[list[dict], int]:
    """Run scene processing under one GDAL environment, with a Windows fallback."""
    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_DEBUG": False,
    }

    try:
        with rasterio.Env(**env_options):
            return process_candidate_items(
                candidate_items=candidate_items,
                index_definition=index_definition,
                cdse_access_token=cdse_access_token,
                geometries=geometries,
                aoi_crs=aoi_crs,
            )
    except UnicodeDecodeError as err:
        # Some Windows GDAL builds emit CP-1252 error strings; rasterio expects UTF-8.
        if os.name != "nt":
            raise

        print(
            "Detectado problema de codificacion GDAL en Windows. "
            "Reintentando con handler de errores silencioso..."
        )

        gdal_module = None
        try:
            from osgeo import gdal as gdal_module  # type: ignore
        except Exception:
            pass

        if gdal_module is None:
            raise RuntimeError(
                "GDAL devolvio mensajes no UTF-8 y no se pudo activar un "
                "handler silencioso. Prueba a ejecutar con PYTHONUTF8=1 o "
                "actualizar rasterio/GDAL en el entorno."
            ) from err

        gdal_module.PushErrorHandler("CPLQuietErrorHandler")
        try:
            with rasterio.Env(**env_options):
                return process_candidate_items(
                    candidate_items=candidate_items,
                    index_definition=index_definition,
                    cdse_access_token=cdse_access_token,
                    geometries=geometries,
                    aoi_crs=aoi_crs,
                )
        finally:
            gdal_module.PopErrorHandler()


def main() -> None:
    load_env_file(DEFAULT_ENV_PATH)
    config = build_config_from_env()
    validate_config(config)

    index_definition = INDEX_DEFINITIONS[config.index_name]
    cdse_access_token = get_cdse_access_token(config)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    geometries, bbox, aoi_crs, used_layer, aoi_reader = load_aoi_geometries(
        vector_path=config.aoi_vector_path,
        layer=config.aoi_layer,
    )

    print(f"AOI: {config.aoi_vector_path}")
    if used_layer:
        print(f"Capa AOI: {used_layer}")
    print(f"Lector AOI: {aoi_reader}")
    print(f"Features AOI: {len(geometries)}")
    print(f"BBox AOI: {bbox}")
    print(f"Indice: {index_definition.name}")
    print(f"Bandas requeridas: {index_definition.required_bands}")

    catalog = Client.open(config.stac_api_url)
    items = search_items_with_retry(
        catalog=catalog,
        collections=config.s2_collections,
        start_date=config.start_date,
        end_date=config.end_date,
        bbox=bbox,
        max_items=config.max_items,
    )

    print(f"Escenas encontradas (sin filtrar): {len(items)}")
    if not items:
        raise RuntimeError("No se encontraron escenas para el AOI y fechas indicadas.")

    candidate_items = items
    if config.max_cloud_cover is not None:
        candidate_items = [
            item
            for item in items
            if get_cloud_cover(item) <= config.max_cloud_cover
        ]
        print(
            f"Escenas con nubosidad <= {config.max_cloud_cover}%: {len(candidate_items)}"
        )
        if not candidate_items:
            raise RuntimeError(
                "No hay escenas con la nubosidad maxima indicada. "
                "Aumenta MAX_CLOUD_COVER o el rango de fechas."
            )

    candidate_items = sorted(
        candidate_items,
        key=lambda item: (
            get_item_datetime(item) or datetime.max,
            item.id,
        ),
    )

    # Fast auth check against first scene/first required band.
    for first_item in candidate_items:
        try:
            first_assets = resolve_item_assets(first_item, index_definition)
            prepared = prefer_https_asset_hrefs(first_item.to_dict())
            first_band = index_definition.required_bands[0]
            sample_href = get_https_asset_href(prepared, first_assets[first_band])
            if sample_href:
                assert_asset_accessible(sample_href, cdse_access_token)
            break
        except RuntimeError:
            continue

    rows, skipped = run_processing_with_rasterio_env(
        candidate_items=candidate_items,
        index_definition=index_definition,
        cdse_access_token=cdse_access_token,
        geometries=geometries,
        aoi_crs=aoi_crs,
    )

    if not rows:
        raise RuntimeError("No se pudieron procesar escenas validas para la serie temporal.")

    rows.sort(key=lambda row: ((row.get("datetime") or ""), row.get("scene_id") or ""))

    output_prefix = build_output_prefix(config)
    output_csv = config.output_dir / f"{output_prefix}_timeseries.csv"
    output_png = config.output_dir / f"{output_prefix}_timeseries.png"

    write_timeseries_csv(rows, output_csv)
    save_timeseries_plot(
        rows=rows,
        output_png=output_png,
        index_name=config.index_name,
        aoi_label=config.aoi_vector_path.stem,
    )

    values = np.array([float(row["index_mean"]) for row in rows], dtype="float32")

    print("\nResumen serie temporal:")
    print(f"- Escenas procesadas: {len(rows)}")
    print(f"- Escenas saltadas: {skipped}")
    print(f"- {config.index_name} medio (global): {float(np.mean(values)):.4f}")
    print(f"- {config.index_name} minimo (global): {float(np.min(values)):.4f}")
    print(f"- {config.index_name} maximo (global): {float(np.max(values)):.4f}")
    print(f"- CSV guardado en: {output_csv.resolve()}")
    print(f"- Grafica guardada en: {output_png.resolve()}")


if __name__ == "__main__":
    main()
