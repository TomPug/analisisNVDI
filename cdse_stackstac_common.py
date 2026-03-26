"""Utilidades compartidas para flujos Copernicus STAC + stackstac.

Este modulo concentra toda la logica transversal que usan los scripts de
Sentinel-1 y Sentinel-2:

1. Carga y parseo de configuracion via variables de entorno.
2. Autenticacion CDSE (token directo u OIDC password flow).
3. Lectura de AOI desde ficheros vectoriales (SHP/GPKG).
4. Busqueda STAC robusta con reintentos.
5. Normalizacion de assets y resolucion de nombres de bandas/polarizaciones.
6. Utilidades de composicion temporal y enmascarado espacial.
7. Escritura de salidas GeoTIFF y CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import getpass
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from urllib.parse import unquote, urlparse
import re

# Evita que GDAL intente cargar plugins externos de instalaciones ajenas
# (habitual en Windows con "C:\\Program Files\\GDAL\\gdalplugins"), que pueden
# no ser compatibles con la version de rasterio del entorno virtual.
#
# Si quieres conservar ese comportamiento, define:
#   STACKSTAC_KEEP_GDAL_DRIVER_PATH=true
if os.name == "nt":
    _keep_driver_path = os.getenv("STACKSTAC_KEEP_GDAL_DRIVER_PATH", "").strip().lower()
    if _keep_driver_path not in {"1", "true", "yes", "y", "on"}:
        os.environ["GDAL_DRIVER_PATH"] = ""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform_geom
import requests
import xarray as xr
from pystac_client import Client
from pystac_client.exceptions import APIError
from stackstac.rio_env import LayeredEnv


STAC_API_URL_DEFAULT = "https://stac.dataspace.copernicus.eu/v1"
CDSE_OIDC_TOKEN_URL_DEFAULT = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_CLIENT_ID_DEFAULT = "cdse-public"
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"


@dataclass(frozen=True)
class AOIInfo:
    """Contenedor inmutable con la geometria del area de estudio.

    Attributes:
        vector_path: Ruta absoluta del archivo vectorial origen.
        layer: Nombre de capa usado (si aplica, por ejemplo en GPKG).
        geometries: Lista de geometrias en formato GeoJSON-like dict.
        crs: CRS original del AOI.
        bbox_latlon: Caja minima del AOI en EPSG:4326 (minx, miny, maxx, maxy).
        label: Nombre corto del AOI, usado en archivos de salida.
    """

    vector_path: Path
    layer: str | None
    geometries: list[dict]
    crs: CRS
    bbox_latlon: tuple[float, float, float, float]
    label: str


def load_env_file(env_path: Path) -> None:
    """Carga variables desde un archivo .env sin pisar el entorno del sistema.

    El comportamiento es equivalente a un cargador de dotenv simple:
    - Ignora lineas vacias y comentarios.
    - Espera formato KEY=VALUE.
    - Si el valor viene entre comillas simples o dobles, las elimina.
    - Solo define la variable si aun no existe en ``os.environ``.

    Args:
        env_path: Ruta al archivo .env.
    """
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
    """Lee una variable de entorno booleana con validacion estricta.

    Valores aceptados para True: ``1, true, yes, y, on``.
    Valores aceptados para False: ``0, false, no, n, off``.

    Args:
        name: Nombre de la variable.
        default: Valor por defecto si la variable no existe.

    Returns:
        Valor booleano parseado.

    Raises:
        ValueError: Si la variable existe pero no coincide con valores validos.
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Variable {name} invalida: {raw!r}.")


def parse_int_env(name: str, default: int, min_value: int = 1) -> int:
    """Lee una variable de entorno entera y aplica cota minima.

    Args:
        name: Nombre de la variable.
        default: Valor por defecto si no existe o esta vacia.
        min_value: Valor minimo permitido.

    Returns:
        Entero validado.

    Raises:
        ValueError: Si el valor final es menor que ``min_value``.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = default
    else:
        value = int(raw.strip())
    if value < min_value:
        raise ValueError(f"Variable {name} debe ser >= {min_value}.")
    return value


def parse_optional_float_env(name: str, default: float | None) -> float | None:
    """Lee una variable float opcional.

    Regla:
    - Si no existe o viene vacia -> ``default``.
    - Si viene como ``none`` o ``null`` (case-insensitive) -> ``None``.
    - En cualquier otro caso intenta convertir a ``float``.

    Args:
        name: Nombre de la variable.
        default: Valor por defecto.

    Returns:
        Float parseado, ``None`` o el valor por defecto.
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        return default
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def parse_optional_int_env(name: str) -> int | None:
    """Lee una variable entera opcional.

    Args:
        name: Nombre de la variable.

    Returns:
        ``int`` si la variable tiene valor numerico; ``None`` en caso contrario
        (no definida, vacia, ``none`` o ``null``).
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def validate_date(date_str: str) -> str:
    """Valida que una fecha tenga formato ``YYYY-MM-DD``.

    Args:
        date_str: Fecha a validar.

    Returns:
        La misma fecha de entrada si es valida.

    Raises:
        ValueError: Si el formato no es correcto o la fecha no existe.
    """
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def parse_csv_env(name: str, default_csv: str) -> list[str]:
    """Parsea una variable separada por comas en una lista de strings.

    Args:
        name: Nombre de la variable.
        default_csv: Valor por defecto en formato CSV.

    Returns:
        Lista de tokens sin espacios externos y sin vacios.

    Raises:
        ValueError: Si despues del parseo la lista queda vacia.
    """
    raw = os.getenv(name, default_csv)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Variable {name} no puede estar vacia.")
    return values


def get_cdse_access_token(
    access_token: str | None,
    username: str | None,
    password: str | None,
    ask_credentials_in_terminal: bool,
    oidc_token_url: str = CDSE_OIDC_TOKEN_URL_DEFAULT,
    client_id: str = CDSE_CLIENT_ID_DEFAULT,
) -> str | None:
    """Resuelve un token de acceso CDSE.

    Estrategia en orden:
    1. Usar ``access_token`` si viene configurado.
    2. Si no hay token, usar ``username/password`` recibidos.
    3. Si faltan credenciales y se permite terminal interactiva, pedirlas.
    4. Solicitar token OIDC usando grant_type=password.

    Args:
        access_token: Token bearer ya disponible.
        username: Usuario CDSE.
        password: Password CDSE.
        ask_credentials_in_terminal: Si ``True``, permite prompt interactivo.
        oidc_token_url: Endpoint OIDC token.
        client_id: Client ID OIDC.

    Returns:
        Token bearer o ``None`` si no hubo credenciales para solicitarlo.

    Raises:
        RuntimeError: Si OIDC responde error o el payload no trae access_token.
    """
    if access_token:
        return access_token.strip()

    user = username
    pwd = password

    if (not user or not pwd) and ask_credentials_in_terminal and sys.stdin and sys.stdin.isatty():
        print("No hay token ni credenciales CDSE configuradas.")
        user_in = input("CDSE username: ").strip()
        pwd_in = getpass.getpass("CDSE password: ").strip()
        if user_in and pwd_in:
            user = user_in
            pwd = pwd_in

    if not user or not pwd:
        return None

    response = requests.post(
        oidc_token_url,
        data={
            "client_id": client_id,
            "grant_type": "password",
            "username": user,
            "password": pwd,
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

    return token.strip()


def load_aoi_info(vector_path: Path, layer: str | None = None) -> AOIInfo:
    """Carga y valida un AOI desde SHP/GPKG.

    Ademas de leer geometrias validas, reproyecta temporalmente a EPSG:4326 para
    calcular la caja minima en lat/lon (util para consultas STAC por ``bbox``).

    Args:
        vector_path: Ruta al archivo vectorial (.shp o .gpkg).
        layer: Capa opcional (especialmente util para geopackage).

    Returns:
        Instancia ``AOIInfo`` con metadatos y geometrias listas para usar.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la extension no es soportada.
        RuntimeError: Si el AOI no tiene geometrias o no tiene CRS.
    """
    if not vector_path.exists():
        raise FileNotFoundError(f"AOI no existe: {vector_path}")
    if vector_path.suffix.lower() not in {".shp", ".gpkg"}:
        raise ValueError("AOI debe ser .shp o .gpkg.")

    read_kwargs: dict[str, Any] = {}
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
        raise RuntimeError("No hay geometrias validas en el AOI.")

    gdf_latlon = gdf.to_crs(4326)
    bounds = tuple(float(v) for v in gdf_latlon.total_bounds)

    return AOIInfo(
        vector_path=vector_path.resolve(),
        layer=layer,
        geometries=geometries,
        crs=CRS.from_user_input(gdf.crs),
        bbox_latlon=(bounds[0], bounds[1], bounds[2], bounds[3]),
        label=vector_path.stem,
    )


def _api_error_status(err: APIError) -> int | None:
    """Extrae el codigo HTTP de un ``APIError`` de pystac-client.

    Args:
        err: Excepcion lanzada por pystac-client.

    Returns:
        Codigo de estado si existe en ``err.response``, o ``None``.
    """
    response = getattr(err, "response", None)
    return getattr(response, "status_code", None)


def search_items_with_retry(
    catalog: Client,
    collections: list[str],
    start_date: str,
    end_date: str,
    bbox_latlon: tuple[float, float, float, float],
    max_items: int,
    retries: int = 4,
) -> list:
    """Ejecuta una busqueda STAC con reintentos para errores temporales.

    Reintenta con backoff exponencial ante 502/503/504 o timeouts de gateway.

    Args:
        catalog: Cliente STAC abierto.
        collections: Colecciones STAC a consultar.
        start_date: Fecha inicial ``YYYY-MM-DD``.
        end_date: Fecha final ``YYYY-MM-DD``.
        bbox_latlon: BBOX en EPSG:4326.
        max_items: Maximo total de items a recuperar.
        retries: Numero maximo de intentos.

    Returns:
        Lista de items STAC.

    Raises:
        APIError: Si falla con error no temporal o se agotan reintentos.
    """
    delay = 2.0
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            search = catalog.search(
                collections=collections,
                datetime=f"{start_date}/{end_date}",
                bbox=list(bbox_latlon),
                method="GET",
                limit=min(100, max_items),
                max_items=max_items,
            )
            return list(search.items())
        except APIError as err:
            status = _api_error_status(err)
            message = str(err)
            temporary = (
                status in {502, 503, 504}
                or "Gateway Time-out" in message
                or "timed out" in message.lower()
            )
            last_error = err
            if not temporary or attempt == retries:
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


def get_cloud_cover(item: Any) -> float:
    """Obtiene nubosidad de un item usando variantes comunes de propiedad.

    Args:
        item: Item STAC como objeto ``pystac.Item`` o ``dict``.

    Returns:
        Nubosidad en porcentaje. Si no existe propiedad, devuelve 100.0.
    """
    props = item.properties if hasattr(item, "properties") else item.get("properties", {})
    return float(props.get("eo:cloud_cover", props.get("cloudCover", 100.0)))


def _zipper_href_to_s3_href(href: str) -> str | None:
    """Convierte un href zipper OData a un href S3 best-effort.

    Nota: esta conversion es de respaldo. El camino preferido es usar
    ``asset['alternate']['s3']['href']`` cuando exista.
    """
    lower = href.strip().lower()
    if not (
        lower.startswith("https://zipper.dataspace.copernicus.eu/")
        or lower.startswith("http://zipper.dataspace.copernicus.eu/")
    ):
        return None

    parsed = urlparse(href)
    raw_nodes = re.findall(r"/Nodes\(([^)]+)\)", parsed.path)
    if not raw_nodes:
        return None

    nodes: list[str] = []
    for raw in raw_nodes:
        text = unquote(raw.strip().strip("'\""))
        if text:
            nodes.append(text)
    if not nodes:
        return None

    return f"s3://eodata/{'/'.join(nodes)}"


def normalize_item_s3(item: Any) -> dict:
    """Normaliza assets para priorizar href S3.

    Orden de resolucion por asset:
    1. ``alternate.s3.href``.
    2. ``href`` actual si ya es ``s3://``.
    3. Conversion best-effort desde URL zipper OData.
    """
    item_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)
    for asset in item_dict.get("assets", {}).values():
        if not isinstance(asset, dict):
            continue

        resolved_href: str | None = None
        alternate = asset.get("alternate")
        if isinstance(alternate, dict):
            s3_alt = alternate.get("s3")
            if isinstance(s3_alt, dict):
                href = s3_alt.get("href")
                if isinstance(href, str) and href.strip():
                    resolved_href = href.strip()

        if resolved_href is None:
            href = asset.get("href")
            if isinstance(href, str) and href.strip():
                href = href.strip()
                if href.lower().startswith("s3://"):
                    resolved_href = href
                else:
                    resolved_href = _zipper_href_to_s3_href(href)

        if resolved_href:
            asset["href"] = resolved_href

    return item_dict


ASSET_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "B02": ("blue",),
    "B03": ("green",),
    "B04": ("red",),
    "B08": ("nir",),
    "B8A": ("nir08",),
    "B11": ("swir16", "swir1"),
    "B12": ("swir22", "swir2"),
    "SCL": ("scene_classification", "cloud_mask"),
    "VV": ("vv",),
    "VH": ("vh",),
    "HH": ("hh",),
    "HV": ("hv",),
}


def _asset_candidates(requested_asset: str) -> list[str]:
    """Construye candidatos de nombre de asset para resolver aliases.

    Ejemplo para ``B08``: prueba ``B08``, ``b08``, ``B08_10m``, ``nir``, etc.

    Args:
        requested_asset: Asset pedido por usuario.

    Returns:
        Lista ordenada de claves candidatas (sin duplicados).
    """
    req = requested_asset.strip()
    req_upper = req.upper()
    candidates: list[str] = [req, req_upper, req.lower()]

    if req_upper.startswith("B"):
        candidates.extend([f"{req_upper}_10m", f"{req_upper}_20m", f"{req_upper}_60m"])
    elif "_" not in req_upper:
        candidates.extend([f"{req_upper}_10m", f"{req_upper}_20m", f"{req_upper}_60m"])

    for alias in ASSET_ALIAS_MAP.get(req_upper, ()):
        candidates.extend([alias, alias.upper(), alias.lower()])

    ordered: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def resolve_requested_assets(items: list[dict], requested_assets: list[str]) -> dict[str, str]:
    """Resuelve nombres logicos de assets contra claves reales STAC.

    Busca el primer item capaz de mapear todos los assets pedidos.

    Args:
        items: Lista de items STAC (dict) ya normalizados.
        requested_assets: Nombres logicos pedidos (p.ej. ``['B04', 'B08']``).

    Returns:
        Diccionario ``{asset_pedido: asset_real_en_stac}``.

    Raises:
        RuntimeError: Si no se logra resolver todos los assets.
    """
    for item in items:
        available = set(item.get("assets", {}).keys())
        mapping: dict[str, str] = {}
        ok = True
        for requested in requested_assets:
            match = next((c for c in _asset_candidates(requested) if c in available), None)
            if match is None:
                ok = False
                break
            mapping[requested] = match
        if ok:
            return mapping

    sample_assets = sorted(items[0].get("assets", {}).keys()) if items else []
    raise RuntimeError(
        f"No se pudieron resolver assets {requested_assets}. "
        f"Ejemplo de assets disponibles: {sample_assets}"
    )


def guess_epsg(items: list[dict]) -> int | None:
    """Intenta inferir EPSG de salida a partir de metadatos STAC.

    Orden de busqueda:
    1. ``item.properties['proj:epsg']``
    2. ``item.properties['proj:code']`` tipo ``EPSG:XXXX``
    3. Mismas claves en cada asset.

    Args:
        items: Lista de items STAC (dict).

    Returns:
        Codigo EPSG o ``None`` si no puede inferirse.
    """
    for item in items:
        props = item.get("properties", {})
        epsg = props.get("proj:epsg")
        if isinstance(epsg, int):
            return epsg

        proj_code = props.get("proj:code")
        if isinstance(proj_code, str) and proj_code.upper().startswith("EPSG:"):
            return int(proj_code.split(":", maxsplit=1)[1])

        for asset in item.get("assets", {}).values():
            if not isinstance(asset, dict):
                continue
            asset_epsg = asset.get("proj:epsg")
            if isinstance(asset_epsg, int):
                return asset_epsg

            asset_code = asset.get("proj:code")
            if isinstance(asset_code, str) and asset_code.upper().startswith("EPSG:"):
                return int(asset_code.split(":", maxsplit=1)[1])
    return None


def build_stackstac_gdal_env(
    aws_access_key_id: str | None,
    aws_secret_access_key: str | None,
) -> LayeredEnv:
    """Construye entorno GDAL para lectura directa S3 en CDSE."""
    options: dict[str, Any] = {
        "AWS_S3_ENDPOINT": CDSE_S3_ENDPOINT,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "CPL_DEBUG": False,
    }
    if aws_access_key_id:
        options["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    if aws_secret_access_key:
        options["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    return LayeredEnv(always=options)


def _sanitize_path_token(text: str) -> str:
    """Devuelve un token seguro para nombres de archivo/directorio."""
    safe_chars = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    token = "".join(safe_chars).strip("._")
    return token or "asset"


def _is_http_href(href: str) -> bool:
    """Indica si un href apunta a recurso HTTP/HTTPS remoto."""
    lower = href.strip().lower()
    return lower.startswith("http://") or lower.startswith("https://")


def _guess_asset_suffix(href: str, fallback: str = ".bin") -> str:
    """Intenta inferir extension de archivo desde la URL del asset."""
    parsed = urlparse(href)
    candidate = Path(parsed.path).name
    if "." in candidate:
        suffix = Path(candidate).suffix
        if suffix:
            return suffix
    return fallback


def _download_asset_with_retry(
    session: requests.Session,
    url: str,
    target_path: Path,
    timeout_sec: int,
    max_attempts: int,
    retry_base_delay_sec: float,
) -> None:
    """Descarga un asset remoto con reintentos y backoff exponencial."""
    retryable_status = {429, 500, 502, 503, 504}
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".part")

    for attempt in range(1, max_attempts + 1):
        try:
            with session.get(url, stream=True, timeout=timeout_sec) as response:
                status_code = response.status_code
                if status_code in retryable_status:
                    raise requests.HTTPError(
                        f"HTTP {status_code}", response=response
                    )
                response.raise_for_status()

                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            temp_path.replace(target_path)
            return
        except Exception as exc:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = (
                isinstance(exc, (requests.ConnectionError, requests.Timeout))
                or status_code in retryable_status
            )
            if (not retryable) or attempt >= max_attempts:
                raise

            wait_seconds = retry_base_delay_sec * (2 ** (attempt - 1))
            print(
                f"Reintento descarga asset ({attempt}/{max_attempts}) en "
                f"{wait_seconds:.1f}s | status={status_code or 'n/a'}"
            )
            time.sleep(wait_seconds)


def cache_assets_locally(
    items: list[dict],
    asset_keys: list[str],
    access_token: str | None,
    cache_dir: Path,
    force_refresh: bool = False,
) -> list[dict]:
    """Descarga assets remotos a cache local y actualiza href en items STAC.

    Esta funcion resuelve fallos de lectura remota ``JP2`` en CDSE (por ejemplo
    mensajes de tipo "Range downloading not supported by this server!") al
    sustituir cada ``href`` HTTP por un archivo local descargado.

    Args:
        items: Items STAC normalizados como diccionarios.
        asset_keys: Claves de asset a cachear (p. ej. ``["B02_10m", ...]``).
        access_token: Token bearer para cabecera Authorization (opcional).
        cache_dir: Directorio raiz del cache local.
        force_refresh: Si ``True``, redescarga aunque el archivo ya exista.

    Returns:
        La misma lista de items, con ``href`` de assets apuntando a ruta local.
    """
    if not items or not asset_keys:
        return items

    cache_dir.mkdir(parents=True, exist_ok=True)
    timeout_sec = parse_int_env("STACKSTAC_ASSET_DOWNLOAD_TIMEOUT_SEC", default=300, min_value=30)
    max_attempts = parse_int_env("STACKSTAC_ASSET_DOWNLOAD_MAX_ATTEMPTS", default=4, min_value=1)
    retry_base_delay = parse_optional_float_env(
        "STACKSTAC_ASSET_DOWNLOAD_RETRY_BASE_DELAY_SEC", default=2.0
    )
    if retry_base_delay is None or retry_base_delay <= 0:
        raise ValueError("STACKSTAC_ASSET_DOWNLOAD_RETRY_BASE_DELAY_SEC debe ser > 0.")

    total_assets = len(items) * len(asset_keys)
    processed = 0
    downloaded = 0
    reused = 0

    session = requests.Session()
    if access_token:
        session.headers.update({"Authorization": f"Bearer {access_token}"})

    try:
        for item in items:
            item_id = str(item.get("id", "item"))
            assets = item.get("assets", {})
            if not isinstance(assets, dict):
                continue

            for asset_key in asset_keys:
                processed += 1
                asset = assets.get(asset_key)
                if not isinstance(asset, dict):
                    continue
                href = asset.get("href")
                if not isinstance(href, str) or not href.strip():
                    continue

                href = href.strip()
                if not _is_http_href(href):
                    continue

                suffix = _guess_asset_suffix(href, fallback=".jp2")
                filename = (
                    f"{_sanitize_path_token(item_id)}__"
                    f"{_sanitize_path_token(asset_key)}{suffix}"
                )
                local_path = cache_dir / filename

                if local_path.exists() and local_path.stat().st_size > 0 and not force_refresh:
                    asset["href"] = str(local_path.resolve())
                    reused += 1
                    continue

                print(
                    f"[cache {processed}/{total_assets}] Descargando "
                    f"{item_id} | {asset_key}"
                )
                _download_asset_with_retry(
                    session=session,
                    url=href,
                    target_path=local_path,
                    timeout_sec=timeout_sec,
                    max_attempts=max_attempts,
                    retry_base_delay_sec=retry_base_delay,
                )
                asset["href"] = str(local_path.resolve())
                downloaded += 1
    finally:
        session.close()

    print(
        "Cache local assets completado: "
        f"total={total_assets}, descargados={downloaded}, reutilizados={reused}"
    )
    return items


def build_intervals(start_date: str, end_date: str, interval_days: int) -> list[tuple[datetime, datetime]]:
    """Genera ventanas temporales semiabiertas [start, end) de longitud fija.

    La fecha final de entrada se interpreta como inclusiva en terminos de dia.

    Args:
        start_date: Fecha inicial ``YYYY-MM-DD``.
        end_date: Fecha final inclusiva ``YYYY-MM-DD``.
        interval_days: Tamano de ventana en dias.

    Returns:
        Lista de tuplas ``(inicio, fin_exclusivo)``.

    Raises:
        ValueError: Si ``interval_days < 1``.
    """
    if interval_days < 1:
        raise ValueError("interval_days debe ser >= 1.")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end_inclusive = datetime.strptime(end_date, "%Y-%m-%d")
    end_exclusive = end_inclusive + timedelta(days=1)

    intervals: list[tuple[datetime, datetime]] = []
    current = start
    while current < end_exclusive:
        next_dt = min(current + timedelta(days=interval_days), end_exclusive)
        intervals.append((current, next_dt))
        current = next_dt
    return intervals


def get_item_datetime(item: Any) -> datetime | None:
    """Extrae datetime util de un item STAC.

    Busca primero ``datetime`` y luego ``start_datetime``.

    Args:
        item: Item STAC objeto o dict.

    Returns:
        ``datetime`` parseado o ``None`` si no hay fecha valida.
    """
    props = item.properties if hasattr(item, "properties") else item.get("properties", {})
    for key in ("datetime", "start_datetime"):
        raw = props.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            continue
    return None


def select_time_window(data: xr.DataArray, start: datetime, end: datetime) -> xr.DataArray | None:
    """Selecciona una subserie temporal [start, end) desde un DataArray stackstac.

    Args:
        data: DataArray con dimension ``time``.
        start: Inicio inclusivo.
        end: Fin exclusivo.

    Returns:
        Sub-DataArray con los tiempos dentro del intervalo, o ``None`` si vacio.
    """
    time_values = np.asarray(data["time"].values)
    start64 = np.datetime64(start.isoformat())
    end64 = np.datetime64(end.isoformat())
    index = np.where((time_values >= start64) & (time_values < end64))[0]
    if index.size == 0:
        return None
    return data.isel(time=index)


def reduce_time_window(data: xr.DataArray, method: str) -> xr.DataArray:
    """Reduce la dimension temporal con un metodo estadistico.

    Metodos soportados: ``median``, ``mean``, ``max``, ``min``.

    Args:
        data: DataArray con dimension ``time``.
        method: Metodo de composicion.

    Returns:
        DataArray sin dimension ``time``.

    Raises:
        ValueError: Si el metodo no esta soportado.
    """
    mode = method.strip().lower()
    if mode == "median":
        return data.median(dim="time", skipna=True)
    if mode == "mean":
        return data.mean(dim="time", skipna=True)
    if mode == "max":
        return data.max(dim="time", skipna=True)
    if mode == "min":
        return data.min(dim="time", skipna=True)
    raise ValueError(f"Metodo de composicion no soportado: {method}")


def _transform_from_coords(data: xr.DataArray) -> Affine:
    """Deriva la transformacion affine a partir de coordenadas x/y.

    Args:
        data: DataArray con coordenadas ``x`` y ``y`` regularmente espaciadas.

    Returns:
        ``Affine`` con origen y resolucion de la grilla.

    Raises:
        RuntimeError: Si no hay suficientes coordenadas para inferir resolucion.
    """
    x = np.asarray(data["x"].values, dtype="float64")
    y = np.asarray(data["y"].values, dtype="float64")
    if x.size < 2 or y.size < 2:
        raise RuntimeError("No se pudo inferir transformacion (x/y insuficientes).")

    x_res = float(np.abs(np.diff(x).mean()))
    y_res = float(np.abs(np.diff(y).mean()))
    left = float(np.min(x)) - (x_res / 2.0)
    top = float(np.max(y)) + (y_res / 2.0)
    return from_origin(left, top, x_res, y_res)


def apply_aoi_mask(data: xr.DataArray, aoi: AOIInfo, output_epsg: int) -> xr.DataArray:
    """Enmascara un raster xarray con el AOI poligonal.

    Convierte geometrias AOI al CRS de salida y genera una mascara booleana
    sobre la grilla ``(y, x)``. Los pixeles fuera del AOI pasan a NaN.

    Args:
        data: Stack/composite con dims ``(..., y, x)``.
        aoi: Informacion de AOI.
        output_epsg: EPSG de la grilla raster.

    Returns:
        DataArray enmascarado.
    """
    transform = _transform_from_coords(data)
    projected_geoms = [
        transform_geom(aoi.crs, CRS.from_epsg(output_epsg), geom)
        for geom in aoi.geometries
    ]
    mask2d = geometry_mask(
        projected_geoms,
        out_shape=(data.sizes["y"], data.sizes["x"]),
        transform=transform,
        invert=True,
    )
    mask_da = xr.DataArray(mask2d, coords={"y": data["y"], "x": data["x"]}, dims=("y", "x"))
    return data.where(mask_da)


def write_multiband_geotiff(
    output_tiff: Path,
    data: xr.DataArray,
    output_epsg: int,
    compress: str = "DEFLATE",
    nodata: float = -9999.0,
) -> None:
    """Escribe un DataArray multibanda a GeoTIFF float32.

    Requisitos:
    - El DataArray debe tener dimension ``band``.
    - Se asume orden espacial en coordenadas ``x`` y ``y``.

    Args:
        output_tiff: Ruta de salida.
        data: DataArray con dims ``(band, y, x)``.
        output_epsg: EPSG de salida.
        compress: Metodo de compresion TIFF.
        nodata: Valor nodata a escribir.

    Raises:
        RuntimeError: Si falta la dimension ``band``.
    """
    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    transform = _transform_from_coords(data)

    if "band" not in data.dims:
        raise RuntimeError("La salida para GeoTIFF debe tener dimension 'band'.")

    values_da = data.transpose("band", "y", "x").astype("float32")

    # CDSE puede responder 429 cuando se hacen muchas lecturas remotas a la vez.
    # Se fuerza un numero bajo de workers y se reintenta con backoff exponencial.
    dask_workers = parse_int_env("STACKSTAC_DASK_WORKERS", default=4, min_value=1)
    max_attempts = parse_int_env("STACKSTAC_HTTP_MAX_ATTEMPTS", default=4, min_value=1)
    retry_base_delay = parse_optional_float_env(
        "STACKSTAC_HTTP_RETRY_BASE_DELAY_SEC", default=2.0
    )
    if retry_base_delay is None or retry_base_delay <= 0:
        raise ValueError("STACKSTAC_HTTP_RETRY_BASE_DELAY_SEC debe ser > 0.")

    values: np.ndarray | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            values = values_da.compute(
                scheduler="threads",
                num_workers=dask_workers,
            ).values
            break
        except Exception as exc:
            message = str(exc).lower()
            is_retryable_http = (
                "http response code: 429" in message
                or "http response code: 500" in message
                or "http response code: 502" in message
                or "http response code: 503" in message
                or "http response code: 504" in message
            )
            if (not is_retryable_http) or attempt >= max_attempts:
                raise

            wait_seconds = retry_base_delay * (2 ** (attempt - 1))
            print(
                f"Advertencia HTTP temporal al leer assets remotos "
                f"(intento {attempt}/{max_attempts}). "
                f"Reintentando en {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)

    if values is None:
        raise RuntimeError("No se pudo materializar el DataArray para exportar GeoTIFF.")

    values = np.where(np.isfinite(values), values, np.float32(nodata)).astype("float32")

    band_count, height, width = values.shape
    with rasterio.open(
        output_tiff,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=band_count,
        dtype="float32",
        crs=CRS.from_epsg(output_epsg),
        transform=transform,
        nodata=float(nodata),
        compress=compress,
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(values)
        band_labels = [str(v) for v in data["band"].values]
        for index, label in enumerate(band_labels, start=1):
            dst.set_band_description(index, label)


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Escribe una lista de diccionarios a CSV.

    Toma las cabeceras del primer registro. Si ``rows`` esta vacio, crea
    un archivo vacio para mantener contrato de salida.

    Args:
        rows: Filas a persistir.
        output_csv: Ruta del CSV de salida.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def remove_directory(path: Path) -> None:
    """Elimina un directorio completo si existe (ignora errores)."""
    if not path.exists():
        return
    shutil.rmtree(path, ignore_errors=True)
