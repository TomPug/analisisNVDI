from __future__ import annotations

from datetime import datetime
import getpass
import os
from pathlib import Path
import tempfile
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import requests
from pystac_client import Client
from pystac_client.exceptions import APIError


# -----------------------------
# CONFIG: edit only these values
# -----------------------------
STAC_API_URL = "https://stac.dataspace.copernicus.eu/v1"
S2_COLLECTIONS = ["sentinel-2-l2a"]
CDSE_OIDC_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_CLIENT_ID = "cdse-public"
CDSE_ACCESS_TOKEN = None  # Opcional: token Bearer de Copernicus (sin "Bearer ")
CDSE_USERNAME = None  # Opcional: usuario CDSE para generar token automatico
CDSE_PASSWORD = None  # Opcional: password CDSE para generar token automatico
ASK_CREDENTIALS_IN_TERMINAL = True  # Si falta token, pedir usuario/password por consola
TILE = "30STJ"
START_DATE = "2025-07-01"
END_DATE = "2025-07-31"
PROCESSING_RESOLUTION_M = 10
CROP_FRACTION = 0.12
MAX_CLOUD_COVER = 10.0
PREFERRED_SCENE_ID = "S2A_MSIL2A_20250731T110651_R137_T30STJ_20250731T163616"


def validate_date(date_str: str) -> str:
    """Validate YYYY-MM-DD date format for STAC queries."""
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def get_cloud_cover(item) -> float:
    """Return cloud cover using common STAC field variants."""
    props = item.properties
    return float(props.get("eo:cloud_cover", props.get("cloudCover", 100.0)))


def get_band_assets(item) -> tuple[str, str]:
    """Resolve red/NIR asset keys across Sentinel-2 STAC variants."""
    assets = item.assets or {}
    keys = set(assets.keys())
    if {"B04", "B08"}.issubset(keys):
        return "B04", "B08"
    if {"red", "nir"}.issubset(keys):
        return "red", "nir"
    if {"B04_10m", "B08_10m"}.issubset(keys):
        return "B04_10m", "B08_10m"
    raise RuntimeError(
        "No se encontraron bandas rojo/NIR esperadas en assets "
        f"({sorted(keys)})."
    )


def _api_error_status(err: APIError) -> int | None:
    """Extract HTTP status code from pystac-client APIError when available."""
    response = getattr(err, "response", None)
    return getattr(response, "status_code", None)


def search_items_with_retry(
    catalog: Client,
    collections: list[str],
    start_date: str,
    end_date: str,
    tile: str,
    retries: int = 4,
) -> list:
    """Search STAC items using CQL2 filter and retry on temporary gateway errors."""
    tile_upper = tile.upper()
    cql_filter = f"id LIKE 'S2%_T{tile_upper}_%'"
    delay = 2.0
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            search = catalog.search(
                collections=collections,
                datetime=f"{start_date}/{end_date}",
                filter_lang="cql2-text",
                filter=cql_filter,
                method="GET",
                limit=100,
                max_items=500,
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


def get_item_epsg(item, preferred_assets: tuple[str, str]) -> int | None:
    """Resolve EPSG from item properties first, then from selected assets."""
    epsg = item.properties.get("proj:epsg")
    if isinstance(epsg, int):
        return epsg

    proj_code = item.properties.get("proj:code")
    if isinstance(proj_code, str) and proj_code.upper().startswith("EPSG:"):
        return int(proj_code.split(":", maxsplit=1)[1])

    for key in preferred_assets:
        asset = item.assets.get(key)
        if asset is None:
            continue
        asset_epsg = asset.extra_fields.get("proj:epsg")
        if isinstance(asset_epsg, int):
            return asset_epsg
        asset_proj_code = asset.extra_fields.get("proj:code")
        if isinstance(asset_proj_code, str) and asset_proj_code.upper().startswith("EPSG:"):
            return int(asset_proj_code.split(":", maxsplit=1)[1])

    return None


def get_https_asset_href(item_dict: dict, asset_key: str) -> str | None:
    """Return current href for an asset after alternate-link normalization."""
    asset = item_dict.get("assets", {}).get(asset_key, {})
    href = asset.get("href")
    return href if isinstance(href, str) and href else None


def assert_asset_accessible(sample_href: str, token: str | None) -> None:
    """Fail fast if Copernicus asset endpoint requires authentication."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    response = requests.get(sample_href, headers=headers, stream=True, timeout=25)
    response.close()
    if response.status_code == 401:
        raise RuntimeError(
            "CDSE devolvio 401 al acceder a los assets raster. "
            "Define CDSE_ACCESS_TOKEN (env o config), o bien CDSE_USERNAME y "
            "CDSE_PASSWORD para generar token automaticamente."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"No se pudo acceder al asset raster (HTTP {response.status_code})."
        )


def download_asset_to_tempfile(href: str, token: str | None, suffix: str = ".jp2") -> Path:
    """Download remote raster asset to a temporary local file."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with requests.get(href, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp_file.write(chunk)
            return Path(tmp_file.name)


def compute_center_window(
    width: int,
    height: int,
    crop_fraction: float,
) -> tuple[int, int, int, int]:
    """Compute centered crop window dimensions from crop fraction."""
    if crop_fraction <= 0 or crop_fraction > 0.5:
        return 0, 0, width, height

    out_w = max(1, int(round(width * (2 * crop_fraction))))
    out_h = max(1, int(round(height * (2 * crop_fraction))))
    out_w = min(out_w, width)
    out_h = min(out_h, height)
    col_off = max(0, (width - out_w) // 2)
    row_off = max(0, (height - out_h) // 2)
    return col_off, row_off, out_w, out_h


def get_cdse_access_token() -> str | None:
    """Resolve CDSE token from env/config or request a new one with credentials."""
    env_token = os.getenv("CDSE_ACCESS_TOKEN")
    if env_token:
        return env_token.strip()

    if CDSE_ACCESS_TOKEN:
        return CDSE_ACCESS_TOKEN.strip()

    username = os.getenv("CDSE_USERNAME") or CDSE_USERNAME
    password = os.getenv("CDSE_PASSWORD") or CDSE_PASSWORD
    client_id = os.getenv("CDSE_CLIENT_ID") or CDSE_CLIENT_ID

    if (
        (not username or not password)
        and ASK_CREDENTIALS_IN_TERMINAL
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
        CDSE_OIDC_TOKEN_URL,
        data={
            "client_id": client_id,
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


def main() -> None:
    tile = TILE
    start_date = START_DATE
    end_date = END_DATE
    processing_resolution_m = PROCESSING_RESOLUTION_M
    crop_fraction = CROP_FRACTION
    preferred_scene_id = PREFERRED_SCENE_ID
    max_cloud_cover = MAX_CLOUD_COVER
    cdse_access_token = get_cdse_access_token()

    validate_date(start_date)
    validate_date(end_date)
    if not (0 < crop_fraction <= 0.5):
        raise ValueError("CROP_FRACTION debe estar entre 0 y 0.5.")
    if max_cloud_cover is not None and not (0 <= max_cloud_cover <= 100):
        raise ValueError("MAX_CLOUD_COVER debe estar entre 0 y 100, o ser None.")

    # Open Copernicus Data Space STAC catalog
    catalog = Client.open(STAC_API_URL)

    # Search Sentinel-2 imagery for a tile and date range
    items = search_items_with_retry(
        catalog=catalog,
        collections=S2_COLLECTIONS,
        start_date=start_date,
        end_date=end_date,
        tile=tile,
    )
    print(f"Encontrados: {len(items)}")

    if not items:
        raise RuntimeError("No se encontraron imagenes para el tile y fechas indicadas.")

    candidate_items = items
    if max_cloud_cover is not None:
        candidate_items = [
            it
            for it in items
            if get_cloud_cover(it) <= max_cloud_cover
        ]
        print(
            f"Escenas con nubosidad <= {max_cloud_cover}%: {len(candidate_items)}"
        )
        if not candidate_items:
            raise RuntimeError(
                "No hay escenas con la nubosidad maxima indicada. "
                "Aumenta MAX_CLOUD_COVER o usa otro rango de fechas."
            )

    # Pick one scene to process: exact preferred scene first, otherwise lowest cloud cover.
    best_item = None
    if preferred_scene_id:
        best_item = next((it for it in candidate_items if it.id == preferred_scene_id), None)
    if best_item is None:
        best_item = min(
            candidate_items, key=get_cloud_cover
        )
        if preferred_scene_id:
            print(
                "Escena preferida no encontrada dentro del filtro; usando menor nubosidad."
            )
        else:
            print("Sin escena fija: usando menor nubosidad.")
    else:
        print("Escena preferida encontrada y seleccionada.")

    cloud_cover = get_cloud_cover(best_item)
    print(f"Escena seleccionada: {best_item.id} | Nubosidad: {cloud_cover}")
    print(f"Resolucion de proceso: {processing_resolution_m} m")

    red_band, nir_band = get_band_assets(best_item)
    print(f"Bandas usadas: RED={red_band}, NIR={nir_band}")

    bounds_latlon = None
    if best_item.bbox and len(best_item.bbox) == 4:
        minx, miny, maxx, maxy = best_item.bbox
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        half_w = (maxx - minx) * crop_fraction
        half_h = (maxy - miny) * crop_fraction
        bounds_latlon = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        print(f"Recorte central activo: CROP_FRACTION={crop_fraction}")

    # Stackstac needs a common output CRS. Read it from STAC metadata.
    epsg = get_item_epsg(best_item, (red_band, nir_band))
    if epsg is None:
        raise RuntimeError(
            "No se pudo determinar el EPSG desde STAC (proj:epsg/proj:code)."
        )

    # Copernicus STAC items are used directly (no Planetary Computer signing).
    prepared_item = prefer_https_asset_hrefs(best_item.to_dict())
    sample_href = get_https_asset_href(prepared_item, red_band)
    if sample_href is None:
        raise RuntimeError(f"No se encontro href valido para el asset {red_band}.")
    assert_asset_accessible(sample_href, cdse_access_token)

    red_href = get_https_asset_href(prepared_item, red_band)
    nir_href = get_https_asset_href(prepared_item, nir_band)
    if red_href is None or nir_href is None:
        raise RuntimeError("No se encontraron href HTTPS para las bandas seleccionadas.")

    print("Descargando bandas a local (evita error de Range HTTP del servidor)...")
    temp_paths: list[Path] = []
    try:
        red_local = download_asset_to_tempfile(red_href, cdse_access_token)
        nir_local = download_asset_to_tempfile(nir_href, cdse_access_token)
        temp_paths.extend([red_local, nir_local])

        with rasterio.open(red_local) as red_ds, rasterio.open(nir_local) as nir_ds:
            if red_ds.width != nir_ds.width or red_ds.height != nir_ds.height:
                raise RuntimeError("Las bandas RED y NIR no tienen la misma grilla.")

            col_off, row_off, out_w, out_h = compute_center_window(
                red_ds.width,
                red_ds.height,
                crop_fraction,
            )

            if crop_fraction and crop_fraction > 0:
                print(f"Recorte aplicado en pixeles: {out_w}x{out_h}")

            window = rasterio.windows.Window(
                col_off=col_off,
                row_off=row_off,
                width=out_w,
                height=out_h,
            )

            red_np = red_ds.read(1, window=window).astype("float32")
            nir_np = nir_ds.read(1, window=window).astype("float32")

        # NDVI = (NIR - RED) / (NIR + RED), with zero-division protection
        denom = nir_np + red_np
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi_np = np.where(denom != 0, (nir_np - red_np) / denom, np.nan)
        ndvi_np = np.clip(ndvi_np, -1, 1)
    finally:
        for tmp_path in temp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    valid = np.isfinite(ndvi_np)
    if not np.any(valid):
        raise RuntimeError("No hay valores NDVI validos para mostrar.")

    ndvi_valid = ndvi_np[valid]
    ndvi_min = float(np.min(ndvi_valid))
    ndvi_max = float(np.max(ndvi_valid))
    ndvi_mean = float(np.mean(ndvi_valid))
    p10 = float(np.percentile(ndvi_valid, 10))
    p50 = float(np.percentile(ndvi_valid, 50))
    p90 = float(np.percentile(ndvi_valid, 90))

    print("\nResumen NDVI (terminal):")
    print(f"- Pixeles validos: {ndvi_valid.size}")
    print(f"- Min: {ndvi_min:.3f}")
    print(f"- P10: {p10:.3f}")
    print(f"- Mediana: {p50:.3f}")
    print(f"- P90: {p90:.3f}")
    print(f"- Max: {ndvi_max:.3f}")
    print(f"- Media: {ndvi_mean:.3f}")

    # Save NDVI map as PNG and display it.
    plt.figure(figsize=(10, 8))
    plt.imshow(ndvi_np, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(label="NDVI")
    plt.title(f"NDVI {start_date} a {end_date} - Tile {tile}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    output_png = Path(f"ndvi_{best_item.id}.png")
    plt.savefig(output_png, dpi=180)
    print(f"Imagen NDVI guardada en: {output_png.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
