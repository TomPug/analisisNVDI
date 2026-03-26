import os
import pystac_client
import planetary_computer
import stackstac
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuración de entorno
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['GDAL_HTTP_UNSAFESSL'] = 'YES'

# --- RUTA INTELIGENTE (Funciona en cualquier PC) ---
# os.path.expanduser("~") encuentra la carpeta de usuario (ej: C:\Users\Nombre)
# Luego le añade "Desktop" para llegar al escritorio
escritorio = os.path.join(os.path.expanduser("~"), "Desktop")
ruta_guardado = os.path.join(escritorio, "NDVI_Detallado_Julio_2025.png")

print(f"La imagen se guardará en: {ruta_guardado}")

# 2. Conexión al catálogo
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# 3. Búsqueda inteligente aqui es donde pones la fecha de lo que quieres buscar y esto lo buscara en internet(Julio 2025 + Tile 30STJ)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    datetime="2025-07-01/2025-07-31",
    query={"s2:mgrs_tile": {"eq": "30STJ"}, "eo:cloud_cover": {"lt": 5}}
)

items = list(search.items())

if items:
    item = items[0]
    print(f"Imagen encontrada: {item.id}")

    # 4. Carga de bandas (ajustado a resolución 20 para evitar errores de memoria en otros PCs)
    stack = stackstac.stack(item, assets=["B04", "B08"], epsg=32630, resolution=20)
    
    print("Procesando NDVI... (Un momento)")
    data = stack.compute().astype(float)

    # 5. Cálculo del NDVI
    ndvi = (data.sel(band="B08") - data.sel(band="B04")) / (data.sel(band="B08") + data.sel(band="B04"))
    ndvi_clean = ndvi.squeeze()

    # 6. Gráfico de alta calidad
    plt.figure(figsize=(12, 10))
    img = plt.imshow(ndvi_clean, cmap="RdYlGn", vmin=0, vmax=0.9)
    plt.colorbar(img, label="Índice NDVI")
    plt.title(f"Escaneo NDVI - Julio 2025\nTile 30STJ")
    plt.axis('off')

    # 7. Guardado automático
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print("¡Listo! Mira en tu escritorio, ahí está la imagen.")
    
    plt.show()
else:
    print("No se encontraron datos.")