import geopandas as gpd
import pandas as pd
import os

candidates = [
    "G:/Unidades compartidas/Proy_ATTEL/DI_Tomas/RIOJA-FENOLOGIA/data/raw/Rioja_prov.shp",
    "G:/Unidades compartidas/Proy_ATTEL/DI_Tomas/RIOJA-FENOLOGIA/data/raw/RIOJA.geojson",
    "G:/Unidades compartidas/Proy_ATTEL/DI_Tomas/RIOJA-FENOLOGIA/data/processed/shapes/ROI.geojson",
    "G:/Unidades compartidas/Proy_ATTEL/DI_Tomas/RIOJA-FENOLOGIA/data/processed/shapes/ROI_wgs84.geojson",
    "G:/Unidades compartidas/Proy_ATTEL/DI_Tomas/RIOJA-FENOLOGIA/data/processed/shapes/ROI.shp",
    "G:/Unidades compartidas/Proy_ATTEL/DI_Tomas/RIOJA-FENOLOGIA/data/processed/shapes/FF/MUP_LaRioja.gpkg"
]

results = []
for path in candidates:
    if not os.path.exists(path):
        continue
    try:
        gdf = gpd.read_file(path)
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        
        # Approximate area in km2 using a pseudo-mercator for calculation if needed, 
        # or just use a basic estimate. Let's use cea for area.
        gdf_ea = gdf.to_crs({'proj':'cea'})
        area_km2 = gdf_ea.geometry.area.sum() / 1e6
        
        results.append({
            "path": path,
            "geom_types": gdf.geometry.type.unique().tolist(),
            "count": len(gdf),
            "bounds": gdf_wgs84.total_bounds.tolist(),
            "approx_area_km2": area_km2
        })
    except Exception as e:
        print(f"Error reading {path}: {e}")

for res in results:
    print("-" * 20)
    print(f"Path: {res['path']}")
    print(f"Geom Types: {res['geom_types']}")
    print(f"Entities: {res['count']}")
    print(f"Bounds (WGS84): {res['bounds']}")
    print(f"Approx Area (km2): {res['approx_area_km2']:.2f}")

