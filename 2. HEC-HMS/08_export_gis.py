# -*- coding: utf-8 -*-
"""
Step 8 (optional) - Export the HEC-HMS sub-basin and reach geometry to GIS
files (GeoPackage, EPSG:32646) for mapping (Fig. 1, Fig. 3).

Requires geopandas (pip install geopandas).

Input  : HMS_Template/Tlawng_SMA_FN.sqlite   (written by HEC-HMS GIS tools)
Output : GIS/Tlawng_HMS_subbasins.gpkg, GIS/Tlawng_HMS_reaches.gpkg

Run    : python 08_export_gis.py
"""

from common import GIS_DIR, TEMPLATE_DIR, BASIN_FILE

SQLITE = TEMPLATE_DIR / BASIN_FILE.replace(".basin", ".sqlite")


def main():
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed - skipping GIS export (pip install geopandas)")
        return
    GIS_DIR.mkdir(exist_ok=True)
    sub = gpd.read_file(SQLITE, layer="subbasin2d")
    reach = gpd.read_file(SQLITE, layer="reach2d")
    sub["area_km2_geom"] = (sub.geometry.area / 1e6).round(3)
    sub[["name", "area_km2_geom", "geometry"]].to_file(GIS_DIR / "Tlawng_HMS_subbasins.gpkg", driver="GPKG")
    reach[["name", "length", "slope", "geometry"]].to_file(GIS_DIR / "Tlawng_HMS_reaches.gpkg", driver="GPKG")
    print(f"{len(sub)} sub-basins, {len(reach)} reaches -> {GIS_DIR.name}/ (CRS {sub.crs})")


if __name__ == "__main__":
    main()
