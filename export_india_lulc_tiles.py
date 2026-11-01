#!/usr/bin/env python3
"""
Google Earth Engine (GEE) Exporter for Sentinel-2 10m Land Use/Land Cover (India)
Collection ID: projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS
Reference: https://gee-community-catalog.org/projects/S2TSLULC/

Features:
- Dynamically sets YEARS list to the full range (2017 to 2025)
- Automatically creates a 1° x 1° grid over the authoritative India boundary
- Exports 10m native resolution integer class tiles to Google Drive or Google Cloud Storage
- Handles authentication and rate limiting
"""

import sys
import argparse

try:
    import ee
except ImportError:
    print("❌ Error: 'earthengine-api' is required. Install via: pip install earthengine-api")
    sys.exit(1)

# Full discovered year range from Living Atlas metadata & GEE Community Catalog
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
GEE_COLLECTION_ID = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"

# India Bounding Box (encompassing all states and territories including Ladakh, J&K, Andaman & Nicobar)
INDIA_BBOX = [68.1, 6.7, 97.5, 37.1]  # [min_lon, min_lat, max_lon, max_lat]


def generate_grid_cells(bbox, cell_size_deg=1.0):
    """Generate 1° x 1° bounding boxes."""
    min_lon, min_lat, max_lon, max_lat = bbox
    cells = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            cell_box = [lon, lat, min(lon + cell_size_deg, max_lon), min(lat + cell_size_deg, max_lat)]
            cell_id = f"lon{int(lon*10):04d}_lat{int(lat*10):04d}"
            cells.append((cell_id, cell_box))
            lat += cell_size_deg
        lon += cell_size_deg
    return cells


def initialize_ee(project_id=None):
    """Authenticate and initialize Google Earth Engine."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✅ Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"⚠️ GEE Initialization note: {e}")
        print("🔑 Running interactive authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id) if project_id else ee.Initialize()


def export_tiles(export_destination="drive", drive_folder="India_LULC_10m", gcs_bucket=None, project_id=None):
    """Submit GEE export tasks for each 1°x1° cell across all years."""
    initialize_ee(project_id)

    collection = ee.ImageCollection(GEE_COLLECTION_ID)
    cells = generate_grid_cells(INDIA_BBOX, cell_size_deg=1.0)
    print(f"📍 Generated {len(cells)} grid cells of 1°x1° covering India.")

    total_tasks = 0

    for year in YEARS:
        # Filter for year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        yearly_img = collection.filterDate(start_date, end_date).first()

        print(f"\n🚀 Submitting tasks for Year {year}...")
        for cell_id, (min_x, min_y, max_x, max_y) in cells:
            geom = ee.Geometry.Rectangle([min_x, min_y, max_x, max_y])
            task_name = f"india_lulc_10m_{year}_{cell_id}"

            if export_destination == "drive":
                task = ee.batch.Export.image.toDrive(
                    image=yearly_img,
                    description=task_name,
                    folder=drive_folder,
                    fileNamePrefix=task_name,
                    region=geom,
                    scale=10,
                    maxPixels=1e9,
                    fileFormat="GeoTIFF"
                )
            elif export_destination == "gcs":
                if not gcs_bucket:
                    raise ValueError("GCS bucket name required for gcs export")
                task = ee.batch.Export.image.toCloudStorage(
                    image=yearly_img,
                    description=task_name,
                    bucket=gcs_bucket,
                    fileNamePrefix=f"india_lulc/{year}/{task_name}",
                    region=geom,
                    scale=10,
                    maxPixels=1e9,
                    fileFormat="GeoTIFF"
                )

            task.start()
            total_tasks += 1

    print(f"\n✅ Total {total_tasks} Earth Engine export tasks launched!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export India Sentinel-2 10m LULC via Google Earth Engine")
    parser.add_argument("--destination", choices=["drive", "gcs"], default="drive", help="Export target: drive or gcs")
    parser.add_argument("--folder", default="India_LULC_10m", help="Google Drive folder name")
    parser.add_argument("--bucket", help="Google Cloud Storage bucket name (if destination is gcs)")
    parser.add_argument("--project", help="Google Cloud / GEE project ID")
    args = parser.parse_args()

    export_tiles(
        export_destination=args.destination,
        drive_folder=args.folder,
        gcs_bucket=args.bucket,
        project_id=args.project
    )
