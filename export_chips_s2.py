#!/usr/bin/env python3
"""
export_chips_s2.py

Google Earth Engine (GEE) export pipeline for nationwide 5 km x 5 km Sentinel-2 chips
matching the 1,500 Esri LULC label chips in ./labels/.

Project Context:
- Chips list: chips.csv (1,500 chips across India, stratified into R, A, B, C, D)
- Corresponding labels: labels/<chip_id>.tif (9 annual bands, 500x500 px, 10 m, EPSG UTM)
- Sensor: Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED)
- Output format: One 7-band GeoTIFF per chip per year:
    Bands: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'n_obs'] (uint16)
    Dimensions: 500 x 500 pixels (10 m native resolution)
    Grid: Exact UTM CRS and affine transform matching labels/<chip_id>.tif
    Naming: <chip_id>_<year>.tif (e.g. 43R_r130_c111_2024.tif)

Features:
- Documented cloud & shadow filtering using Sentinel-2 QA60 and SCL masks.
- Explicit observation count band (n_obs) per pixel; 0 indicates missing/unobserved.
- Pixel-perfect alignment: enforces crsTransform [10, 0, x0, 0, -10, y1] and dimensions [500, 500].
- Safe batching support: --pilot-5, --chip-ids, --start, --end, --limit.
- Dry-run mode (--dry-run) to inspect plan, task counts, and parameters without submitting.
- Duplicate prevention: checks active/completed GEE task lists before submission.
- Built-in verification utility (--verify) to audit downloaded chips against labels.

Usage Examples:
    # 1. Inspect dry-run plan for the 5-chip pilot test (no tasks submitted):
    python3 export_chips_s2.py --pilot-5 --dry-run

    # 2. Submit the 5-chip pilot test to dedicated test folder:
    python3 export_chips_s2.py --pilot-5 --folder india_s2_chips_test --project YOUR_PROJECT_ID

    # 3. Export specific chips:
    python3 export_chips_s2.py --chip-ids 43R_r130_c111 44R_r131_c81 --folder india_s2_chips_test

    # 4. Verify a downloaded chip against its matching label:
    python3 export_chips_s2.py --verify-file downloaded/43R_r130_c111_2024.tif
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Default parameters
BASE_DIR = Path("/Users/ashi/ResearchPaper/india_lulc")
CHIPS_FILE = BASE_DIR / "chips.csv"
LABELS_DIR = BASE_DIR / "labels"
ALL_YEARS = list(range(2017, 2026))  # 2017..2025 (9 years)
CHIP_PX = 500  # 500 px @ 10 m = 5,000 m (5 km)
SPECTRAL_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
OUTPUT_BANDS = SPECTRAL_BANDS + ["n_obs"]

# Five representative chips chosen reproducibly (first chip of each stratum R, A, B, C, D)
PILOT_5_CHIP_IDS = [
    "43R_r130_c111",  # Stratum R: Random Unbiased (Rajasthan/MP, UTM 43N)
    "46R_r101_c74",   # Stratum A: Rapid Urban Growth (Assam, UTM 46N)
    "44R_r131_c81",   # Stratum B: Dense Urban Core (Uttar Pradesh, UTM 44N)
    "44Q_r113_c70",   # Stratum C: Low Built / Semi-Urban (Chhattisgarh, UTM 44N)
    "43R_r81_c35"     # Stratum D: Empty Baseline / Desert (Thar Desert, UTM 43N)
]


def load_selected_chips(args):
    """Loads and filters chip rows from chips.csv based on CLI arguments."""
    if not os.path.exists(args.chips):
        raise FileNotFoundError(f"Chips file not found: {args.chips}")
    df = pd.read_csv(args.chips)

    if args.pilot_5:
        selected = df[df["chip_id"].isin(PILOT_5_CHIP_IDS)].copy()
        # Maintain order
        selected["sort_order"] = selected["chip_id"].map({cid: i for i, cid in enumerate(PILOT_5_CHIP_IDS)})
        selected = selected.sort_values("sort_order").drop(columns=["sort_order"])
        return selected

    if args.chip_ids:
        selected = df[df["chip_id"].isin(args.chip_ids)].copy()
        missing = set(args.chip_ids) - set(selected["chip_id"])
        if missing:
            print(f"Warning: The following requested chip IDs were not found in chips.csv: {missing}")
        return selected

    start = args.start or 0
    end = args.end or len(df)
    selected = df.iloc[start:end].copy()

    if args.limit:
        selected = selected.head(args.limit)

    return selected


def initialize_earth_engine(project_id=None):
    """Initializes Google Earth Engine without hardcoding credentials."""
    try:
        import ee
    except ImportError:
        print("ERROR: 'earthengine-api' is required. Install via: pip install earthengine-api")
        sys.exit(1)

    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✓ Google Earth Engine initialized successfully.")
        return ee
    except Exception as e:
        print(f"\n⚠️  Earth Engine Initialization Note: {e}")
        print("To authenticate, run: earthengine authenticate")
        print("Or provide a Google Cloud Project ID via: --project YOUR_PROJECT_ID")
        raise e


def mask_s2_sr_clouds(image, ee):
    """
    Applies cloud, shadow, and cirrus masking to Sentinel-2 Level-2A Surface Reflectance.
    
    1. QA60 Bitmask:
       - Bit 10: Opaque clouds (0 = clear, 1 = cloud)
       - Bit 11: Cirrus clouds (0 = clear, 1 = cirrus)
    2. Scene Classification Layer (SCL):
       - Masks values: 3 (Cloud shadow), 8 (Cloud medium prob), 9 (Cloud high prob), 10 (Thin cirrus)
    """
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))

    scl = image.select("SCL")
    scl_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))

    valid_mask = qa_mask.And(scl_mask)
    return image.updateMask(valid_mask)


def build_annual_composite(year, aoi_geom, ee):
    """
    Builds a single-year 7-band composite for a given year and region of interest.
    
    Bands:
      1. B2: Blue (490 nm, 10 m)
      2. B3: Green (560 nm, 10 m)
      3. B4: Red (665 nm, 10 m)
      4. B8: NIR (842 nm, 10 m)
      5. B11: SWIR 1 (1610 nm, 20 m resampled to 10 m)
      6. B12: SWIR 2 (2190 nm, 20 m resampled to 10 m)
      7. n_obs: Count of cloud-free surface reflectance observations
    
    Returns:
      ee.Image with 7 bands, cast to uint16, with unmask(0) for missing observations.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    # Harmonized Level-2A Surface Reflectance Collection
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi_geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
        .map(lambda img: mask_s2_sr_clouds(img, ee))
    )

    # Observation count band (valid cloud-free observations)
    n_obs = s2_col.select("B2").count().rename("n_obs")

    # Annual median composite of spectral bands
    spectral_median = s2_col.select(SPECTRAL_BANDS).median()

    # Combine spectral bands with observation count
    composite = spectral_median.addBands(n_obs)

    # Explicitly unmask to 0 so unobserved pixels are 0, cast to uint16
    return composite.unmask(0).toUint16()


def print_plan_summary(chips_df, years, folder, is_stack=False):
    """Prints a clear, formatted summary of the planned export batch."""
    n_chips = len(chips_df)
    n_years = len(years)
    total_tasks = n_chips if is_stack else n_chips * n_years
    est_mb_per_chip_yr = 3.2  # ~3.2 MB per 500x500 7-band uint16 GeoTIFF
    total_est_mb = n_chips * n_years * est_mb_per_chip_yr

    print("\n" + "=" * 85)
    print("SENTINEL-2 CHIP EXPORT PLAN SUMMARY")
    print("=" * 85)
    print(f"  • Destination Google Drive Folder : {folder}")
    print(f"  • Export Architecture             : {'Multi-Year Stack (63 bands)' if is_stack else 'Annual 7-Band GeoTIFFs (Recommended)'}")
    print(f"  • Selected Chips Count            : {n_chips}")
    print(f"  • Target Years ({len(years)})             : {years[0]} to {years[-1]} ({years})")
    print(f"  • Bands per Annual Image (7)      : {OUTPUT_BANDS}")
    print(f"  • Data Type                       : uint16 (Surface Reflectance scale 0-10,000, 0 = NoData)")
    print(f"  • Target Chip Dimensions          : 500 x 500 pixels (10 m native resolution = 5 km x 5 km)")
    print(f"  • Total GEE Tasks to Submit       : {total_tasks:,}")
    print(f"  • Estimated Total Drive Storage   : ~{total_est_mb:.1f} MB (~{total_est_mb / 1024:.2f} GB)")
    print("\nSelected Chips Specification Table:")
    print("| Chip ID | Stratum | Split | Native Tile | UTM EPSG | Origin (x0, y1) | Centroid (Lon, Lat) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, r in chips_df.iterrows():
        print(f"| {r['chip_id']} | {r['stratum']} | {r['split']} | {r['tile']} | EPSG:{int(r['epsg'])} | ({r['x0']:.0f}, {r['y1']:.0f}) | ({r['lon']:.4f}°E, {r['lat']:.4f}°N) |")
    print("=" * 85 + "\n")


def submit_export_tasks(chips_df, years, folder, project_id=None, dry_run=False, is_stack=False):
    """Generates and submits Earth Engine Export tasks for all selected chips and years."""
    print_plan_summary(chips_df, years, folder, is_stack)

    if dry_run:
        print("DRY-RUN MODE: No Earth Engine tasks were submitted. Parameter validation passed.\n")
        return []

    ee = initialize_earth_engine(project_id)

    # Query active/pending tasks to avoid duplicate submissions
    try:
        active_task_list = ee.data.getTaskList()
        existing_task_descs = {
            t["description"]: t["state"]
            for t in active_task_list
            if t.get("state") in ["READY", "RUNNING", "COMPLETED"]
        }
    except Exception as e:
        print(f"Note: Could not retrieve existing task list ({e}). Proceeding carefully.")
        existing_task_descs = {}

    submitted_tasks = []
    skipped_tasks = []

    for _, r in chips_df.iterrows():
        cid = r["chip_id"]
        epsg = int(r["epsg"])
        x0 = float(r["x0"])
        y1 = float(r["y1"])

        # Native UTM bounding box: [x0, y1 - 5000, x0 + 5000, y1]
        x_min, x_max = x0, x0 + (CHIP_PX * 10.0)
        y_min, y_max = y1 - (CHIP_PX * 10.0), y1

        aoi_geom = ee.Geometry.Rectangle(
            coords=[x_min, y_min, x_max, y_max],
            proj=f"EPSG:{epsg}",
            geodesic=False
        )

        # Affine transform for 10 m north-up UTM grid: [10, 0, x0, 0, -10, y1]
        crs_transform = [10.0, 0.0, x0, 0.0, -10.0, y1]

        if is_stack:
            # Multi-year stacked export (7 bands x N years)
            task_desc = f"s2_{cid}_stack"
            file_prefix = f"{cid}_s2_stack"

            if task_desc in existing_task_descs:
                skipped_tasks.append(task_desc)
                continue

            yearly_images = []
            for y in years:
                img_y = build_annual_composite(y, aoi_geom, ee)
                renamed_bands = [f"{y}_{b}" for b in OUTPUT_BANDS]
                yearly_images.append(img_y.rename(renamed_bands))

            stacked_img = ee.Image.cat(yearly_images)

            task = ee.batch.Export.image.toDrive(
                image=stacked_img,
                description=task_desc,
                folder=folder,
                fileNamePrefix=file_prefix,
                region=aoi_geom,
                dimensions=[CHIP_PX, CHIP_PX],
                crs=f"EPSG:{epsg}",
                crsTransform=crs_transform,
                maxPixels=1e9,
                fileFormat="GeoTIFF"
            )
            task.start()
            submitted_tasks.append((cid, "stack", task.id))
            print(f"  ✓ Submitted stack task: {task_desc} (Task ID: {task.id})")
        else:
            # Default recommended: One annual 7-band GeoTIFF per year
            for y in years:
                task_desc = f"s2_{cid}_{y}"
                file_prefix = f"{cid}_{y}"

                if task_desc in existing_task_descs:
                    skipped_tasks.append(task_desc)
                    continue

                annual_img = build_annual_composite(y, aoi_geom, ee)

                task = ee.batch.Export.image.toDrive(
                    image=annual_img,
                    description=task_desc,
                    folder=folder,
                    fileNamePrefix=file_prefix,
                    region=aoi_geom,
                    dimensions=[CHIP_PX, CHIP_PX],
                    crs=f"EPSG:{epsg}",
                    crsTransform=crs_transform,
                    maxPixels=1e9,
                    fileFormat="GeoTIFF"
                )
                task.start()
                submitted_tasks.append((cid, y, task.id))
                print(f"  ✓ Submitted annual task: {task_desc} (Task ID: {task.id})")

    print(f"\nTask Submission Summary:")
    print(f"  • Tasks successfully submitted : {len(submitted_tasks)}")
    print(f"  • Tasks skipped (already active): {len(skipped_tasks)}")
    return submitted_tasks


def verify_downloaded_chip(chip_file_path):
    """
    Verifies a downloaded Sentinel-2 chip against its corresponding label chip:
    - Dimensions (500x500)
    - Native UTM CRS and exact affine transform
    - Band count, data types, and band descriptions
    - Absence of pixel grid misalignment
    """
    try:
        import rasterio
    except ImportError:
        print("rasterio is required for verification. Install via: pip install rasterio")
        return False

    p = Path(chip_file_path)
    if not p.exists():
        print(f"Error: File does not exist: {p}")
        return False

    # Extract chip_id from filename (handles both <chip_id>_<year>.tif and <chip_id>_s2_stack.tif)
    parts = p.stem.split("_")
    # Chip ID format is typically TILE_rROW_cCOL (e.g. 43R_r130_c111)
    if len(parts) >= 3 and parts[1].startswith("r") and parts[2].startswith("c"):
        chip_id = "_".join(parts[:3])
    else:
        chip_id = p.stem

    label_path = LABELS_DIR / f"{chip_id}.tif"
    print(f"\n=== VERIFYING SENTINEL-2 CHIP: {p.name} ===")
    print(f"  • Matched Label Chip: {label_path.name}")

    if not label_path.exists():
        print(f"  ⚠️ Warning: Matching label file not found at {label_path}")
        return False

    with rasterio.open(p) as s2_src, rasterio.open(label_path) as lbl_src:
        print(f"  • Dimensions check   : S2={s2_src.shape} vs Label={lbl_src.shape} -> {'PASSED' if s2_src.shape == lbl_src.shape else 'FAILED'}")
        print(f"  • CRS check          : S2={s2_src.crs} vs Label={lbl_src.crs} -> {'PASSED' if s2_src.crs == lbl_src.crs else 'FAILED'}")
        print(f"  • Resolution check   : S2={s2_src.res} vs Label={lbl_src.res} -> {'PASSED' if s2_src.res == lbl_src.res else 'FAILED'}")
        print(f"  • Transform check    : S2={s2_src.transform} vs Label={lbl_src.transform} -> {'PASSED' if s2_src.transform == lbl_src.transform else 'FAILED'}")
        print(f"  • Bounds check       : S2={s2_src.bounds} vs Label={lbl_src.bounds} -> {'PASSED' if s2_src.bounds == lbl_src.bounds else 'FAILED'}")
        print(f"  • S2 Band Count      : {s2_src.count}")
        print(f"  • S2 Data Types      : {s2_src.dtypes}")
        print(f"  • S2 Descriptions    : {s2_src.descriptions}")

        # Quality check on n_obs band
        if s2_src.count >= 7:
            n_obs = s2_src.read(s2_src.count)
            zero_obs = int((n_obs == 0).sum())
            mean_obs = float(n_obs.mean())
            print(f"  • n_obs Band Stats   : Mean={mean_obs:.1f}, Zero-obs Pixels={zero_obs:,} ({zero_obs/n_obs.size*100:.2f}%)")

        passed = (
            s2_src.shape == lbl_src.shape
            and s2_src.crs == lbl_src.crs
            and s2_src.res == lbl_src.res
            and s2_src.transform == lbl_src.transform
        )
        print(f"  • OVERALL GRID ALIGNMENT: {'100% PERFECT MATCH' if passed else 'ALIGNMENT ERROR DETECTED'}\n")
        return passed


def main():
    parser = argparse.ArgumentParser(
        description="Google Earth Engine Sentinel-2 chip exporter for India urban research."
    )
    parser.add_argument("--chips", default=str(CHIPS_FILE), help="Path to chips.csv")
    parser.add_argument("--pilot-5", action="store_true", help="Select the 5 reproducible pilot chips (R, A, B, C, D)")
    parser.add_argument("--chip-ids", nargs="+", help="Specific chip IDs to export")
    parser.add_argument("--start", type=int, default=None, help="Start row index in chips.csv")
    parser.add_argument("--end", type=int, default=None, help="End row index in chips.csv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chips to export")
    parser.add_argument("--years", nargs="+", type=int, default=ALL_YEARS, help="Years to export (default: 2017..2025)")
    parser.add_argument("--folder", default="india_s2_chips_test", help="Google Drive destination folder")
    parser.add_argument("--project", default=None, help="Google Cloud / Earth Engine project ID")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and validate parameters without submitting tasks")
    parser.add_argument("--stack", action="store_true", help="Export multi-year stack instead of annual files")
    parser.add_argument("--verify-file", help="Audit a downloaded Sentinel-2 GeoTIFF against its label file")

    args = parser.parse_args()

    if args.verify_file:
        verify_downloaded_chip(args.verify_file)
        return

    chips_to_export = load_selected_chips(args)
    if len(chips_to_export) == 0:
        print("No chips selected. Check your filters or use --pilot-5.")
        return

    submit_export_tasks(
        chips_df=chips_to_export,
        years=args.years,
        folder=args.folder,
        project_id=args.project,
        dry_run=args.dry_run,
        is_stack=args.stack
    )


if __name__ == "__main__":
    main()
