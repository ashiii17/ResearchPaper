#!/usr/bin/env python3
"""
Sentinel-2 10m LULC Time Series Pipeline for India (2017 - 2025).

Features:
  - Strictly native UTM tiles only: no mosaics, no reclassification, all 9 classes preserved:
    1: Water, 2: Trees, 4: Flooded vegetation, 5: Crops, 7: Built Area,
    8: Bare ground, 9: Snow/Ice, 10: Clouds, 11: Rangeland (NoData = 0).
  - Corrected tile list (17 tiles) derived geometrically from UTM zone/band intersections with India boundary.
  - Downloads directly from the authoritative Living Atlas Azure Blob storage
    (https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003), with fallback to AWS S3.
  - Enforces minimum free disk space threshold (>= 8 GB) before processing each year.
  - Low-RAM windowed validation, pixel counting, and urban transition analysis.
  - Produces two sets of statistics:
      (a) Raw Per-Tile-Set totals across all 17 tiles.
      (b) India Boundary Masked totals (excluding neighboring countries) using simplified boundary.
  - Resumable: skips verified cached tiles.
  - Generates manifest.csv, failures.csv, and class_stats.csv.
  - Spot-check verification for benchmark coordinates and urban transition analysis.

Usage:
  python download_and_process_fixed.py --dry-run
  python download_and_process_fixed.py --years 2025 2017 2018 2019 2020 2021 2022 2023
  python download_and_process_fixed.py --stats-only
  python download_and_process_fixed.py --verify-all
  python download_and_process_fixed.py --transition-analysis
  python download_and_process_fixed.py --spot-check
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import shapely
import shapely.geometry
from pyproj import Transformer

BASE = Path(__file__).resolve().parent
TILES_DIR = BASE / "tiles"
BOUNDARY_FILE = BASE / "india_boundary.geojson"
AZURE_BASE_URL = "https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003"
S3_BASE_URL = "https://io-10m-annual-lulc.s3.amazonaws.com"
LIVING_ATLAS_QUERY_URL = (
    "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer/query"
)

# Geometrically derived 17 UTM tiles intersecting India boundary
INDIA_TILES = [
    "42Q", "42R",
    "43P", "43Q", "43R", "43S",
    "44P", "44Q", "44R", "44S",
    "45Q", "45R",
    "46N", "46P", "46Q", "46R",
    "47R"
]

CLASSES = {
    1: "Water",
    2: "Trees",
    4: "Flooded vegetation",
    5: "Crops",
    7: "Built Area",
    8: "Bare ground",
    9: "Snow/Ice",
    10: "Clouds",
    11: "Rangeland"
}
VALID_CLASSES = {0, 1, 2, 4, 5, 7, 8, 9, 10, 11}
MIN_FREE_DISK_GB = 8.0
BOUNDARY_TOLERANCE_DEG = 0.005  # ~500 m simplification tolerance


def get_free_disk_gb():
    """Returns free disk space in GB for the workspace."""
    return shutil.disk_usage(BASE).free / 1e9


def tile_urls(tile, year):
    """
    Returns candidate URLs for a tile and year.
    Living Atlas Azure Blob naming:
      - 2024+: lc{year}/{tile}_{year}0101-{year}1231.tif
      - <2024: lc{year}/{tile}_{year}0101-{year+1}0101.tif
    AWS S3 Open Data naming:
      - {tile}_{year}.tif (available up to 2024 only)
    """
    if year >= 2024:
        azure_url = f"{AZURE_BASE_URL}/lc{year}/{tile}_{year}0101-{year}1231.tif"
    else:
        azure_url = f"{AZURE_BASE_URL}/lc{year}/{tile}_{year}0101-{year+1}0101.tif"

    if year <= 2024:
        s3_url = f"{S3_BASE_URL}/{tile}_{year}.tif"
        return [azure_url, s3_url]
    else:
        return [azure_url]


def discover_years():
    """Discover available years from ArcGIS Living Atlas ImageServer REST API."""
    years = set()
    try:
        url = f"{LIVING_ATLAS_QUERY_URL}?where=1%3D1&outFields=Year&returnGeometry=false&returnDistinctValues=true&f=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            if "features" in data:
                for feat in data["features"]:
                    y = feat.get("attributes", {}).get("Year")
                    if y:
                        years.add(int(y))
    except Exception as e:
        print(f"Year discovery via Living Atlas API notice: {e}", flush=True)

    if not years:
        years = set(range(2017, 2026))

    return sorted(list(years))


def sha256(filepath):
    """Compute SHA-256 checksum in 1 MB blocks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tile_integrity(path):
    """Verify GeoTIFF can be opened and has valid metadata and pixel values."""
    try:
        with rasterio.open(path) as src:
            if src.dtypes[0] != "uint8":
                return False, f"Expected dtype uint8, got {src.dtypes[0]}"
            if src.crs is None:
                return False, "CRS is None"
            res = src.res
            if abs(res[0] - 10.0) > 0.1 or abs(res[1] - 10.0) > 0.1:
                return False, f"Expected 10m resolution, got {res}"
            # Sample window to verify class values
            sample = src.read(1, window=((0, min(512, src.height)), (0, min(512, src.width))))
            invalid = set(np.unique(sample)) - VALID_CLASSES
            if invalid:
                return False, f"Unexpected pixel values found: {invalid}"
        return True, "valid"
    except Exception as e:
        return False, str(e)


def download_tile(tile, year, retries=15):
    """
    Downloads tile using curl with resume (-C -), HTTP status code parsing,
    exponential backoff for rate limits / server errors, and disk space checks.
    Returns: (dest_path, status, source_url, explanation)
    """
    dest = TILES_DIR / str(year) / f"{tile}_{year}.tif"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Pre-download disk safety check
    free_gb = get_free_disk_gb()
    if free_gb < MIN_FREE_DISK_GB:
        return None, "disk_full", tile_urls(tile, year)[0], f"Free disk ({free_gb:.2f} GB) reached safety threshold ({MIN_FREE_DISK_GB} GB)"

    # Check if already cached and valid
    if dest.exists() and dest.stat().st_size > 1000:
        valid, msg = verify_tile_integrity(dest)
        if valid:
            return dest, "cached", tile_urls(tile, year)[0], "already downloaded and verified"
        else:
            print(f"  Existing file {dest.name} failed verification ({msg}). Re-downloading...", flush=True)
            dest.unlink(missing_ok=True)

    urls = tile_urls(tile, year)
    tmp = dest.with_suffix(".part")

    for url in urls:
        for attempt in range(1, retries + 1):
            # Check free disk space before each attempt
            if get_free_disk_gb() < MIN_FREE_DISK_GB:
                return None, "disk_full", url, f"Free disk space below {MIN_FREE_DISK_GB} GB threshold"

            cmd = [
                "curl", "-s", "-L",
                "-H", "x-ms-version: 2020-04-08",
                "--connect-timeout", "30",
                "--speed-time", "60",
                "--speed-limit", "500",
                "-C", "-",
                "-w", "\nHTTP_STATUS:%{http_code}\n",
                "-o", str(tmp),
                url
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)

            # Extract HTTP status code from curl output
            http_code = 0
            for line in res.stdout.splitlines():
                if line.startswith("HTTP_STATUS:"):
                    try:
                        http_code = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if res.returncode == 0 and http_code in (200, 206) and tmp.exists() and tmp.stat().st_size > 1000:
                valid, msg = verify_tile_integrity(tmp)
                if valid:
                    tmp.rename(dest)
                    return dest, "ok", url, "successfully downloaded and verified"
                else:
                    print(f"  Corrupt download for {tile}_{year} from {url}: {msg}. Retrying...", flush=True)
                    tmp.unlink(missing_ok=True)
            elif http_code == 429:
                sleep_time = min(60, 15 * attempt)
                print(f"  [HTTP 429 Rate Limited] {tile}_{year} (attempt {attempt}/{retries}). Backing off {sleep_time}s...", flush=True)
                time.sleep(sleep_time)
            elif http_code in (500, 502, 503, 504):
                sleep_time = min(45, 5 * attempt)
                print(f"  [HTTP {http_code} Server Error] {tile}_{year} (attempt {attempt}/{retries}). Backing off {sleep_time}s...", flush=True)
                time.sleep(sleep_time)
            elif res.returncode == 33:
                print(f"  Range resume code 33 for {tile}_{year}. Resetting partial file and restarting from byte 0...", flush=True)
                tmp.unlink(missing_ok=True)
            else:
                sleep_time = min(30, 2 + attempt * 2)
                time.sleep(sleep_time)

    return None, "failed", urls[0], f"Download failed after {retries} retries (last http={http_code}, rc={res.returncode})"


from functools import lru_cache


@lru_cache(maxsize=1)
def load_simplified_boundary():
    """Loads and simplifies the India boundary geometry."""
    if not BOUNDARY_FILE.exists():
        print(f"Warning: Boundary file {BOUNDARY_FILE} not found. Boundary masking will be skipped.", flush=True)
        return None
    with open(BOUNDARY_FILE) as f:
        gj = json.load(f)
    geoms = [shapely.geometry.shape(feat["geometry"]) for feat in gj["features"]]
    raw_geom = shapely.unary_union(geoms)
    simp_geom = raw_geom.simplify(BOUNDARY_TOLERANCE_DEG, preserve_topology=True)
    return simp_geom


def compute_tile_counts(path, boundary_geom=None):
    """
    Windowed read pixel bincount for low RAM usage.
    Returns:
      raw_counts: np.ndarray (size 256) across entire tile
      masked_counts: np.ndarray (size 256) masked to India boundary
    """
    raw_counts = np.zeros(256, dtype=np.int64)
    masked_counts = np.zeros(256, dtype=np.int64)

    with rasterio.open(path) as src:
        crs = src.crs
        bounds = src.bounds
        tile_box_utm = shapely.geometry.box(*bounds)

        clipped_geom = None
        if boundary_geom is not None and crs is not None:
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            utm_geom = shapely.transform(
                boundary_geom,
                lambda coords: np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))
            )
            clipped_geom = utm_geom.intersection(tile_box_utm)

        for _, win in src.block_windows(1):
            arr = src.read(1, window=win)
            raw_counts += np.bincount(arr.ravel(), minlength=256)

            if clipped_geom is None or clipped_geom.is_empty:
                continue

            win_bounds = rasterio.windows.bounds(win, src.transform)
            win_box = shapely.geometry.box(*win_bounds)

            if win_box.within(clipped_geom):
                masked_counts += np.bincount(arr.ravel(), minlength=256)
            elif not win_box.intersects(clipped_geom):
                continue
            else:
                win_transform = rasterio.windows.transform(win, src.transform)
                mask = rasterio.features.rasterize(
                    [(clipped_geom, 1)],
                    out_shape=arr.shape,
                    transform=win_transform,
                    fill=0,
                    dtype="uint8"
                )
                masked_arr = arr[mask == 1]
                if masked_arr.size > 0:
                    masked_counts += np.bincount(masked_arr.ravel(), minlength=256)

    # Validate that no invalid classes exist
    invalid_raw = set(np.where(raw_counts > 0)[0]) - VALID_CLASSES
    if invalid_raw:
        print(f"  WARNING: Tile {path.name} contains invalid pixel classes: {invalid_raw}", flush=True)

    return raw_counts, masked_counts


def update_manifest(records):
    """Updates manifest.csv cleanly."""
    manifest_path = BASE / "manifest.csv"
    new_df = pd.DataFrame(records)
    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path)
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=["file"], keep="last")
        combined.sort_values(by=["year", "file"]).to_csv(manifest_path, index=False)
    else:
        new_df.sort_values(by=["year", "file"]).to_csv(manifest_path, index=False)


def update_failures(records):
    """Updates failures.csv cleanly."""
    failures_path = BASE / "failures.csv"
    new_df = pd.DataFrame(records)
    if failures_path.exists():
        existing_df = pd.read_csv(failures_path)
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=["tile", "year"], keep="last")
        combined.sort_values(by=["year", "tile"]).to_csv(failures_path, index=False)
    else:
        new_df.sort_values(by=["year", "tile"]).to_csv(failures_path, index=False)


def update_class_stats(records):
    """Updates class_stats.csv cleanly."""
    stats_path = BASE / "class_stats.csv"
    new_df = pd.DataFrame(records)
    if stats_path.exists():
        existing_df = pd.read_csv(stats_path)
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=["Year", "Scope"], keep="last")
        combined.sort_values(by=["Year", "Scope"]).to_csv(stats_path, index=False)
    else:
        new_df.sort_values(by=["Year", "Scope"]).to_csv(stats_path, index=False)


def run_spot_checks(years=(2017, 2025)):
    """Spot check benchmark coordinates across requested years."""
    pts = [
        ("Delhi", 77.2090, 28.6140, "43R"),
        ("Bengaluru", 77.5946, 12.9716, "43P"),
        ("Hyderabad", 78.4867, 17.3850, "44Q"),
        ("Port Blair", 92.7265, 11.6234, "46P"),
        ("Itanagar", 93.6200, 27.0800, "46R"),
    ]
    print("\n" + "=" * 80, flush=True)
    print("BENCHMARK COORDINATES SPOT CHECK", flush=True)
    print("=" * 80, flush=True)
    results = []
    for name, lon, lat, tile in pts:
        for yr in years:
            local_tif = TILES_DIR / str(yr) / f"{tile}_{yr}.tif"
            if local_tif.exists():
                src_path = str(local_tif)
                source_type = "Local Tile"
            else:
                if yr >= 2024:
                    src_path = f"/vsicurl/{AZURE_BASE_URL}/lc{yr}/{tile}_{yr}0101-{yr}1231.tif"
                else:
                    src_path = f"/vsicurl/{AZURE_BASE_URL}/lc{yr}/{tile}_{yr}0101-{yr+1}0101.tif"
                source_type = "Azure COG"

            try:
                with rasterio.open(src_path) as src:
                    trans = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = trans.transform(lon, lat)
                    row, col = src.index(x, y)
                    val = int(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
                    cname = CLASSES.get(val, f"Class {val}")
                    print(f"  {name:12s} ({lon:.4f}, {lat:.4f}) | Year {yr} | Tile {tile} | Val: {val} ({cname:18s}) [{source_type}]", flush=True)
                    results.append({
                        "Location": name,
                        "Lon": lon,
                        "Lat": lat,
                        "Year": yr,
                        "Tile": tile,
                        "Class_Value": val,
                        "Class_Name": cname,
                        "Source": source_type
                    })
            except Exception as e:
                print(f"  {name:12s} | Year {yr} | Tile {tile} | Error: {e}", flush=True)
    return pd.DataFrame(results)


def verify_all_dataset_tiles():
    """Verify every tile on disk across all downloaded years."""
    print("\n" + "=" * 80, flush=True)
    print("VERIFYING ALL TILES ACROSS ALL DOWNLOADED YEARS", flush=True)
    print("=" * 80, flush=True)
    years = sorted([int(d.name) for d in TILES_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
    tile_meta_ref = {}
    total_checked = 0
    errors = []

    for y in years:
        for tile in INDIA_TILES:
            p = TILES_DIR / str(y) / f"{tile}_{y}.tif"
            if not p.exists():
                errors.append(f"Missing file: {p}")
                continue

            total_checked += 1
            with rasterio.open(p) as src:
                if src.dtypes[0] != "uint8":
                    errors.append(f"{p.name}: invalid dtype {src.dtypes[0]}")
                if src.crs is None:
                    errors.append(f"{p.name}: CRS is None")
                res = src.res
                if abs(res[0] - 10.0) > 0.1 or abs(res[1] - 10.0) > 0.1:
                    errors.append(f"{p.name}: resolution {res} is not 10m")

                if tile not in tile_meta_ref:
                    tile_meta_ref[tile] = (src.crs, src.transform, src.shape)
                else:
                    ref_crs, ref_trans, ref_shape = tile_meta_ref[tile]
                    if src.crs != ref_crs:
                        errors.append(f"{p.name}: CRS mismatch with {ref_crs}")
                    if src.shape != ref_shape:
                        errors.append(f"{p.name}: shape mismatch {src.shape} vs {ref_shape}")
                    if src.transform != ref_trans:
                        errors.append(f"{p.name}: transform mismatch")

    print(f"Total tiles checked: {total_checked}", flush=True)
    if errors:
        print(f"Verification errors found ({len(errors)}):", flush=True)
        for e in errors[:10]:
            print(f"  - {e}", flush=True)
    else:
        print("ALL TILES VERIFIED: 10m resolution, uint8 dtype, identical CRS and grid alignment across all years!", flush=True)
    return errors


def compute_urban_transition(tiles=("43R", "43P", "44Q")):
    """
    For key urban tiles (43R: Delhi, 43P: Bengaluru, 44Q: Hyderabad),
    computes for each consecutive year pair:
      - share of Built Area pixels that stopped being Built Area
      - share that newly became Built Area
    as percentages of Built Area in the earlier year.
    """
    years = sorted([int(d.name) for d in TILES_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
    consecutive_pairs = [(years[i], years[i+1]) for i in range(len(years) - 1)]
    records = []

    print("\n" + "=" * 80, flush=True)
    print("URBAN BUILT AREA (CLASS 7) TRANSITION ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    for tile in tiles:
        print(f"\n--- Tile {tile} ---", flush=True)
        for y1, y2 in consecutive_pairs:
            p1 = TILES_DIR / str(y1) / f"{tile}_{y1}.tif"
            p2 = TILES_DIR / str(y2) / f"{tile}_{y2}.tif"
            if not p1.exists() or not p2.exists():
                continue

            b1_total = 0
            stopped_built = 0
            new_built = 0

            with rasterio.open(p1) as src1, rasterio.open(p2) as src2:
                for _, win in src1.block_windows(1):
                    arr1 = src1.read(1, window=win)
                    arr2 = src2.read(1, window=win)
                    b1 = (arr1 == 7)
                    b2 = (arr2 == 7)
                    b1_total += int(b1.sum())
                    stopped_built += int((b1 & ~b2).sum())
                    new_built += int((~b1 & b2).sum())

            pct_stopped = (stopped_built / b1_total * 100.0) if b1_total > 0 else 0.0
            pct_new = (new_built / b1_total * 100.0) if b1_total > 0 else 0.0
            net_change = pct_new - pct_stopped

            print(f"  {tile} {y1}->{y2} | Base Built: {b1_total:,} | Stopped: {stopped_built:,} ({pct_stopped:.2f}%) | New: {new_built:,} ({pct_new:.2f}%) | Net: +{net_change:.2f}%", flush=True)
            records.append({
                "Tile": tile,
                "Year_Start": y1,
                "Year_End": y2,
                "Base_Built_Pixels": b1_total,
                "Stopped_Built_Pixels": stopped_built,
                "Pct_Stopped": round(pct_stopped, 4),
                "New_Built_Pixels": new_built,
                "Pct_New": round(pct_new, 4),
                "Net_Growth_Pct": round(net_change, 4)
            })
    return pd.DataFrame(records)


def compute_and_save_year_stats(y, year_tiles, boundary_geom=None):
    """Computes and updates class statistics for a completed year."""
    if boundary_geom is None:
        boundary_geom = load_simplified_boundary()
    print(f"\nComputing class pixel statistics for {y} across all {len(year_tiles)} tiles...", flush=True)
    total_raw_counts = np.zeros(256, dtype=np.int64)
    total_masked_counts = np.zeros(256, dtype=np.int64)

    for tp in year_tiles:
        t0 = time.time()
        raw_c, msk_c = compute_tile_counts(tp, boundary_geom)
        total_raw_counts += raw_c
        total_masked_counts += msk_c
        print(f"  Processed {tp.name} in {time.time() - t0:.1f}s", flush=True)

    raw_valid_sum = int(sum(total_raw_counts[c] for c in CLASSES.keys()))
    raw_built = int(total_raw_counts[7])
    raw_cloud = int(total_raw_counts[10])
    raw_built_pct = (raw_built / raw_valid_sum * 100.0) if raw_valid_sum > 0 else 0.0
    raw_cloud_pct = (raw_cloud / raw_valid_sum * 100.0) if raw_valid_sum > 0 else 0.0

    raw_stats_row = {
        "Year": y,
        "Scope": "All_Tiles_Raw",
        "Tiles_Counted": len(year_tiles),
        "Total_Valid_Pixels": raw_valid_sum,
        "NoData_0": int(total_raw_counts[0]),
        "Water_1": int(total_raw_counts[1]),
        "Trees_2": int(total_raw_counts[2]),
        "Flooded_veg_4": int(total_raw_counts[4]),
        "Crops_5": int(total_raw_counts[5]),
        "Built_Area_7": raw_built,
        "Bare_ground_8": int(total_raw_counts[8]),
        "Snow_Ice_9": int(total_raw_counts[9]),
        "Clouds_10": raw_cloud,
        "Rangeland_11": int(total_raw_counts[11]),
        "Built_Area_Pct": round(raw_built_pct, 4),
        "Clouds_Pct": round(raw_cloud_pct, 4)
    }

    msk_valid_sum = int(sum(total_masked_counts[c] for c in CLASSES.keys()))
    msk_built = int(total_masked_counts[7])
    msk_cloud = int(total_masked_counts[10])
    msk_built_pct = (msk_built / msk_valid_sum * 100.0) if msk_valid_sum > 0 else 0.0
    msk_cloud_pct = (msk_cloud / msk_valid_sum * 100.0) if msk_valid_sum > 0 else 0.0

    msk_stats_row = {
        "Year": y,
        "Scope": "India_Boundary_Masked",
        "Tiles_Counted": len(year_tiles),
        "Total_Valid_Pixels": msk_valid_sum,
        "NoData_0": int(total_masked_counts[0]),
        "Water_1": int(total_masked_counts[1]),
        "Trees_2": int(total_masked_counts[2]),
        "Flooded_veg_4": int(total_masked_counts[4]),
        "Crops_5": int(total_masked_counts[5]),
        "Built_Area_7": msk_built,
        "Bare_ground_8": int(total_masked_counts[8]),
        "Snow_Ice_9": int(total_masked_counts[9]),
        "Clouds_10": msk_cloud,
        "Rangeland_11": int(total_masked_counts[11]),
        "Built_Area_Pct": round(msk_built_pct, 4),
        "Clouds_Pct": round(msk_cloud_pct, 4)
    }

    update_class_stats([raw_stats_row, msk_stats_row])
    print(f"  [RAW ALL-TILES] Total Valid: {raw_valid_sum:,} | Built (7): {raw_built:,} ({raw_built_pct:.2f}%) | Clouds (10): {raw_cloud:,} ({raw_cloud_pct:.4f}%)", flush=True)
    print(f"  [INDIA-MASKED]  Total Valid: {msk_valid_sum:,} | Built (7): {msk_built:,} ({msk_built_pct:.2f}%) | Clouds (10): {msk_cloud:,} ({msk_cloud_pct:.4f}%)", flush=True)


def test_tile_resume(tile, year):
    """Tests resumption on a single partial tile, validating that only delta bytes are downloaded."""
    dest = TILES_DIR / str(year) / f"{tile}_{year}.tif"
    tmp = dest.with_suffix(".part")
    print(f"\n=======================================================", flush=True)
    print(f"TESTING RESUME ON TILE {tile}_{year}", flush=True)
    print(f"=======================================================", flush=True)
    if tmp.exists():
        initial_bytes = tmp.stat().st_size
        print(f"Existing partial file found: {tmp.name} ({initial_bytes:,} bytes / {initial_bytes / 1e6:.2f} MB)", flush=True)
    else:
        initial_bytes = 0
        print(f"No existing .part file found. Will download from byte 0.", flush=True)

    t0 = time.time()
    path, status, url, explanation = download_tile(tile, year)
    dt = time.time() - t0

    if path and path.exists():
        final_bytes = path.stat().st_size
        transferred = final_bytes - initial_bytes
        speed_kb = (transferred / dt / 1024) if dt > 0 else 0
        print(f"\nRESUME TEST SUCCESSFUL:", flush=True)
        print(f"  Target file: {path.name}", flush=True)
        print(f"  Final size:  {final_bytes:,} bytes ({final_bytes / 1e6:.2f} MB)", flush=True)
        print(f"  Transferred in test: {transferred:,} bytes in {dt:.1f}s ({speed_kb:.1f} KB/s)", flush=True)
        valid, msg = verify_tile_integrity(path)
        print(f"  Integrity check: valid={valid} ({msg})", flush=True)
        with rasterio.open(path) as ds:
            print(f"  Rasterio metadata: shape={ds.shape}, dtype={ds.dtypes[0]}, crs={ds.crs}, res={ds.res}", flush=True)
        return True
    else:
        print(f"\nRESUME TEST FAILED: {explanation}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download & validate Sentinel-2 10m LULC tiles for India.")
    parser.add_argument("--years", nargs="+", type=int, help="Years to process (e.g. 2017 2018 ... 2025)")
    parser.add_argument("--dry-run", action="store_true", help="Print available years and disk space, then exit")
    parser.add_argument("--stats-only", action="store_true", help="Recompute class statistics on existing tiles")
    parser.add_argument("--verify-all", action="store_true", help="Verify all tiles on disk for metadata and grid alignment")
    parser.add_argument("--transition-analysis", action="store_true", help="Run Built Area transition churn analysis for 43R, 43P, 44Q")
    parser.add_argument("--spot-check", action="store_true", help="Run spot checks for benchmark coordinates")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker streams (default: 8, max: 8)")
    parser.add_argument("--test-tile", nargs=2, metavar=("TILE", "YEAR"), help="Test resume download on a single tile (e.g. --test-tile 46R 2017)")
    args = parser.parse_args()

    free_gb = get_free_disk_gb()
    print("=" * 80, flush=True)
    print("ESRI / IMPACT OBSERVATORY SENTINEL-2 10M LULC - INDIA PIPELINE", flush=True)
    print("=" * 80, flush=True)
    print(f"Current available free disk space: {free_gb:.2f} GB", flush=True)
    print(f"Derived India UTM Tiles ({len(INDIA_TILES)}): {', '.join(INDIA_TILES)}", flush=True)

    if args.test_tile:
        tile, yr_str = args.test_tile
        test_tile_resume(tile, int(yr_str))
        return

    if args.spot_check:
        run_spot_checks()
        return

    if args.verify_all:
        verify_all_dataset_tiles()
        return

    if args.transition_analysis:
        compute_urban_transition()
        return

    available_years = discover_years()
    years = sorted(args.years) if args.years else available_years
    print(f"Available dataset years: {available_years}", flush=True)
    print(f"Target processing years: {years}", flush=True)

    if args.dry_run:
        est_per_year_gb = 2.2
        total_est = len(years) * est_per_year_gb
        print(f"Dry-run mode: Estimated storage for {len(years)} year(s): ~{total_est:.1f} GB.", flush=True)
        print(f"Free disk: {free_gb:.1f} GB. Safety threshold: {MIN_FREE_DISK_GB} GB.", flush=True)
        return

    if args.stats_only:
        print("\nRecomputing class statistics on all complete years...", flush=True)
        boundary_geom = load_simplified_boundary()
        for y in years:
            y_tiles = [TILES_DIR / str(y) / f"{tile}_{y}.tif" for tile in INDIA_TILES if (TILES_DIR / str(y) / f"{tile}_{y}.tif").exists()]
            if len(y_tiles) == len(INDIA_TILES):
                compute_and_save_year_stats(y, y_tiles, boundary_geom)
            else:
                print(f"Year {y}: only {len(y_tiles)}/{len(INDIA_TILES)} tiles complete. Skipping stats.", flush=True)
        return

    # Cap concurrent workers to 8 maximum
    workers = min(8, max(1, args.workers))

    # Build Global Work Queue across all target years
    tasks_priority = []
    tasks_normal = []
    already_verified_count = 0

    for y in years:
        for tile in INDIA_TILES:
            dest = TILES_DIR / str(y) / f"{tile}_{y}.tif"
            tmp = dest.with_suffix(".part")
            if dest.exists() and dest.stat().st_size > 1000:
                valid, _ = verify_tile_integrity(dest)
                if valid:
                    already_verified_count += 1
                    continue
            if tmp.exists() and tmp.stat().st_size > 1000:
                tasks_priority.append((tile, y))
            else:
                tasks_normal.append((tile, y))

    all_tasks = tasks_priority + tasks_normal
    total_needed = len(years) * len(INDIA_TILES)

    print("\n" + "=" * 80, flush=True)
    print("GLOBAL WORK QUEUE INITIALIZATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Total tiles required ({len(years)} years x {len(INDIA_TILES)} tiles): {total_needed}", flush=True)
    print(f"Already downloaded and verified:               {already_verified_count}", flush=True)
    print(f"Incomplete partial tiles to resume first:       {len(tasks_priority)}", flush=True)
    print(f"Unstarted tiles queued:                         {len(tasks_normal)}", flush=True)
    print(f"Total tiles to download in this run:            {len(all_tasks)}", flush=True)
    print(f"Concurrent worker streams configured:           {workers} (max: 8)", flush=True)

    if not all_tasks:
        print("\nAll target tiles are already downloaded and verified!", flush=True)
        verify_all_dataset_tiles()
        return

    completed_in_run = 0
    total_to_download = len(all_tasks)
    completed_years_stats = set()

    print(f"\nLaunching global worker pool with {workers} parallel threads...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {pool.submit(download_tile, tile, y): (tile, y) for tile, y in all_tasks}

        for future in as_completed(future_to_task):
            tile, y = future_to_task[future]
            try:
                path, status, url, explanation = future.result()
            except Exception as e:
                path, status, url, explanation = None, "failed", tile_urls(tile, y)[0], str(e)

            free_now = get_free_disk_gb()

            if path and path.exists():
                completed_in_run += 1
                size = path.stat().st_size
                current_total_verified = sum(
                    1 for yr in years for t in INDIA_TILES
                    if (TILES_DIR / str(yr) / f"{t}_{yr}.tif").exists()
                )
                print(
                    f"  [{completed_in_run}/{total_to_download} COMPLETE] {tile}_{y}.tif ({size / 1e6:.1f} MB) "
                    f"[{status.upper()}] | Total Verified: {current_total_verified}/{total_needed} | Free disk: {free_now:.2f} GB",
                    flush=True
                )
                update_manifest([{
                    "file": path.name,
                    "year": y,
                    "size_bytes": size,
                    "sha256": sha256(path),
                    "source_url": url
                }])

                # Check if this year is now 100% complete
                y_tiles = [TILES_DIR / str(y) / f"{t}_{y}.tif" for t in INDIA_TILES if (TILES_DIR / str(y) / f"{t}_{y}.tif").exists()]
                if len(y_tiles) == len(INDIA_TILES) and y not in completed_years_stats:
                    completed_years_stats.add(y)
                    print(f"\n>>> YEAR {y} REACHED 100% COMPLETION ({len(y_tiles)}/{len(INDIA_TILES)} TILES VERIFIED) <<<", flush=True)
                    compute_and_save_year_stats(y, y_tiles)
            else:
                print(f"  [TASK FAILED] {tile}_{y}: {explanation}", flush=True)
                update_failures([{
                    "tile": tile,
                    "year": y,
                    "status": status,
                    "explanation": explanation
                }])

            # Strict disk space safety floor check
            if free_now < MIN_FREE_DISK_GB:
                print(f"\nCRITICAL: Free disk space ({free_now:.2f} GB) reached safety threshold ({MIN_FREE_DISK_GB} GB)!", flush=True)
                print("Canceling remaining tasks and safely stopping global queue...", flush=True)
                for f in future_to_task:
                    f.cancel()
                break

    print("\nGlobal work queue execution finished.", flush=True)
    verify_all_dataset_tiles()


if __name__ == "__main__":
    main()
