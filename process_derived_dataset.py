#!/usr/bin/env python3
"""
process_derived_dataset.py

Builds compact derived datasets from the 153 finished tiles in ./tiles/:
1. derived/first_built_year/{tile}.tif (uint8, 1=2017...9=2025, 0=never persistent, 255=invalid)
2. derived/built_2017/{tile}.tif (uint8, 1=built, 0=not built, 255=invalid)
3. derived/built_2025/{tile}.tif (uint8, 1=built, 0=not built, 255=invalid)

Features:
- Pure windowed streaming (never loads a full tile into RAM).
- Identical native grid, CRS, transform, shape as raw tiles.
- Tiled 512x512, DEFLATE compression with predictor 2.
- Quantifies classification noise: stable built, persistent new built,
  unstable pixels (% ever built, split into 'built early disappears' vs 'flickers'),
  and annual raw vs persistence-cleaned built totals.
- Evaluates both India-boundary-masked (simplified 0.005 deg) and raw tile extent.
- Hard disk check: caps at 4 GB derived output, keeps >= 3 GB free.
- Multiprocessing worker pool with persistent JSON stats and CSV report generation.
"""

import sys
import os
import json
import time
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio
import rasterio.features
from rasterio.windows import Window
import shapely
import shapely.geometry
from pyproj import Transformer

# Paths
BASE = Path("/Users/ashi/ResearchPaper/india_lulc")
TILES_DIR = BASE / "tiles"
DERIVED_DIR = BASE / "derived"
FBY_DIR = DERIVED_DIR / "first_built_year"
B2017_DIR = DERIVED_DIR / "built_2017"
B2025_DIR = DERIVED_DIR / "built_2025"
STATS_DIR = DERIVED_DIR / "stats"

SIMPLIFIED_BOUNDARY_FILE = BASE / "india_boundary_simplified_0.005.geojson"
RAW_BOUNDARY_FILE = BASE / "india_boundary.geojson"

YEARS = list(range(2017, 2026))  # 2017..2025 (9 years)
ALL_TILES = [
    "42Q", "42R", "43P", "43Q", "43R", "43S",
    "44P", "44Q", "44R", "44S", "45Q", "45R",
    "46N", "46P", "46Q", "46R", "47R"
]

WINDOW_SIZE = 2048  # Multiple of 512 and 256 for optimal I/O block alignment
MAX_DERIVED_GB = 4.0
MIN_FREE_DISK_GB = 3.0


def get_free_disk_gb():
    """Returns free disk space in GB."""
    return shutil.disk_usage(BASE).free / (1024 ** 3)


def get_dir_size_bytes(directory):
    """Calculates total size of all files in directory in bytes."""
    total = 0
    if not directory.exists():
        return 0
    for p in directory.glob("**/*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def load_simplified_boundary():
    """Loads cached simplified boundary geometry."""
    if SIMPLIFIED_BOUNDARY_FILE.exists():
        with open(SIMPLIFIED_BOUNDARY_FILE) as f:
            gj = json.load(f)
        return shapely.geometry.shape(gj)
    elif RAW_BOUNDARY_FILE.exists():
        print("Creating simplified boundary cache...", flush=True)
        with open(RAW_BOUNDARY_FILE) as f:
            gj = json.load(f)
        raw_geom = shapely.unary_union([shapely.geometry.shape(feat["geometry"]) for feat in gj["features"]])
        simp_geom = raw_geom.simplify(0.005, preserve_topology=True)
        with open(SIMPLIFIED_BOUNDARY_FILE, "w") as f:
            json.dump(shapely.geometry.mapping(simp_geom), f)
        return simp_geom
    else:
        print("Warning: No boundary file found. Boundary masking will be skipped.", flush=True)
        return None


class TileStats:
    """Accumulates noise quantification metrics for a tile (raw and masked)."""

    def __init__(self, tile_name):
        self.tile_name = tile_name
        self.scopes = ["raw", "masked"]
        self.data = {
            s: {
                "total_valid_pixels": 0,
                "stable_built_2017": 0,          # fby == 1
                "persistent_new_built": {str(y): 0 for y in range(2018, 2026)},  # fby == 2..9
                "total_persistent_built": 0,     # fby in 1..9
                "ever_built": 0,                 # built in >= 1 year
                "unstable_total": 0,             # ever_built and not persistent through 2025
                "unstable_early_disappears": 0,  # contiguous run ending before 2025
                "unstable_single_year": 0,       # built in exactly 1 year
                "unstable_multi_disappears": 0,  # contiguous run > 1 year
                "unstable_flickers": 0,          # non-contiguous (on and off)
                "annual_raw_built": {str(y): 0 for y in range(2017, 2026)},
                "annual_cleaned_built": {str(y): 0 for y in range(2017, 2026)},
            }
            for s in self.scopes
        }

    def to_dict(self):
        return {"tile": self.tile_name, "scopes": self.data}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["tile"])
        obj.data = d["scopes"]
        return obj

    def add_window(self, arr_9y, fby, mask_india=None):
        H, W = fby.shape
        B = (arr_9y == 7)  # shape (9, H, W)
        
        # Determine masks for scopes
        masks = {}
        masks["raw"] = slice(None)  # all pixels in window
        if mask_india is None:
            masks["masked"] = None
        elif isinstance(mask_india, (bool, np.bool_)):
            masks["masked"] = slice(None) if mask_india else None
        else:
            masks["masked"] = mask_india

        # Any built pixels in this window?
        ever_built_win = np.any(B, axis=0)  # shape (H, W)
        has_built = np.any(ever_built_win)

        for scope in ["raw", "masked"]:
            sub_mask = masks[scope]
            if sub_mask is None:
                continue

            # Sub-arrays for this scope
            if sub_mask is slice(None):
                s_arr_9y = arr_9y
                s_B = B
                s_fby = fby
                s_ever = ever_built_win
            else:
                s_arr_9y = arr_9y[:, sub_mask]
                s_B = B[:, sub_mask]
                s_fby = fby[sub_mask]
                s_ever = ever_built_win[sub_mask]

            if s_fby.size == 0:
                continue

            st = self.data[scope]
            # Valid pixels (not nodata 0 and not cloud 10 in 2025)
            st["total_valid_pixels"] += int(np.sum((s_arr_9y[8] != 0) & (s_arr_9y[8] != 10)))

            # Annual raw built
            raw_sums = np.sum(s_B, axis=1 if s_B.ndim == 2 else (1, 2))
            for i, y in enumerate(YEARS):
                st["annual_raw_built"][str(y)] += int(raw_sums[i])

            # Persistent counts
            # Stable built in 2017: fby == 1
            st["stable_built_2017"] += int(np.sum(s_fby == 1))

            # Persistent new built: fby == 2..9
            for i, y in enumerate(range(2018, 2026), start=2):
                c = int(np.sum(s_fby == i))
                st["persistent_new_built"][str(y)] += c

            tot_pers = int(np.sum((s_fby >= 1) & (s_fby <= 9)))
            st["total_persistent_built"] += tot_pers

            # Annual cleaned built: fby <= (y - 2016)
            for i, y in enumerate(YEARS, start=1):
                st["annual_cleaned_built"][str(y)] += int(np.sum((s_fby >= 1) & (s_fby <= i)))

            # Ever built & Unstable analysis
            if has_built and np.any(s_ever):
                ev_count = int(np.sum(s_ever))
                st["ever_built"] += ev_count

                # Unstable = ever built and not persistent through 2025
                unstable_mask = s_ever & ~((s_fby >= 1) & (s_fby <= 9))
                u_count = int(np.sum(unstable_mask))
                st["unstable_total"] += u_count

                if u_count > 0:
                    # Analyze 9-year sequence for unstable pixels
                    B_un = s_B[:, unstable_mask]  # shape (9, U)

                    indices = np.arange(9)[:, None]
                    first = np.min(np.where(B_un, indices, 99), axis=0)
                    last = np.max(np.where(B_un, indices, -1), axis=0)
                    cnt = np.sum(B_un, axis=0)
                    span = last - first + 1
                    is_contig = (cnt == span)
                    is_single = (cnt == 1)

                    early_disp = int(np.sum(is_contig))
                    single_yr = int(np.sum(is_single))
                    multi_disp = int(np.sum(is_contig & ~is_single))
                    flickers = int(np.sum(~is_contig))

                    st["unstable_early_disappears"] += early_disp
                    st["unstable_single_year"] += single_yr
                    st["unstable_multi_disappears"] += multi_disp
                    st["unstable_flickers"] += flickers


def process_single_tile(tile, boundary_geom=None, force=False):
    """
    Processes all 9 years for a single tile window by window.
    Writes derived GeoTIFFs:
      - derived/first_built_year/{tile}.tif
      - derived/built_2017/{tile}.tif
      - derived/built_2025/{tile}.tif
      - derived/stats/{tile}.json
    Returns TileStats object.
    """
    FBY_DIR.mkdir(parents=True, exist_ok=True)
    B2017_DIR.mkdir(parents=True, exist_ok=True)
    B2025_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    out_fby_path = FBY_DIR / f"{tile}.tif"
    out_b2017_path = B2017_DIR / f"{tile}.tif"
    out_b2025_path = B2025_DIR / f"{tile}.tif"
    out_stats_path = STATS_DIR / f"{tile}.json"

    # Resume check: if all 3 tif files and stats.json exist and are valid, skip
    if not force and out_fby_path.exists() and out_b2017_path.exists() and out_b2025_path.exists() and out_stats_path.exists():
        if out_fby_path.stat().st_size > 1000 and out_stats_path.stat().st_size > 10:
            print(f"Loading cached results for Tile {tile}...", flush=True)
            with open(out_stats_path) as f:
                d = json.load(f)
            return TileStats.from_dict(d)

    src_paths = [TILES_DIR / str(y) / f"{tile}_{y}.tif" for y in YEARS]
    for p in src_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required source tile: {p}")

    if boundary_geom is None:
        boundary_geom = load_simplified_boundary()

    stats = TileStats(tile)

    # Open all 9 years
    srcs = [rasterio.open(p) for p in src_paths]
    ref_src = srcs[0]
    crs = ref_src.crs
    transform = ref_src.transform
    width = ref_src.width
    height = ref_src.height
    bounds = ref_src.bounds

    # Transform boundary geometry to tile CRS
    clipped_geom = None
    if boundary_geom is not None and crs is not None:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        utm_geom = shapely.transform(
            boundary_geom,
            lambda coords: np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))
        )
        tile_box_utm = shapely.geometry.box(*bounds)
        clipped_geom = utm_geom.intersection(tile_box_utm)

    # Output raster profile
    out_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 255,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 2
    }

    t0 = time.time()
    print(f"\nProcessing Tile {tile} [Shape: {height}x{width}, Window: {WINDOW_SIZE}x{WINDOW_SIZE}]...", flush=True)

    with rasterio.open(out_fby_path, "w", **out_profile) as dst_fby, \
         rasterio.open(out_b2017_path, "w", **out_profile) as dst_b2017, \
         rasterio.open(out_b2025_path, "w", **out_profile) as dst_b2025:

        n_windows_y = (height + WINDOW_SIZE - 1) // WINDOW_SIZE
        n_windows_x = (width + WINDOW_SIZE - 1) // WINDOW_SIZE
        total_windows = n_windows_y * n_windows_x
        win_idx = 0

        for row_off in range(0, height, WINDOW_SIZE):
            win_h = min(WINDOW_SIZE, height - row_off)
            for col_off in range(0, width, WINDOW_SIZE):
                win_w = min(WINDOW_SIZE, width - col_off)
                win = Window(col_off, row_off, win_w, win_h)
                win_idx += 1

                # Read 9 years for this window
                arr_9y = np.stack([s.read(1, window=win) for s in srcs], axis=0)  # shape (9, win_h, win_w)

                # Class 7 is Built, Class 0 is NoData, Class 10 is Clouds
                B = (arr_9y == 7)
                inv_2025 = (arr_9y[8] == 0) | (arr_9y[8] == 10)
                inv_2017 = (arr_9y[0] == 0) | (arr_9y[0] == 10)

                # Compute first_built_year
                # Working backward from 2025 (index 8) to 2017 (index 0):
                fby = np.zeros((win_h, win_w), dtype=np.uint8)
                curr_persistent = B[8].copy()
                fby[curr_persistent] = 9  # 2025 - 2016 = 9

                for idx in range(7, -1, -1):  # 2024 down to 2017
                    curr_persistent &= B[idx]
                    fby[curr_persistent] = idx + 1  # 2017=1 ... 2024=8

                fby[inv_2025] = 255

                # Compute built_2017
                b2017 = np.zeros((win_h, win_w), dtype=np.uint8)
                b2017[B[0]] = 1
                b2017[inv_2017] = 255

                # Compute built_2025
                b2025 = np.zeros((win_h, win_w), dtype=np.uint8)
                b2025[B[8]] = 1
                b2025[inv_2025] = 255

                # Write derived windows
                dst_fby.write(fby, 1, window=win)
                dst_b2017.write(b2017, 1, window=win)
                dst_b2025.write(b2025, 1, window=win)

                # Boundary masking check for this window
                mask_india = None
                if clipped_geom is not None and not clipped_geom.is_empty:
                    win_bounds = rasterio.windows.bounds(win, transform)
                    win_box = shapely.geometry.box(*win_bounds)
                    if win_box.within(clipped_geom):
                        mask_india = True
                    elif not win_box.intersects(clipped_geom):
                        mask_india = False
                    else:
                        win_trans = rasterio.windows.transform(win, transform)
                        mask_india = rasterio.features.rasterize(
                            [(clipped_geom, 1)],
                            out_shape=(win_h, win_w),
                            transform=win_trans,
                            fill=0,
                            dtype="uint8"
                        ) == 1

                # Accumulate statistics
                stats.add_window(arr_9y, fby, mask_india)

                if win_idx % 250 == 0 or win_idx == total_windows:
                    elapsed = time.time() - t0
                    print(f"  [{tile}] Window {win_idx}/{total_windows} ({win_idx/total_windows*100:.1f}%) in {elapsed:.1f}s", flush=True)

    for s in srcs:
        s.close()

    elapsed = time.time() - t0
    fby_sz = out_fby_path.stat().st_size / (1024 ** 2)
    b17_sz = out_b2017_path.stat().st_size / (1024 ** 2)
    b25_sz = out_b2025_path.stat().st_size / (1024 ** 2)
    tot_sz = fby_sz + b17_sz + b25_sz
    print(f"Done Tile {tile} in {elapsed:.1f}s | Sizes: fby={fby_sz:.2f}MB, b2017={b17_sz:.2f}MB, b2025={b25_sz:.2f}MB | Total={tot_sz:.2f}MB", flush=True)

    # Save stats to json
    with open(out_stats_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    return stats


def _worker_wrapper(tile):
    """Wrapper function for multiprocessing pool."""
    return process_single_tile(tile, None)


def print_stats_table(stats_list):
    """Formats, prints markdown tables, and exports CSV files for Step 3."""
    # Combine stats for All Tiles
    combined = TileStats("ALL_17_TILES")
    for st in stats_list:
        for scope in ["raw", "masked"]:
            for k in ["total_valid_pixels", "stable_built_2017", "total_persistent_built",
                      "ever_built", "unstable_total", "unstable_early_disappears",
                      "unstable_single_year", "unstable_multi_disappears", "unstable_flickers"]:
                combined.data[scope][k] += st.data[scope][k]
            for y in range(2018, 2026):
                combined.data[scope]["persistent_new_built"][str(y)] += st.data[scope]["persistent_new_built"][str(y)]
            for y in range(2017, 2026):
                combined.data[scope]["annual_raw_built"][str(y)] += st.data[scope]["annual_raw_built"][str(y)]
                combined.data[scope]["annual_cleaned_built"][str(y)] += st.data[scope]["annual_cleaned_built"][str(y)]

    # Tiles to display
    display_tiles = [st for st in stats_list if st.tile_name in ("43R", "43P", "44Q")]
    display_tiles.append(combined)

    t1_rows = []
    t2_rows = []

    for scope in ["masked", "raw"]:
        scope_label = "India Boundary Masked (0.005° tolerance)" if scope == "masked" else "Raw (Entire Tile Grid)"
        print(f"\n{'=' * 95}")
        print(f"TABLE 1: PERSISTENT VS UNSTABLE BUILT-UP DYNAMICS [{scope_label}]")
        print(f"{'=' * 95}")
        print(f"| Metric | 43R (Delhi) | 43P (Bengaluru) | 44Q (Hyderabad) | ALL 17 TILES COMBINED |")
        print(f"| :--- | :--- | :--- | :--- | :--- |")

        def col_vals(fn):
            cols = []
            for t in ["43R", "43P", "44Q", "ALL_17_TILES"]:
                st_match = next((s for s in display_tiles if s.tile_name == t), None)
                if st_match is None:
                    cols.append("N/A")
                else:
                    cols.append(fn(st_match.data[scope]))
            return " | ".join(cols)

        # 1. Stable built 2017 (persistent to 2025)
        print(f"| **a) Stable Built (2017 -> 2025)** | {col_vals(lambda d: '{:,}'.format(d['stable_built_2017']))} |")

        # 2. Persistent new built by year 2018..2025
        print(f"| **b) Persistent New Built:** | | | | |")
        for y in range(2018, 2026):
            print(f"| &nbsp;&nbsp;&nbsp;&nbsp;• First Built in {y} | {col_vals(lambda d, yr=y: '{:,}'.format(d['persistent_new_built'][str(yr)]))} |")

        # Total Persistent Built through 2025
        print(f"| **Total Persistent Built (2025)** | {col_vals(lambda d: '{:,}'.format(d['total_persistent_built']))} |")

        # 3. Unstable pixels
        print(f"| **c) UNSTABLE PIXELS (Non-persistent)** | | | | |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;• Total Unstable Pixels | {col_vals(lambda d: '{:,}'.format(d['unstable_total']))} |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;• Ever-Built Pixels (>= 1 yr) | {col_vals(lambda d: '{:,}'.format(d['ever_built']))} |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;• **Unstable % of Ever-Built** | {col_vals(lambda d: '{:.2f}%'.format(d['unstable_total'] / d['ever_built'] * 100.0 if d['ever_built'] > 0 else 0.0))} |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;• *Split 1: Built early disappears* | {col_vals(lambda d: '{:,} ({:.2f}%)'.format(d['unstable_early_disappears'], d['unstable_early_disappears'] / d['ever_built'] * 100.0 if d['ever_built'] > 0 else 0.0))} |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Single-year transient | {col_vals(lambda d: '{:,} ({:.2f}%)'.format(d['unstable_single_year'], d['unstable_single_year'] / d['ever_built'] * 100.0 if d['ever_built'] > 0 else 0.0))} |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Multi-year persistent lost | {col_vals(lambda d: '{:,} ({:.2f}%)'.format(d['unstable_multi_disappears'], d['unstable_multi_disappears'] / d['ever_built'] * 100.0 if d['ever_built'] > 0 else 0.0))} |")
        print(f"| &nbsp;&nbsp;&nbsp;&nbsp;• *Split 2: Flickers on and off* | {col_vals(lambda d: '{:,} ({:.2f}%)'.format(d['unstable_flickers'], d['unstable_flickers'] / d['ever_built'] * 100.0 if d['ever_built'] > 0 else 0.0))} |")

        # Table 2: Annual Series Raw vs Cleaned
        print(f"\n{'=' * 95}")
        print(f"TABLE 2: ANNUAL BUILT-UP AREA: RAW VS PERSISTENCE-CLEANED [{scope_label}]")
        print(f"{'=' * 95}")
        print(f"| Year | Scope | 43R (Delhi) Raw | 43R Cleaned | 43P (Bengaluru) Raw | 43P Cleaned | 44Q (Hyd) Raw | 44Q Cleaned | ALL 17 Raw | ALL 17 Cleaned |")
        print(f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for y in YEARS:
            def get_rc(tile_name, yr=y):
                m = next((s for s in display_tiles if s.tile_name == tile_name), None)
                if not m:
                    return 0, 0
                raw_val = m.data[scope]["annual_raw_built"][str(yr)]
                cln_val = m.data[scope]["annual_cleaned_built"][str(yr)]
                return raw_val, cln_val

            r_43r, c_43r = get_rc("43R")
            r_43p, c_43p = get_rc("43P")
            r_44q, c_44q = get_rc("44Q")
            r_all, c_all = get_rc("ALL_17_TILES")
            print(f"| {y} | {scope.capitalize()} | {r_43r:,} | {c_43r:,} | {r_43p:,} | {c_43p:,} | {r_44q:,} | {c_44q:,} | {r_all:,} | {c_all:,} |")

            t2_rows.append({
                "Year": y, "Scope": scope,
                "43R_Raw": r_43r, "43R_Cleaned": c_43r,
                "43P_Raw": r_43p, "43P_Cleaned": c_43p,
                "44Q_Raw": r_44q, "44Q_Cleaned": c_44q,
                "ALL_17_Raw": r_all, "ALL_17_Cleaned": c_all
            })

    # Save CSVs
    import pandas as pd
    pd.DataFrame(t2_rows).to_csv(DERIVED_DIR / "annual_built_comparison_table2.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Build compact persistent built-up dataset.")
    parser.add_argument("--size-test-only", action="store_true", help="Run size test on 43R only, project total, then exit.")
    parser.add_argument("--tiles", nargs="+", default=None, help="Tiles to process.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4).")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already completed tiles.")
    args = parser.parse_args()

    free_start = get_free_disk_gb()
    print(f"Initial Free Disk Space: {free_start:.2f} GiB", flush=True)
    if free_start < MIN_FREE_DISK_GB:
        print(f"ERROR: Free disk space ({free_start:.2f} GiB) is below minimum threshold ({MIN_FREE_DISK_GB} GiB)!", flush=True)
        sys.exit(1)

    boundary_geom = load_simplified_boundary()

    # STEP 1: SIZE TEST ON TILE 43R
    print("\n" + "=" * 80, flush=True)
    print("STEP 1: SIZE TEST ON TILE 43R", flush=True)
    print("=" * 80, flush=True)

    stats_43r = process_single_tile("43R", boundary_geom, force=args.force)

    # Measure output files
    fby_43r_bytes = (FBY_DIR / "43R.tif").stat().st_size
    b17_43r_bytes = (B2017_DIR / "43R.tif").stat().st_size
    b25_43r_bytes = (B2025_DIR / "43R.tif").stat().st_size
    tile_43r_total_bytes = fby_43r_bytes + b17_43r_bytes + b25_43r_bytes
    tile_43r_total_mb = tile_43r_total_bytes / (1024 ** 2)

    projected_17_tiles_bytes = tile_43r_total_bytes * 17
    projected_17_tiles_gb = projected_17_tiles_bytes / (1024 ** 3)
    free_now = get_free_disk_gb()

    print("\n--- SIZE TEST RESULTS FOR TILE 43R ---", flush=True)
    print(f"  first_built_year/43R.tif : {fby_43r_bytes / (1024**2):.2f} MB", flush=True)
    print(f"  built_2017/43R.tif       : {b17_43r_bytes / (1024**2):.2f} MB", flush=True)
    print(f"  built_2025/43R.tif       : {b25_43r_bytes / (1024**2):.2f} MB", flush=True)
    print(f"  Single Tile Total (43R)  : {tile_43r_total_mb:.2f} MB ({tile_43r_total_bytes / (1024**3):.4f} GB)", flush=True)
    print(f"  Projected for 17 Tiles   : {projected_17_tiles_gb:.3f} GB ({projected_17_tiles_bytes / (1024**2):.1f} MB)", flush=True)
    print(f"  Max Output Cap Allowed   : {MAX_DERIVED_GB:.1f} GB", flush=True)
    print(f"  Current Free Disk Space  : {free_now:.2f} GiB (Floor: {MIN_FREE_DISK_GB:.1f} GiB)", flush=True)

    if projected_17_tiles_gb > MAX_DERIVED_GB:
        print(f"\nSTOP: Projected size ({projected_17_tiles_gb:.3f} GB) exceeds 4 GB limit!", flush=True)
        print("Please provide an external-drive path to proceed.", flush=True)
        sys.exit(2)

    if (free_now - projected_17_tiles_gb) < MIN_FREE_DISK_GB:
        print(f"\nSTOP: Projected free space after processing ({free_now - projected_17_tiles_gb:.2f} GiB) drops below {MIN_FREE_DISK_GB} GiB!", flush=True)
        print("Please provide an external-drive path to proceed.", flush=True)
        sys.exit(3)

    print("SUCCESS: Projected size is well within 4 GB cap and keeps > 3 GB free!", flush=True)

    if args.size_test_only:
        print("\nSize test only requested. Stopping before continuing to other tiles.")
        return

    # STEP 2: PROCESS REMAINING 16 TILES
    tiles_to_process = [t for t in ALL_TILES if t != "43R"]
    if args.tiles:
        tiles_to_process = [t for t in args.tiles if t != "43R"]

    print("\n" + "=" * 80, flush=True)
    print(f"STEP 2: PROCESSING REMAINING {len(tiles_to_process)} TILES [Workers: {args.workers}]", flush=True)
    print("=" * 80, flush=True)

    all_stats = [stats_43r]

    # Process tiles with process pool
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_tile = {executor.submit(_worker_wrapper, t): t for t in tiles_to_process}
        for future in as_completed(future_to_tile):
            t = future_to_tile[future]
            try:
                st = future.result()
                all_stats.append(st)
                cur_free = get_free_disk_gb()
                print(f"Finished {t} | Free space: {cur_free:.2f} GiB", flush=True)
                if cur_free < MIN_FREE_DISK_GB:
                    print(f"\nSTOP: Free disk space reached floor of {MIN_FREE_DISK_GB} GiB! Cancelling remaining.", flush=True)
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(4)
            except Exception as e:
                print(f"ERROR processing tile {t}: {e}", flush=True)
                raise e

    # STEP 3: NOISE QUANTIFICATION TABLES
    print("\n" + "=" * 80, flush=True)
    print("STEP 3: NOISE QUANTIFICATION TABLES", flush=True)
    print("=" * 80, flush=True)
    print_stats_table(all_stats)

    # STEP 4: FINAL SUMMARY & DISK REPORT
    print("\n" + "=" * 80, flush=True)
    print("STEP 4: FINAL STORAGE REPORT", flush=True)
    print("=" * 80, flush=True)
    total_derived_bytes = get_dir_size_bytes(DERIVED_DIR)
    final_free_gb = get_free_disk_gb()
    files_written = list(DERIVED_DIR.glob("**/*.tif"))

    print(f"Total Derived Dataset Size : {total_derived_bytes / (1024**3):.3f} GB ({total_derived_bytes / (1024**2):.1f} MB)", flush=True)
    print(f"Remaining Free Disk Space   : {final_free_gb:.2f} GiB", flush=True)
    print(f"Total Derived Files Written: {len(files_written)} (Expected: 51 = 17 tiles x 3 layers)", flush=True)
    print("ALL DERIVED DATASETS PROCESSED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
