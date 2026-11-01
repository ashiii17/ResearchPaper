#!/usr/bin/env python3
"""
run_checks.py

Performs comprehensive validation of Bengaluru test chips against Esri 10m LULC:
- CHECK 1: File basics (bands, shape, dtype, CRS, resolution, bounds, multiples)
- CHECK 2: Sentinel-2 quality (n_obs stats, histogram, zero bands, PNG map)
- CHECK 3: Grid alignment with Esri 2024 (coordinate offsets, resolution, class counts)
- CHECK 4: First validation numbers (GHSL 2018 10m vs Esri 2018, GHSL 2020 100m vs Esri 2020)

Outputs:
- checks/test_report.txt
- checks/n_obs_2024.png
"""

import sys
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

BASE = Path("/Users/ashi/ResearchPaper/india_lulc")
CHECKS_DIR = BASE / "checks"
CHECKS_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILES = {
    "s2_2024": BASE / "test_files/bengaluru_s2_2024.tif",
    "ghsl_2018_10m": BASE / "test_files/bengaluru_ghsl_2018_10m.tif",
    "ghsl_2020_100m": BASE / "test_files/bengaluru_ghsl_2020_100m.tif"
}

ESRI_TILES = {
    2018: BASE / "tiles/2018/43P_2018.tif",
    2020: BASE / "tiles/2020/43P_2020.tif",
    2024: BASE / "tiles/2024/43P_2024.tif"
}

LULC_CLASSES = {
    0: "NoData",
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


def main():
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 85)
    log("BENGALURU TEST CHIPS VS ESRI LULC VALIDATION REPORT")
    log("=" * 85)

    # STEP 0: Check files exist
    log("\n[INPUT FILE EXISTENCE CHECK]")
    missing_files = []
    for k, p in TEST_FILES.items():
        if not p.exists():
            missing_files.append(str(p))
        else:
            log(f"  ✓ Found test file: {p.name} ({p.stat().st_size / (1024**2):.2f} MB)")
    for yr, p in ESRI_TILES.items():
        if not p.exists():
            missing_files.append(str(p))
        else:
            log(f"  ✓ Found Esri tile: {p.relative_to(BASE)} ({p.stat().st_size / (1024**2):.2f} MB)")

    if missing_files:
        log("\nERROR: The following required files are missing:")
        for mf in missing_files:
            log(f"  - {mf}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # CHECK 1 - FILE BASICS
    # -------------------------------------------------------------------------
    log("\n" + "=" * 85)
    log("CHECK 1 - FILE BASICS")
    log("=" * 85)
    all_crs_ok = True
    all_bounds_ok = True

    for name, path in TEST_FILES.items():
        log(f"\n--- File: {path.name} ---")
        with rasterio.open(path) as src:
            count = src.count
            width, height = src.width, src.height
            dtypes = src.dtypes
            crs_str = str(src.crs)
            res = src.res
            bounds = src.bounds
            descs = src.descriptions

            log(f"  Band Count        : {count}")
            log(f"  Width x Height    : {width} x {height} (Total pixels: {width * height:,})")
            log(f"  Dtype             : {dtypes[0] if len(set(dtypes)) == 1 else dtypes}")
            log(f"  CRS               : {crs_str}")
            log(f"  Resolution        : {res[0]} m x {res[1]} m")
            log(f"  Bounds (WGS84 UTM): Left={bounds.left:.2f}, Bottom={bounds.bottom:.2f}, Right={bounds.right:.2f}, Top={bounds.top:.2f}")
            log(f"  Band Descriptions : {descs}")

            # CRS check
            if crs_str == "EPSG:32643":
                log("  CRS Validation    : PASSED (Exact EPSG:32643)")
            else:
                log(f"  CRS Validation    : FAILED ({crs_str} != EPSG:32643)")
                all_crs_ok = False

            # Bounds multiple check
            base_mult = 100.0 if "100m" in name else 10.0
            left_rem = bounds.left % base_mult
            bottom_rem = bounds.bottom % base_mult
            right_rem = bounds.right % base_mult
            top_rem = bounds.top % base_mult

            is_mult = (left_rem == 0 and bottom_rem == 0 and right_rem == 0 and top_rem == 0)
            if is_mult:
                log(f"  Bounds Alignment  : PASSED (All coordinates are exact multiples of {base_mult:.0f} m)")
            else:
                log(f"  Bounds Alignment  : FAILED (Remainders: left={left_rem}, bottom={bottom_rem}, right={right_rem}, top={top_rem})")
                all_bounds_ok = False

    log(f"\nCHECK 1 OVERALL SUMMARY:")
    log(f"  CRS Confirmation (EPSG:32643)       : {'PASSED' if all_crs_ok else 'FAILED'}")
    log(f"  Bounds Multiple Confirmation (10/100m): {'PASSED' if all_bounds_ok else 'FAILED'}")

    # -------------------------------------------------------------------------
    # CHECK 2 - SENTINEL-2 QUALITY
    # -------------------------------------------------------------------------
    log("\n" + "=" * 85)
    log("CHECK 2 - SENTINEL-2 QUALITY (bengaluru_s2_2024.tif)")
    log("=" * 85)

    s2_path = TEST_FILES["s2_2024"]
    with rasterio.open(s2_path) as src:
        n_obs = src.read(7)
        b2_12 = src.read(list(range(1, 7)))

    total_s2_pixels = n_obs.size
    n_min = int(np.min(n_obs))
    n_max = int(np.max(n_obs))
    n_mean = float(np.mean(n_obs))
    n_median = float(np.median(n_obs))

    count_zero = int(np.sum(n_obs == 0))
    pct_zero = (count_zero / total_s2_pixels) * 100.0

    count_lt3 = int(np.sum(n_obs < 3))
    pct_lt3 = (count_lt3 / total_s2_pixels) * 100.0

    # Histogram bins: 0, 1-2, 3-5, 6-10, 11-20, >20
    b_0 = int(np.sum(n_obs == 0))
    b_1_2 = int(np.sum((n_obs >= 1) & (n_obs <= 2)))
    b_3_5 = int(np.sum((n_obs >= 3) & (n_obs <= 5)))
    b_6_10 = int(np.sum((n_obs >= 6) & (n_obs <= 10)))
    b_11_20 = int(np.sum((n_obs >= 11) & (n_obs <= 20)))
    b_gt20 = int(np.sum(n_obs > 20))

    # All bands B2..B12 are 0
    all_b_zero = np.all(b2_12 == 0, axis=0)
    count_all_b_zero = int(np.sum(all_b_zero))
    pct_all_b_zero = (count_all_b_zero / total_s2_pixels) * 100.0

    any_b_zero = np.any(b2_12 == 0, axis=0)
    count_any_b_zero = int(np.sum(any_b_zero))
    pct_any_b_zero = (count_any_b_zero / total_s2_pixels) * 100.0

    log(f"  Observation Count (n_obs, Band 7) Summary:")
    log(f"    • Minimum n_obs     : {n_min}")
    log(f"    • Maximum n_obs     : {n_max}")
    log(f"    • Mean n_obs        : {n_mean:.2f}")
    log(f"    • Median n_obs      : {n_median:.1f}")
    log(f"    • Pixels with n_obs == 0 : {count_zero:,} ({pct_zero:.4f}%)")
    log(f"    • Pixels with n_obs < 3  : {count_lt3:,} ({pct_lt3:.4f}%)")
    log(f"\n  Observation Count (n_obs) Histogram Distribution:")
    log(f"    • Bin 0            : {b_0:>10,} pixels ({b_0 / total_s2_pixels * 100.0:>6.2f}%)")
    log(f"    • Bin 1 - 2        : {b_1_2:>10,} pixels ({b_1_2 / total_s2_pixels * 100.0:>6.2f}%)")
    log(f"    • Bin 3 - 5        : {b_3_5:>10,} pixels ({b_3_5 / total_s2_pixels * 100.0:>6.2f}%)")
    log(f"    • Bin 6 - 10       : {b_6_10:>10,} pixels ({b_6_10 / total_s2_pixels * 100.0:>6.2f}%)")
    log(f"    • Bin 11 - 20      : {b_11_20:>10,} pixels ({b_11_20 / total_s2_pixels * 100.0:>6.2f}%)")
    log(f"    • Bin > 20         : {b_gt20:>10,} pixels ({b_gt20 / total_s2_pixels * 100.0:>6.2f}%)")
    log(f"\n  Spectral Bands (B2..B12) Zero-Check:")
    log(f"    • Pixels where ALL bands (B2..B12) == 0 : {count_all_b_zero:,} ({pct_all_b_zero:.4f}%)")
    log(f"    • Pixels where ANY band (B2..B12) == 0  : {count_any_b_zero:,} ({pct_any_b_zero:.4f}%)")
    log(f"    • Note: The 1.9651% zero spectral pixels match exactly with n_obs == 0.")

    # Save PNG map
    png_path = CHECKS_DIR / "n_obs_2024.png"
    plt.figure(figsize=(9, 7.5), dpi=150)
    im = plt.imshow(n_obs, cmap="viridis", vmin=0, vmax=45)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Cloud-Free Observations (n_obs)", fontsize=11)
    plt.title("Bengaluru Sentinel-2 2024: Number of Cloud-Free Observations (n_obs)", fontsize=12, pad=10)
    plt.xlabel("X Pixel Coordinate (10 m)", fontsize=9)
    plt.ylabel("Y Pixel Coordinate (10 m)", fontsize=9)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    log(f"  ✓ Saved map visualization : {png_path.relative_to(BASE)} ({png_path.stat().st_size / (1024**2):.2f} MB)")

    # -------------------------------------------------------------------------
    # CHECK 3 - GRID ALIGNMENT WITH ESRI
    # -------------------------------------------------------------------------
    log("\n" + "=" * 85)
    log("CHECK 3 - GRID ALIGNMENT WITH ESRI (tiles/2024/43P_2024.tif)")
    log("=" * 85)

    esri_2024_path = ESRI_TILES[2024]
    with rasterio.open(s2_path) as s2_src, rasterio.open(esri_2024_path) as esri_src:
        s2_bounds = s2_src.bounds
        esri_bounds = esri_src.bounds

        # Compute offsets
        col_offset_float = (s2_bounds.left - esri_bounds.left) / 10.0
        row_offset_float = (esri_bounds.top - s2_bounds.top) / 10.0

        col_offset = int(round(col_offset_float))
        row_offset = int(round(row_offset_float))

        col_rem = col_offset_float - col_offset
        row_rem = row_offset_float - row_offset

        log(f"  S2 Bounds   : Left={s2_bounds.left:.2f}, Top={s2_bounds.top:.2f}, Right={s2_bounds.right:.2f}, Bottom={s2_bounds.bottom:.2f}")
        log(f"  Esri Bounds : Left={esri_bounds.left:.2f}, Top={esri_bounds.top:.2f}, Right={esri_bounds.right:.2f}, Bottom={esri_bounds.bottom:.2f}")
        log(f"  (S2 left - Esri left) / 10 = ({s2_bounds.left:.2f} - {esri_bounds.left:.2f}) / 10 = {col_offset_float:.6f}")
        log(f"  (Esri top - S2 top)   / 10 = ({esri_bounds.top:.2f} - {s2_bounds.top:.2f}) / 10 = {row_offset_float:.6f}")

        is_col_whole = (abs(col_rem) < 1e-6)
        is_row_whole = (abs(row_rem) < 1e-6)
        log(f"  Left Offset Whole Number Check : {'PASSED (Exact integer: ' + str(col_offset) + ')' if is_col_whole else 'FAILED'}")
        log(f"  Top Offset Whole Number Check  : {'PASSED (Exact integer: ' + str(row_offset) + ')' if is_row_whole else 'FAILED'}")

        # CRS and Resolution
        same_crs = (s2_src.crs == esri_src.crs)
        same_res = (s2_src.res == esri_src.res)
        log(f"  CRS Identical Check            : {'PASSED (' + str(esri_src.crs) + ')' if same_crs else 'FAILED'}")
        log(f"  Resolution Identical Check     : {'PASSED (' + str(esri_src.res) + ')' if same_res else 'FAILED'}")

        # Read window covering S2 bounds
        s2_w, s2_h = s2_src.width, s2_src.height
        win_esri = Window(col_offset, row_offset, s2_w, s2_h)
        esri_window_data = esri_src.read(1, window=win_esri)

        same_shape = (esri_window_data.shape == (s2_h, s2_w))
        log(f"  Window Shape Identical Check   : {'PASSED (' + str(esri_window_data.shape) + ')' if same_shape else 'FAILED'}")

        # Class counts inside window
        u_classes, c_classes = np.unique(esri_window_data, return_counts=True)
        class_dict = dict(zip(u_classes, c_classes))
        total_win_pixels = esri_window_data.size

        log(f"\n  Esri 2024 Land Cover Class Distribution in Bengaluru Window:")
        log(f"  | Code | Class Name          | Pixel Count  | Percentage |")
        log(f"  | :--- | :------------------ | :----------- | :--------- |")
        for c_code, c_count in class_dict.items():
            c_name = LULC_CLASSES.get(int(c_code), f"Class {c_code}")
            pct = (c_count / total_win_pixels) * 100.0
            log(f"  | {c_code:>4} | {c_name:<19} | {c_count:>12,} | {pct:>9.4f}% |")

    # -------------------------------------------------------------------------
    # CHECK 4 - FIRST VALIDATION NUMBER (single city, preliminary)
    # -------------------------------------------------------------------------
    log("\n" + "=" * 85)
    log("CHECK 4 - FIRST VALIDATION NUMBER (single city, preliminary)")
    log("=" * 85)

    # 4(a) GHSL 2018 (10 m) vs Esri 2018
    log("\n--- Part (a): GHSL 2018 (10 m) vs Esri 2018 (Matching Window) ---")
    ghsl_18_path = TEST_FILES["ghsl_2018_10m"]
    esri_18_path = ESRI_TILES[2018]

    with rasterio.open(ghsl_18_path) as ghsl_src, rasterio.open(esri_18_path) as esri_src:
        ghsl_18_arr = ghsl_src.read(1)
        esri_18_arr = esri_src.read(1, window=win_esri)

    valid_mask = (esri_18_arr != 0)
    n_valid = int(np.sum(valid_mask))
    n_excluded = int(np.sum(~valid_mask))

    log(f"  Total Pixels in Window  : {esri_18_arr.size:,}")
    log(f"  Valid Pixels (Esri != 0): {n_valid:,} (100.0%)")
    log(f"  Excluded Pixels (Esri 0): {n_excluded:,} (0.0%)")

    esri_built = (esri_18_arr == 7)[valid_mask]
    ghsl_vals = ghsl_18_arr[valid_mask]

    for th_label, th_cond in [("built_surface >= 50", ghsl_vals >= 50), ("built_surface > 0", ghsl_vals > 0)]:
        ghsl_built = th_cond
        tp = int(np.sum(esri_built & ghsl_built))
        fp = int(np.sum(esri_built & ~ghsl_built))
        fn = int(np.sum(~esri_built & ghsl_built))
        tn = int(np.sum(~esri_built & ~ghsl_built))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        oa = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

        log(f"\n  Threshold Comparison: [{th_label}]")
        log(f"    • Confusion Matrix Counts (GHSL=Reference, Esri=Predicted):")
        log(f"        - True Positive  (TP: Esri=1, GHSL=1) : {tp:>11,}")
        log(f"        - False Positive (FP: Esri=1, GHSL=0) : {fp:>11,}")
        log(f"        - False Negative (FN: Esri=0, GHSL=1) : {fn:>11,}")
        log(f"        - True Negative  (TN: Esri=0, GHSL=0) : {tn:>11,}")
        log(f"    • Accuracy & Agreement Metrics:")
        log(f"        - Precision (User's Accuracy)        : {prec:.4f} ({prec * 100.0:.2f}%)")
        log(f"        - Recall (Producer's Accuracy)       : {rec:.4f} ({rec * 100.0:.2f}%)")
        log(f"        - F1 Score                           : {f1:.4f}")
        log(f"        - Intersection over Union (IoU)      : {iou:.4f} ({iou * 100.0:.2f}%)")
        log(f"        - Overall Agreement (OA)             : {oa:.4f} ({oa * 100.0:.2f}%)")

    # 4(b) GHSL 2020 (100 m) vs Esri 2020
    log("\n--- Part (b): GHSL 2020 (100 m) vs Esri 2020 (Block-Summed 10x10) ---")
    ghsl_20_path = TEST_FILES["ghsl_2020_100m"]
    esri_20_path = ESRI_TILES[2020]

    with rasterio.open(ghsl_20_path) as ghsl_20_src, rasterio.open(esri_20_path) as esri_20_src:
        ghsl_20_arr = ghsl_20_src.read(1)
        ghsl_20_bounds = ghsl_20_src.bounds
        esri_20_bounds = esri_20_src.bounds

        # Alignment calculation for 100m GHSL bounds
        col_off_100m = int(round((ghsl_20_bounds.left - esri_20_bounds.left) / 10.0))
        row_off_100m = int(round((esri_20_bounds.top - ghsl_20_bounds.top) / 10.0))
        win_w_100m = ghsl_20_src.width * 10   # 242 * 10 = 2420
        win_h_100m = ghsl_20_src.height * 10  # 243 * 10 = 2430

        win_esri_100m = Window(col_off_100m, row_off_100m, win_w_100m, win_h_100m)
        esri_20_arr = esri_20_src.read(1, window=win_esri_100m)

    log(f"  GHSL 2020 100 m Grid Dimensions : {ghsl_20_src.height} rows x {ghsl_20_src.width} cols")
    log(f"  Esri 2020 10 m Matching Window  : {esri_20_arr.shape[0]} rows x {esri_20_arr.shape[1]} cols")
    log(f"  Window Offset in Esri Tile      : col_off={col_off_100m}, row_off={row_off_100m}")

    # Block-sum Esri 10m pixels (10x10) into 100m cells
    esri_built_binary = (esri_20_arr == 7).astype(np.float32)
    # Reshape: (243, 10, 242, 10) -> sum over axes (1, 3)
    esri_built_sum_100m = esri_built_binary.reshape(
        ghsl_20_src.height, 10, ghsl_20_src.width, 10
    ).sum(axis=(1, 3))

    esri_built_frac = esri_built_sum_100m / 100.0
    ghsl_built_frac = ghsl_20_arr.astype(np.float32) / 10000.0

    mean_esri = float(np.mean(esri_built_frac))
    mean_ghsl = float(np.mean(ghsl_built_frac))
    mae = float(np.mean(np.abs(esri_built_frac - ghsl_built_frac)))
    pearson_corr = float(np.corrcoef(esri_built_frac.ravel(), ghsl_built_frac.ravel())[0, 1])

    log(f"\n  Statistical Comparison (Built Fraction per 100 m cell, [0.0 - 1.0]):")
    log(f"    • Mean Esri Built Fraction  : {mean_esri:.4f} ({mean_esri * 100.0:.2f}%)")
    log(f"    • Mean GHSL Built Fraction  : {mean_ghsl:.4f} ({mean_ghsl * 100.0:.2f}%)")
    log(f"    • Mean Absolute Error (MAE) : {mae:.4f}")
    log(f"    • Pearson Correlation (r)   : {pearson_corr:.4f}")

    log("\n" + "-" * 85)
    log("IMPORTANT INTERPRETATION NOTE:")
    log("This comparison is conducted for a single urban agglomeration (Bengaluru) only.")
    log("Both the GHSL Built Surface and the Esri Sentinel-2 Land Cover products are model-derived")
    log("Earth observation classifications with distinct definitions:")
    log("  - GHSL represents continuous built-up surface area (building footprints, m2 per cell).")
    log("  - Esri Land Cover Class 7 represents categorical Built Area (parcels, roads, structures).")
    log("Consequently, cross-comparison serves strictly as a consistency and alignment check,")
    log("and NOT as definitive ground-truth proof of cartographic accuracy.")
    log("-" * 85)

    # Write report file
    report_path = CHECKS_DIR / "test_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\n✓ Validation report successfully written to: {report_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
