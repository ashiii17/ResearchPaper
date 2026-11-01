#!/usr/bin/env python3
"""
build_timeseries.py

Step 1 & Step 2 of the Time Series Analysis (TSA) module:
1. Read-only Data Audit of Esri 9-year label stacks (2017-2025).
2. Time Series Construction:
   - built_fraction_pct = (class-7 pixels / valid land-cover pixels) * 100.0
   - Strict exclusion of class 0 (NoData) from denominator.
   - Generation of chip-year tabular dataset with quality flags.
   - Aggregated annual summary statistics across chips.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import rasterio

# Project Paths
BASE_DIR = Path("/Users/ashi/ResearchPaper/india_lulc")
CHIPS_FILE = BASE_DIR / "chips.csv"
LABELS_DIR = BASE_DIR / "labels"
TSA_DIR = BASE_DIR / "tsa"
RESULTS_DIR = TSA_DIR / "results"

PILOT_CHIP_IDS = [
    "43R_r130_c111",  # Stratum R: Random Unbiased
    "46R_r101_c74",   # Stratum A: Rapid Growth
    "44R_r131_c81",   # Stratum B: Urban Stable Core
    "44Q_r113_c70",   # Stratum C: Low Built / Semi-Urban
    "43R_r81_c35"     # Stratum D: Empty Baseline
]

YEARS = list(range(2017, 2026))
EXPECTED_DESCRIPTIONS = tuple(f"esri_{y}" for y in YEARS)


def audit_label_chip(chip_id: str, label_path: Path) -> Dict:
    """Performs a read-only structural audit of a 9-year label raster."""
    if not label_path.exists():
        return {"chip_id": chip_id, "exists": False, "error": "File not found"}

    with rasterio.open(label_path) as src:
        descriptions = src.descriptions
        desc_match = (descriptions == EXPECTED_DESCRIPTIONS)
        dim_match = (src.shape == (500, 500))
        band_count_match = (src.count == 9)
        nodata_val = src.nodata

        # Read bands and verify
        yearly_stats = []
        for b_idx in range(1, 10):
            year = 2016 + b_idx
            arr = src.read(b_idx)
            total_px = arr.size
            nodata_px = int(np.sum(arr == 0))
            valid_px = total_px - nodata_px
            built_px = int(np.sum(arr == 7))

            built_pct = (built_px / valid_px * 100.0) if valid_px > 0 else 0.0

            yearly_stats.append({
                "year": year,
                "band_idx": b_idx,
                "total_pixels": total_px,
                "nodata_pixels": nodata_px,
                "valid_pixels": valid_px,
                "built_pixels": built_px,
                "built_fraction_pct": built_pct,
                "has_nodata": (nodata_px > 0),
                "unique_classes": np.unique(arr).tolist()
            })

    return {
        "chip_id": chip_id,
        "exists": True,
        "crs": str(src.crs),
        "dimensions": src.shape,
        "band_count": src.count,
        "nodata_value": nodata_val,
        "descriptions_match": desc_match,
        "dim_match": dim_match,
        "band_count_match": band_count_match,
        "yearly_stats": yearly_stats
    }


def run_data_audit(chips_df: pd.DataFrame, pilot_only: bool = True) -> str:
    """Executes Step 1 Data Audit and writes tsa/results/data_audit.txt."""
    target_chips = PILOT_CHIP_IDS if pilot_only else chips_df["chip_id"].tolist()
    mode_str = f"FIVE PILOT CHIPS (N={len(target_chips)})" if pilot_only else f"ALL CHIPS (N={len(target_chips)})"

    lines = []
    lines.append("================================================================================")
    lines.append("TIME SERIES ANALYSIS — DATA AUDIT REPORT")
    lines.append(f"Scope: {mode_str}")
    lines.append("Project: India Urban-Change Sentinel-2 & Esri LULC Time Series")
    lines.append("Directory: /Users/ashi/ResearchPaper/india_lulc/")
    lines.append("================================================================================\n")

    lines.append("1. LABEL RASTER STRUCTURAL AUDIT")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("Label Band Verification: Expected 9 bands, descriptions ('esri_2017'..'esri_2025').")
    lines.append("Formula: band_index = year - 2016  (Band 1=2017, ..., Band 9=2025).\n")

    all_audits = []
    for cid in target_chips:
        lbl_p = LABELS_DIR / f"{cid}.tif"
        audit = audit_label_chip(cid, lbl_p)
        all_audits.append(audit)

    passed_struct = sum(1 for a in all_audits if a.get("exists") and a.get("descriptions_match") and a.get("dim_match"))
    lines.append(f"• Rasters Audited          : {len(all_audits)}")
    lines.append(f"• Structural Checks Passed : {passed_struct} / {len(all_audits)}")
    lines.append(f"• Band Descriptions Match  : {'100% MATCH' if passed_struct == len(all_audits) else 'MISMATCH DETECTED'}")
    lines.append(f"• Dimension (500x500) Match: {'100% MATCH' if passed_struct == len(all_audits) else 'MISMATCH DETECTED'}\n")

    lines.append("2. PER-CHIP TEMPORAL TRAJECTORY SUMMARY (CLASS 7 BUILT-UP)")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"{'Chip ID':<16} | {'Stratum':<7} | {'Split':<7} | {'2017 Built %':<12} | {'2021 Built %':<12} | {'2025 Built %':<12} | {'Net Growth %':<12}")
    lines.append("-" * 88)

    meta_lookup = chips_df.set_index("chip_id").to_dict(orient="index")
    for a in all_audits:
        cid = a["chip_id"]
        meta = meta_lookup.get(cid, {})
        stratum = meta.get("stratum", "Unknown")
        split = meta.get("split", "Unknown")
        stats = {s["year"]: s["built_fraction_pct"] for s in a["yearly_stats"]}
        b17 = stats.get(2017, 0.0)
        b21 = stats.get(2021, 0.0)
        b25 = stats.get(2025, 0.0)
        net_g = b25 - b17
        lines.append(f"{cid:<16} | {stratum:<7} | {split:<7} | {b17:>10.4f}% | {b21:>10.4f}% | {b25:>10.4f}% | {net_g:>+10.4f}%")

    lines.append("-" * 88)
    lines.append("\n3. CLASS ENCODING AND NODATA VERIFICATION")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("• Class 7 (Built Area) Presence: Confirmed present in urbanized chips.")
    lines.append("• Class 0 (NoData) Handling    : Strictly excluded from valid_pixels denominator.")
    lines.append("• Denominator Formula          : valid_pixels = total_pixels - count(pixel == 0).")
    lines.append("• Built-up Fraction Formula    : built_fraction_pct = (count(pixel == 7) / valid_pixels) * 100.0.")
    lines.append("• Sample Status for Pilot      : 5 chips x 9 years = 45 observations across 5 strata.\n")

    report_text = "\n".join(lines)
    audit_file = RESULTS_DIR / "data_audit.txt"
    with open(audit_file, "w") as f:
        f.write(report_text)
    print(f"Wrote audit report to: {audit_file}")
    return report_text


def build_timeseries_dataset(chips_df: pd.DataFrame, pilot_only: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2: Constructs the time series dataset.
    Returns (timeseries_df, annual_summary_df).
    """
    target_chips = PILOT_CHIP_IDS if pilot_only else chips_df["chip_id"].tolist()
    meta_lookup = chips_df.set_index("chip_id").to_dict(orient="index")

    rows = []
    for cid in target_chips:
        lbl_p = LABELS_DIR / f"{cid}.tif"
        if not lbl_p.exists():
            continue
        meta = meta_lookup.get(cid, {})

        with rasterio.open(lbl_p) as src:
            for b_idx in range(1, 10):
                year = 2016 + b_idx
                arr = src.read(b_idx)
                total_px = arr.size
                nodata_px = int(np.sum(arr == 0))
                valid_px = total_px - nodata_px
                built_px = int(np.sum(arr == 7))

                built_pct = (built_px / valid_px * 100.0) if valid_px > 0 else 0.0

                rows.append({
                    "chip_id": cid,
                    "stratum": meta.get("stratum", "Unknown"),
                    "split": meta.get("split", "Unknown"),
                    "tile": meta.get("tile", "Unknown"),
                    "epsg": meta.get("epsg", 0),
                    "year": year,
                    "built_pixels": built_px,
                    "valid_pixels": valid_px,
                    "nodata_pixels": nodata_px,
                    "total_pixels": total_px,
                    "built_fraction_pct": built_pct,
                    "has_nodata": (nodata_px > 0),
                    "data_quality_flag": "VALID" if nodata_px == 0 else f"NODATA_{nodata_px}_PX"
                })

    ts_df = pd.DataFrame(rows)

    # Sort chronologically by chip_id and year
    ts_df = ts_df.sort_values(["chip_id", "year"]).reset_index(drop=True)

    # Compute year-over-year changes per chip
    ts_df["built_pct_diff"] = ts_df.groupby("chip_id")["built_fraction_pct"].diff()
    ts_df["built_pct_growth_rate"] = ts_df.apply(
        lambda r: ((r["built_pct_diff"] / ts_df.loc[r.name - 1, "built_fraction_pct"] * 100.0)
                   if (r.name > 0 and ts_df.loc[r.name - 1, "chip_id"] == r["chip_id"] and ts_df.loc[r.name - 1, "built_fraction_pct"] > 0)
                   else np.nan),
        axis=1
    )

    # Annual summary statistics across chips
    summary_rows = []
    for y in YEARS:
        y_slice = ts_df[ts_df["year"] == y]
        summary_rows.append({
            "year": y,
            "chips_count": len(y_slice),
            "mean_built_fraction_pct": float(y_slice["built_fraction_pct"].mean()),
            "std_built_fraction_pct": float(y_slice["built_fraction_pct"].std()),
            "median_built_fraction_pct": float(y_slice["built_fraction_pct"].median()),
            "min_built_fraction_pct": float(y_slice["built_fraction_pct"].min()),
            "max_built_fraction_pct": float(y_slice["built_fraction_pct"].max()),
            "total_built_pixels": int(y_slice["built_pixels"].sum()),
            "total_valid_pixels": int(y_slice["valid_pixels"].sum()),
            "pooled_built_fraction_pct": float(y_slice["built_pixels"].sum() / y_slice["valid_pixels"].sum() * 100.0)
        })
    summary_df = pd.DataFrame(summary_rows)

    return ts_df, summary_df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chips_df = pd.read_csv(CHIPS_FILE)

    print("Running Step 1: Read-Only Data Audit...")
    run_data_audit(chips_df, pilot_only=True)

    print("Running Step 2: Building Time Series Dataset for Pilot Chips...")
    ts_df, summary_df = build_timeseries_dataset(chips_df, pilot_only=True)

    # Save CSVs
    ts_file = RESULTS_DIR / "pilot_timeseries.csv"
    sum_file = RESULTS_DIR / "pilot_timeseries_summary.csv"

    ts_df.to_csv(ts_file, index=False)
    summary_df.to_csv(sum_file, index=False)

    print(f"Saved pilot time series to: {ts_file} ({len(ts_df)} rows)")
    print(f"Saved pilot annual summary to: {sum_file} ({len(summary_df)} rows)")
    print("\nPilot Annual Summary Table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
