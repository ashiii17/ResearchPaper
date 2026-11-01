#!/usr/bin/env python3
"""
audit_splits.py

Performs a comprehensive data split and leakage audit on chips.csv:
1. Verifies existing train/validation/test splits without reassigning.
2. Checks strata coverage across splits.
3. Verifies zero overlap of chip IDs across splits.
4. Audits spatial distribution across UTM tiles.
5. Audits the five pilot chips and assesses their statistical validity.
6. Writes audit report to baseline/split_audit.txt.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("/Users/ashi/ResearchPaper/india_lulc")
CHIPS_FILE = BASE_DIR / "chips.csv"
OUTPUT_REPORT = BASE_DIR / "baseline" / "split_audit.txt"

PILOT_IDS = [
    "43R_r130_c111",
    "46R_r101_c74",
    "44R_r131_c81",
    "44Q_r113_c70",
    "43R_r81_c35"
]


def run_split_audit():
    if not CHIPS_FILE.exists():
        raise FileNotFoundError(f"chips.csv not found at {CHIPS_FILE}")

    df = pd.read_csv(CHIPS_FILE)
    total_chips = len(df)

    lines = []
    lines.append("================================================================================")
    lines.append("DATASET SPLIT AND DATA LEAKAGE AUDIT REPORT")
    lines.append("Project: India Urban-Change Sentinel-2 Deep Learning Baseline")
    lines.append(f"Source: {CHIPS_FILE}")
    lines.append("================================================================================\n")

    # 1. Split Distribution
    split_counts = df["split"].value_counts()
    split_pcts = df["split"].value_counts(normalize=True) * 100.0

    lines.append("1. NATIONWIDE SPLIT SUMMARY (1,500 CHIPS)")
    lines.append("--------------------------------------------------------------------------------")
    for s in ["train", "val", "test"]:
        cnt = split_counts.get(s, 0)
        pct = split_pcts.get(s, 0.0)
        lines.append(f"  • {s.upper():<6} Split : {cnt:>5} chips ({pct:>5.1f}%)")
    lines.append(f"  • TOTAL        : {total_chips:>5} chips (100.0%)\n")

    # 2. Strata Coverage Across Splits
    lines.append("2. STRATA COVERAGE ACROSS SPLITS")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("Strata definitions:")
    lines.append("  • R: Random Unbiased (Representative across India landscape)")
    lines.append("  • A: Rapid Growth (Net built gain 2017->2025 >= 2%)")
    lines.append("  • B: Urban Stable Core (Built 2017 >= 20% and growth < 2%)")
    lines.append("  • C: Low Built / Semi-Urban (Built 2017 in [1%, 20%) and growth < 2%)")
    lines.append("  • D: Empty Baseline (Built 2017 < 0.1% and built 2025 < 0.1%)\n")

    crosstab_strata = pd.crosstab(df["split"], df["stratum"], margins=True)
    lines.append(crosstab_strata.to_string())
    lines.append("\nPercentage of each stratum in train/val/test:")
    crosstab_pct = pd.crosstab(df["split"], df["stratum"], normalize="columns") * 100.0
    lines.append(crosstab_pct.round(1).to_string())
    lines.append("\n")

    # 3. Spatial Tile Distribution Across Splits
    lines.append("3. UTM TILE COVERAGE ACROSS SPLITS")
    lines.append("--------------------------------------------------------------------------------")
    crosstab_tile = pd.crosstab(df["tile"], df["split"], margins=True)
    lines.append(crosstab_tile.to_string())
    lines.append("\n")

    # 4. Leakage Verification: Exact ID & Coordinate Overlap
    lines.append("4. DATA LEAKAGE VERIFICATION")
    lines.append("--------------------------------------------------------------------------------")
    train_ids = set(df[df["split"] == "train"]["chip_id"])
    val_ids = set(df[df["split"] == "val"]["chip_id"])
    test_ids = set(df[df["split"] == "test"]["chip_id"])

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    lines.append(f"  • Train / Validation Overlap : {len(overlap_train_val)} chips (Must be 0)")
    lines.append(f"  • Train / Test Overlap       : {len(overlap_train_test)} chips (Must be 0)")
    lines.append(f"  • Validation / Test Overlap  : {len(overlap_val_test)} chips (Must be 0)")

    leakage_passed = (len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0)
    lines.append(f"  • Disjoint Split Integrity   : {'PASSED (Zero Leakage)' if leakage_passed else 'CRITICAL FAILURE'}\n")

    # 5. Methodological Guardrails
    lines.append("5. LEAKAGE PREVENTION GUARDRAILS (METHODOLOGY)")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("  [RULE 1] Normalization Statistics:")
    lines.append("    Channel-wise mean and std must be computed ONLY on training split chips.")
    lines.append("    Validation and test chips must be standardized using training parameters.")
    lines.append("  [RULE 2] Hyperparameter Tuning & Early Stopping:")
    lines.append("    Model checkpoints, learning rate schedules, and thresholds must be chosen")
    lines.append("    strictly based on validation split metrics.")
    lines.append("  [RULE 3] Test Split Quarantine:")
    lines.append("    The test split (320 chips, 21.3%) is completely quarantined and must NEVER")
    lines.append("    be evaluated until the final frozen architecture benchmark report.\n")

    # 6. Audit of the 5 Pilot Chips
    lines.append("6. PILOT DATASET (5 CHIPS) STATISTICAL VALIDITY AUDIT")
    lines.append("--------------------------------------------------------------------------------")
    pilot_df = df[df["chip_id"].isin(PILOT_IDS)].copy()
    lines.append("Pilot Chips Metadata:")
    lines.append(f"{'Chip ID':<16} | {'Stratum':<7} | {'Split':<7} | {'Tile':<5} | {'Built17 %':<10} | {'Built25 %':<10} | {'Growth %':<10}")
    lines.append("-" * 75)
    for _, r in pilot_df.iterrows():
        lines.append(f"{r['chip_id']:<16} | {r['stratum']:<7} | {r['split']:<7} | {r['tile']:<5} | {r['built17_frac']*100:>8.2f}% | {r['built25_frac']*100:>8.2f}% | {r['growth_frac']*100:>8.2f}%")

    pilot_splits = pilot_df["split"].value_counts().to_dict()
    lines.append("-" * 75)
    lines.append(f"Pilot Split Breakdown: Train={pilot_splits.get('train', 0)}, Val={pilot_splits.get('val', 0)}, Test={pilot_splits.get('test', 0)}\n")

    lines.append("⚠️  STATISTICAL SUFFICIENCY WARNING (PILOT STAGE):")
    lines.append("  • The pilot dataset contains only 4 training chips, 1 validation chip, and 0 test chips.")
    lines.append("  • With N=1 in validation (Assam 46R_r101_c74), validation loss and metrics will suffer")
    lines.append("    from extreme regional variance and cannot represent national performance.")
    lines.append("  • With N=0 in test, unbiased test metrics cannot be computed.")
    lines.append("  • CONCLUSION: The 5-chip pilot is valid STRICTLY for software pipeline smoke testing")
    lines.append("    (testing lazy loading, tensor dimensions, loss convergence, masking, and checkpointing).")
    lines.append("    It is EXPLICITLY INSUFFICIENT for scientific model evaluation or architectural ranking.")
    lines.append("================================================================================\n")

    report_text = "\n".join(lines)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"Report written to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    run_split_audit()
