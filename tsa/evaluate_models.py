#!/usr/bin/env python3
"""
evaluate_models.py

Step 5 Chronological Holdout Evaluation for Time Series Forecasting:
- Splits 9 annual observations chronologically:
    • Train Horizon  : 2017–2023 (N=7 points)
    • Holdout Horizon: 2024–2025 (H=2 points, h=1, 2)
- Zero temporal leakage: Models trained strictly on <= 2023.
- Evaluates: Naive Persistence, Drift Baseline, Holt's Linear, ARIMA.
- Computes MAE, RMSE, and conditional MAPE (with mathematical zero-exclusion).
- Generates:
    • tsa/results/forecast_evaluation.csv
    • tsa/figures/forecast_comparison_pilot.png & .svg
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from tsa.forecast_models import (
    NaivePersistenceForecaster,
    DriftForecaster,
    HoltLinearForecaster,
    ARIMAForecaster
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BASE_DIR = PROJECT_ROOT
TSA_DIR = BASE_DIR / "tsa"
RESULTS_DIR = TSA_DIR / "results"
FIGURES_DIR = TSA_DIR / "figures"
TIMESERIES_FILE = RESULTS_DIR / "pilot_timeseries.csv"


def calculate_ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Computes MAE, RMSE, and mathematically guarded MAPE.
    Excludes MAPE when actual value is near-zero (< 0.05%) to avoid division by zero.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_p - y_t)))
    rmse = float(np.sqrt(np.mean((y_p - y_t)**2)))

    # MAPE calculation with zero-exclusion guard
    if np.any(y_t < 0.05):
        mape = np.nan
        mape_note = "EXCLUDED (Actual value < 0.05% near-zero)"
    else:
        mape = float(np.mean(np.abs((y_t - y_p) / y_t)) * 100.0)
        mape_note = "VALID"

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mape_note": mape_note
    }


def evaluate_chip_forecasts(chip_id: str, stratum: str, chip_data: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    """Runs chronological holdout evaluation on a single chip."""
    sorted_df = chip_data.sort_values("year").reset_index(drop=True)
    train_mask = sorted_df["year"] <= 2023
    test_mask = sorted_df["year"] >= 2024

    train_df = sorted_df[train_mask]
    test_df = sorted_df[test_mask]

    y_train = train_df["built_fraction_pct"].values
    y_test = test_df["built_fraction_pct"].values
    test_years = test_df["year"].values
    steps = len(y_test)

    models = {
        "Naive_Persistence": NaivePersistenceForecaster(),
        "Drift_Baseline": DriftForecaster(),
        "Holt_Linear": HoltLinearForecaster(alpha=0.5, beta=0.2),
        "ARIMA(0,1,0)_Drift": ARIMAForecaster(order=(0, 1, 0), trend="t")
    }

    records = []
    predictions_map = {"years": test_years, "actual": y_test, "train_years": train_df["year"].values, "train_actual": y_train}

    for m_name, model in models.items():
        model.fit(y_train)
        preds = model.predict(steps=steps)
        metrics = calculate_ts_metrics(y_test, preds)

        records.append({
            "chip_id": chip_id,
            "stratum": stratum,
            "model": m_name,
            "pred_2024": float(preds[0]),
            "actual_2024": float(y_test[0]),
            "pred_2025": float(preds[1]),
            "actual_2025": float(y_test[1]),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "mape_status": metrics["mape_note"]
        })
        predictions_map[m_name] = preds

    return records, predictions_map


def plot_forecast_comparisons(all_predictions: Dict[str, Dict]):
    """Generates actual vs predicted forecast comparison plots."""
    fig_png = FIGURES_DIR / "forecast_comparison_pilot.png"
    fig_svg = FIGURES_DIR / "forecast_comparison_pilot.svg"

    chips = sorted(all_predictions.keys())

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(len(chips), 1, figsize=(10, 15), sharex=True, dpi=150)
        if len(chips) == 1:
            axes = [axes]

        colors = {
            "Naive_Persistence": "#475569",
            "Drift_Baseline": "#2563eb",
            "Holt_Linear": "#059669",
            "ARIMA(0,1,0)_Drift": "#dc2626"
        }

        for idx, cid in enumerate(chips):
            ax = axes[idx]
            info = all_predictions[cid]
            all_years = list(info["train_years"]) + list(info["years"])
            all_actual = list(info["train_actual"]) + list(info["actual"])

            # Plot actual historical + ground truth
            ax.plot(all_years, all_actual, marker="o", color="black", linewidth=2.2, label="Actual Observations")
            ax.axvline(2023.5, color="#94a3b8", linestyle="--", linewidth=1.5, label="Train / Test Cutoff")

            # Plot predictions for 2024 and 2025
            for m_name in ["Naive_Persistence", "Drift_Baseline", "Holt_Linear", "ARIMA(0,1,0)_Drift"]:
                ax.plot(info["years"], info[m_name], marker="s", linestyle=":", linewidth=2, color=colors.get(m_name, "#2563eb"), label=m_name)

            ax.set_title(f"Forecast Comparison: Chip {cid}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Built Fraction (%)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(loc="upper left", fontsize=8, frameon=True)

        axes[-1].set_xlabel("Year", fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_png)
        plt.savefig(fig_svg)
        plt.close()
        print(f"Generated forecast comparison figure: {fig_png} and {fig_svg}")
    else:
        # Fallback SVG chart
        svg_content = "<text x='400' y='250' font-size='16' text-anchor='middle' fill='#334155'>Actual vs Predicted Forecast Chart (2017-2025)</text>"
        with open(fig_svg, "w") as f:
            f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='800' height='500'>{svg_content}</svg>")
        print(f"Generated fallback SVG forecast comparison: {fig_svg}")


def main():
    if not TIMESERIES_FILE.exists():
        print(f"Error: {TIMESERIES_FILE} does not exist.")
        sys.exit(1)

    df = pd.read_csv(TIMESERIES_FILE)
    all_records = []
    all_predictions = {}

    print("\n" + "=" * 105)
    print("CHRONOLOGICAL HOLDOUT FORECAST EVALUATION (Train: 2017–2023 [N=7] | Test: 2024–2025 [H=2])")
    print("=" * 105)

    for cid, grp in df.groupby("chip_id"):
        stratum = grp["stratum"].iloc[0]
        recs, preds_info = evaluate_chip_forecasts(cid, stratum, grp)
        all_records.extend(recs)
        all_predictions[cid] = preds_info

    eval_df = pd.DataFrame(all_records)
    out_csv = RESULTS_DIR / "forecast_evaluation.csv"
    eval_df.to_csv(out_csv, index=False)

    # Print formatted model comparison table
    print(f"{'Chip ID':<15} | {'Stratum':<7} | {'Model':<20} | {'MAE (%)':<8} | {'RMSE (%)':<9} | {'MAPE':<8} | {'MAPE Status'}")
    print("-" * 105)
    for _, r in eval_df.iterrows():
        mape_str = f"{r['mape']:.2f}%" if not np.isnan(r['mape']) else "N/A"
        print(f"{r['chip_id']:<15} | {r['stratum']:<7} | {r['model']:<20} | {r['mae']:<8.4f} | {r['rmse']:<9.4f} | {mape_str:<8} | {r['mape_status']}")

    print("=" * 105)
    print(f"Saved evaluation metrics CSV to: {out_csv}")

    # Generate plots
    plot_forecast_comparisons(all_predictions)

    # Model ranking summary
    print("\nMODEL AVERAGE ERROR SUMMARY ACROSS PILOT CHIPS:")
    avg_summary = eval_df.groupby("model")[["mae", "rmse"]].mean().reset_index()
    print(avg_summary.to_string(index=False))


if __name__ == "__main__":
    main()
