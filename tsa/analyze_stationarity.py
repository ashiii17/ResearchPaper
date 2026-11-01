#!/usr/bin/env python3
"""
analyze_stationarity.py

Step 4 of Time Series Analysis module:
Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
tests on annual built-up fraction series (2017-2025).

Methodological Highlights:
- Rigorous joint interpretation of ADF (H0: unit root) vs KPSS (H0: stationary).
- Tests both raw levels y_t and first-differences delta_y_t.
- Graceful handling of short series (N=9) and zero-variance constant series.
- Seamless execution with statsmodels or built-in numerical OLS/residuals fallback.
- Output: tsa/results/stationarity_results.csv.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = Path("/Users/ashi/ResearchPaper/india_lulc")
TSA_DIR = BASE_DIR / "tsa"
RESULTS_DIR = TSA_DIR / "results"
TIMESERIES_FILE = RESULTS_DIR / "pilot_timeseries.csv"

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def fallback_adf_test(series: np.ndarray) -> Dict:
    """
    Fallback implementation of Dickey-Fuller regression:
    delta_y_t = alpha + beta * y_{t-1} + e_t
    Computes t-statistic on beta and compares to MacKinnon critical values for N=9.
    """
    y = np.asarray(series, dtype=float)
    if np.all(y == y[0]) or np.var(y) < 1e-12:
        return {"stat": 0.0, "pvalue": 1.0, "is_constant": True, "crit_5pct": -3.0}

    n = len(y)
    dy = np.diff(y)
    y_lag = y[:-1]
    # OLS: dy = alpha + beta * y_lag
    X = np.column_stack([np.ones(n - 1), y_lag])
    try:
        beta_hat, residuals, rank, s = np.linalg.lstsq(X, dy, rcond=None)
        res = dy - X @ beta_hat
        df_e = max(n - 3, 1)
        s_sq = np.sum(res**2) / df_e
        cov = s_sq * np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(max(cov[1, 1], 1e-12))
        t_stat = float(beta_hat[1] / se_beta)

        # MacKinnon approximate critical values with constant for small N
        crit_1 = -4.1
        crit_5 = -3.1
        crit_10 = -2.7

        if t_stat < crit_1:
            p_val = 0.01
        elif t_stat < crit_5:
            p_val = 0.05
        elif t_stat < crit_10:
            p_val = 0.10
        else:
            p_val = 0.50

        return {"stat": t_stat, "pvalue": p_val, "is_constant": False, "crit_5pct": crit_5}
    except Exception as e:
        return {"stat": np.nan, "pvalue": np.nan, "error": str(e), "is_constant": False}


def fallback_kpss_test(series: np.ndarray) -> Dict:
    """
    Fallback implementation of KPSS test:
    y_t = mu + e_t
    S_t = cumsum(e_t), LM = sum(S_t^2) / (n^2 * s^2)
    """
    y = np.asarray(series, dtype=float)
    if np.all(y == y[0]) or np.var(y) < 1e-12:
        return {"stat": 0.0, "pvalue": 0.10, "is_constant": True, "crit_5pct": 0.463}

    n = len(y)
    e = y - np.mean(y)
    S = np.cumsum(e)
    # Newey-West variance estimator (lag 1)
    s_sq = np.sum(e**2) / n
    if n > 1:
        gamma1 = np.sum(e[1:] * e[:-1]) / n
        s_sq += 2.0 * (1.0 - 1.0/2.0) * gamma1
    s_sq = max(s_sq, 1e-12)

    kpss_stat = float(np.sum(S**2) / (n**2 * s_sq))
    crit_1 = 0.739
    crit_5 = 0.463
    crit_10 = 0.347

    if kpss_stat > crit_1:
        p_val = 0.01
    elif kpss_stat > crit_5:
        p_val = 0.05
    elif kpss_stat > crit_10:
        p_val = 0.10
    else:
        p_val = 0.50

    return {"stat": kpss_stat, "pvalue": p_val, "is_constant": False, "crit_5pct": crit_5}


def evaluate_chip_stationarity(chip_id: str, stratum: str, series: np.ndarray) -> Dict:
    """Runs ADF and KPSS tests on level and differenced series."""
    y = np.asarray(series, dtype=float)
    is_const = np.all(y == y[0]) or (np.var(y) < 1e-12)

    # 1. ADF on Level
    if HAS_STATSMODELS and not is_const:
        try:
            adf_res = adfuller(y, maxlag=1, autolag=None)
            adf_stat, adf_pval, adf_crit5 = float(adf_res[0]), float(adf_res[1]), float(adf_res[4]["5%"])
        except Exception:
            f_adf = fallback_adf_test(y)
            adf_stat, adf_pval, adf_crit5 = f_adf["stat"], f_adf["pvalue"], f_adf.get("crit_5pct", -3.1)
    else:
        f_adf = fallback_adf_test(y)
        adf_stat, adf_pval, adf_crit5 = f_adf["stat"], f_adf["pvalue"], f_adf.get("crit_5pct", -3.1)

    # 2. KPSS on Level
    if HAS_STATSMODELS and not is_const:
        try:
            kpss_res = kpss(y, regression="c", nlags=1)
            kpss_stat, kpss_pval, kpss_crit5 = float(kpss_res[0]), float(kpss_res[1]), float(kpss_res[3]["5%"])
        except Exception:
            f_kpss = fallback_kpss_test(y)
            kpss_stat, kpss_pval, kpss_crit5 = f_kpss["stat"], f_kpss["pvalue"], f_kpss.get("crit_5pct", 0.463)
    else:
        f_kpss = fallback_kpss_test(y)
        kpss_stat, kpss_pval, kpss_crit5 = f_kpss["stat"], f_kpss["pvalue"], f_kpss.get("crit_5pct", 0.463)

    # 3. Differenced Series Delta y_t
    dy = np.diff(y)
    is_dy_const = np.all(dy == dy[0]) or (np.var(dy) < 1e-12)

    f_adf_diff = fallback_adf_test(dy) if not (HAS_STATSMODELS and not is_dy_const) else None
    if f_adf_diff is None:
        try:
            d_res = adfuller(dy, maxlag=0, autolag=None)
            diff_adf_stat, diff_adf_pval = float(d_res[0]), float(d_res[1])
        except Exception:
            f_adf_diff = fallback_adf_test(dy)
            diff_adf_stat, diff_adf_pval = f_adf_diff["stat"], f_adf_diff["pvalue"]
    else:
        diff_adf_stat, diff_adf_pval = f_adf_diff["stat"], f_adf_diff["pvalue"]

    # Joint Hypothesis Decision Logic
    # ADF: H0 = non-stationary (p < 0.05 -> stationary)
    # KPSS: H0 = stationary (p < 0.05 -> non-stationary)
    if is_const:
        conclusion = "Constant Series (Zero Variance)"
    elif adf_pval < 0.05 and kpss_pval >= 0.05:
        conclusion = "Stationary (ADF rejects unit root; KPSS fails to reject)"
    elif adf_pval >= 0.05 and kpss_pval < 0.05:
        conclusion = "Unit Root / Non-Stationary (ADF fails to reject; KPSS rejects)"
    elif adf_pval >= 0.05 and kpss_pval >= 0.05:
        conclusion = "Inconclusive (ADF & KPSS fail to reject; small N=9 sample power limitation)"
    else:
        conclusion = "Mixed Signals / Structural Break (Both tests reject)"

    return {
        "chip_id": chip_id,
        "stratum": stratum,
        "n_obs": len(y),
        "mean_built_pct": float(np.mean(y)),
        "std_built_pct": float(np.std(y)),
        "level_adf_stat": adf_stat,
        "level_adf_pvalue": adf_pval,
        "level_adf_crit5": adf_crit5,
        "level_kpss_stat": kpss_stat,
        "level_kpss_pvalue": kpss_pval,
        "level_kpss_crit5": kpss_crit5,
        "diff_adf_stat": diff_adf_stat,
        "diff_adf_pvalue": diff_adf_pval,
        "joint_conclusion": conclusion
    }


def main():
    if not TIMESERIES_FILE.exists():
        print(f"Error: {TIMESERIES_FILE} not found. Run build_timeseries.py first.")
        sys.exit(1)

    df = pd.read_csv(TIMESERIES_FILE)
    results = []

    print("\n" + "=" * 95)
    print("STATIONARITY ANALYSIS: AUGMENTED DICKEY-FULLER (ADF) & KPSS TESTS (N=9, 2017–2025)")
    print("=" * 95)
    print("Hypotheses Interpretation:")
    print("  • ADF  : H0 = Series has unit root (Non-stationary) | H1 = Series is stationary")
    print("  • KPSS : H0 = Series is stationary / trend-stationary | H1 = Series has unit root")
    print("-" * 95)

    for cid, grp in df.groupby("chip_id"):
        grp_sorted = grp.sort_values("year")
        series = grp_sorted["built_fraction_pct"].values
        stratum = grp_sorted["stratum"].iloc[0]
        res = evaluate_chip_stationarity(cid, stratum, series)
        results.append(res)

    res_df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "stationarity_results.csv"
    res_df.to_csv(out_csv, index=False)

    # Print formatted summary table
    print(f"{'Chip ID':<15} | {'Stratum':<7} | {'ADF Stat':<10} | {'ADF p':<7} | {'KPSS Stat':<10} | {'KPSS p':<7} | {'Joint Interpretation'}")
    print("-" * 95)
    for _, r in res_df.iterrows():
        adf_s = f"{r['level_adf_stat']:.3f}" if not np.isnan(r['level_adf_stat']) else "N/A"
        adf_p = f"{r['level_adf_pvalue']:.3f}" if not np.isnan(r['level_adf_pvalue']) else "N/A"
        kpss_s = f"{r['level_kpss_stat']:.3f}" if not np.isnan(r['level_kpss_stat']) else "N/A"
        kpss_p = f"{r['level_kpss_pvalue']:.3f}" if not np.isnan(r['level_kpss_pvalue']) else "N/A"
        print(f"{r['chip_id']:<15} | {r['stratum']:<7} | {adf_s:<10} | {adf_p:<7} | {kpss_s:<10} | {kpss_p:<7} | {r['joint_conclusion']}")

    print("=" * 95)
    print(f"Detailed stationarity results saved to: {out_csv}\n")


if __name__ == "__main__":
    main()
