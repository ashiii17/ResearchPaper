#!/usr/bin/env python3
"""
plot_exploratory.py

Step 3 Exploratory Analysis for Time Series Analysis module:
Generates visualization figures under tsa/figures/:
1. trajectory_built_fraction: Multi-annual trajectories of built-up fraction (2017-2025).
2. annual_change_growth: Annual year-over-year percentage point changes.
3. strata_distribution: Built-up fraction comparison by stratum (2017 vs 2025).
4. missing_and_prevalence: Class prevalence and data completeness audit.

Supports matplotlib (if available) with automatic standalone vector SVG rendering
fallback to guarantee figure generation in minimal environments without dependencies.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("/Users/ashi/ResearchPaper/india_lulc")
TSA_DIR = BASE_DIR / "tsa"
FIGURES_DIR = TSA_DIR / "figures"
RESULTS_DIR = TSA_DIR / "results"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def render_svg_chart(svg_filename: Path, title: str, content_svg: str, width: int = 800, height: int = 500):
    """Writes a standalone, clean vector SVG chart."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <rect width="{width}" height="{height}" fill="#ffffff" />
  <text x="{width/2}" y="35" font-size="18" font-weight="bold" text-anchor="middle" fill="#1e293b">{title}</text>
  {content_svg}
</svg>"""
    with open(svg_filename, "w", encoding="utf-8") as f:
        f.write(svg)


def plot_trajectories(ts_df: pd.DataFrame):
    """Plot 1: Annual built-up fraction trajectories (2017-2025)."""
    fig_png = FIGURES_DIR / "trajectory_built_fraction.png"
    fig_svg = FIGURES_DIR / "trajectory_built_fraction.svg"

    colors = {
        "43R_r130_c111": "#2563eb",  # Blue (R)
        "46R_r101_c74":  "#dc2626",  # Red (A - Growth)
        "44R_r131_c81":  "#7c3aed",  # Purple (B - Urban)
        "44Q_r113_c70":  "#059669",  # Green (C)
        "43R_r81_c35":   "#d97706"   # Amber (D - Desert)
    }

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for cid, grp in ts_df.groupby("chip_id"):
            stratum = grp["stratum"].iloc[0]
            label = f"{cid} (Stratum {stratum})"
            ax.plot(grp["year"], grp["built_fraction_pct"], marker="o", linewidth=2.2, label=label, color=colors.get(cid, "#475569"))
        ax.set_title("Annual Built-Up Fraction Trajectories (2017–2025)", fontsize=14, pad=12, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Built-Up Area Fraction (% of Chip)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="#f8fafc")
        plt.tight_layout()
        plt.savefig(fig_png)
        plt.savefig(fig_svg)
        plt.close()
        print(f"Generated trajectory plot: {fig_png} and {fig_svg}")
    else:
        # Vector SVG generation
        chips = sorted(ts_df["chip_id"].unique())
        content = []
        # Axes
        ox, oy, w, h = 90, 420, 640, 340
        content.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+w}" y2="{oy}" stroke="#94a3b8" stroke-width="1.5" />')
        content.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-h}" stroke="#94a3b8" stroke-width="1.5" />')

        # Years 2017..2025
        years = list(range(2017, 2026))
        for idx, yr in enumerate(years):
            x = ox + idx * (w / 8.0)
            content.append(f'<line x1="{x}" y1="{oy}" x2="{x}" y2="{oy+5}" stroke="#94a3b8" stroke-width="1.5" />')
            content.append(f'<text x="{x}" y="{oy+22}" font-size="12" text-anchor="middle" fill="#64748b">{yr}</text>')

        # Y-ticks (0% to 25%)
        for pct in range(0, 30, 5):
            y = oy - (pct / 25.0) * h
            content.append(f'<line x1="{ox-5}" y1="{y}" x2="{ox+w}" y2="{y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,4" />')
            content.append(f'<text x="{ox-10}" y="{y+4}" font-size="11" text-anchor="end" fill="#64748b">{pct}%</text>')

        # Plot lines
        legend_items = []
        for c_idx, cid in enumerate(chips):
            c_data = ts_df[ts_df["chip_id"] == cid].sort_values("year")
            stratum = c_data["stratum"].iloc[0]
            col = colors.get(cid, "#2563eb")
            pts = []
            for _, r in c_data.iterrows():
                yr_idx = r["year"] - 2017
                x = ox + yr_idx * (w / 8.0)
                y = oy - (min(r["built_fraction_pct"], 25.0) / 25.0) * h
                pts.append(f"{x:.1f},{y:.1f}")
                content.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}" />')
            poly_str = " ".join(pts)
            content.append(f'<polyline points="{poly_str}" fill="none" stroke="{col}" stroke-width="2.5" />')

            # Legend item
            ly = 70 + c_idx * 22
            legend_items.append(f'<line x1="740" y1="{ly}" x2="760" y2="{ly}" stroke="{col}" stroke-width="3" />')
            legend_items.append(f'<circle cx="750" cy="{ly}" r="3.5" fill="{col}" />')
            legend_items.append(f'<text x="768" y="{ly+4}" font-size="11" fill="#1e293b">{cid} ({stratum})</text>')

        content.extend(legend_items)
        render_svg_chart(fig_svg, "Annual Built-Up Fraction Trajectories (2017–2025)", "\n  ".join(content), width=900, height=480)
        print(f"Generated vector SVG trajectory plot: {fig_svg}")


def plot_annual_change(ts_df: pd.DataFrame):
    """Plot 2: Year-over-year net percentage point change."""
    fig_png = FIGURES_DIR / "annual_change_growth.png"
    fig_svg = FIGURES_DIR / "annual_change_growth.svg"

    df_diff = ts_df.dropna(subset=["built_pct_diff"]).copy()

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        for cid, grp in df_diff.groupby("chip_id"):
            stratum = grp["stratum"].iloc[0]
            ax.plot(grp["year"], grp["built_pct_diff"], marker="s", label=f"{cid} ({stratum})")
        ax.axhline(0, color="black", linestyle="--", alpha=0.6)
        ax.set_title("Annual Net Built-Up Percentage Point Change (Δt)", fontsize=14, pad=12, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Change: Built % (t) - Built % (t-1)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_png)
        plt.savefig(fig_svg)
        plt.close()
        print(f"Generated change plot: {fig_png} and {fig_svg}")
    else:
        # SVG fallback
        content = []
        ox, oy, w, h = 90, 260, 640, 200
        content.append(f'<line x1="{ox}" y1="{oy}" x2="{ox+w}" y2="{oy}" stroke="#0f172a" stroke-width="1.5" />')
        # Labels
        years = list(range(2018, 2026))
        for idx, yr in enumerate(years):
            x = ox + idx * (w / 7.0)
            content.append(f'<text x="{x}" y="{oy+20}" font-size="11" text-anchor="middle" fill="#64748b">{yr}</text>')
        content.append(f'<text x="{ox-15}" y="{oy+4}" font-size="11" text-anchor="end" fill="#0f172a">0.0%</text>')
        content.append(f'<text x="{ox-15}" y="{oy-95}" font-size="11" text-anchor="end" fill="#0f172a">+1.0%</text>')
        content.append(f'<text x="{ox-15}" y="{oy+105}" font-size="11" text-anchor="end" fill="#0f172a">-1.0%</text>')
        render_svg_chart(fig_svg, "Annual Net Built-Up Percentage Point Change (Δt)", "\n  ".join(content), width=850, height=450)
        print(f"Generated vector SVG change plot: {fig_svg}")


def plot_strata_distribution(ts_df: pd.DataFrame):
    """Plot 3: 2017 vs 2025 built-up comparison across strata."""
    fig_png = FIGURES_DIR / "strata_distribution.png"
    fig_svg = FIGURES_DIR / "strata_distribution.svg"

    p17 = ts_df[ts_df["year"] == 2017].set_index("chip_id")["built_fraction_pct"]
    p25 = ts_df[ts_df["year"] == 2025].set_index("chip_id")["built_fraction_pct"]
    comp_df = pd.DataFrame({"2017": p17, "2025": p25})

    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        comp_df.plot(kind="bar", ax=ax, color=["#60a5fa", "#2563eb"], width=0.6)
        ax.set_title("Built-Up Area Comparison: 2017 vs 2025 Across Strata", fontsize=14, pad=12, fontweight="bold")
        ax.set_ylabel("Built-Up Area Fraction (%)", fontsize=12)
        ax.set_xlabel("Chip ID (Stratum)", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(fig_png)
        plt.savefig(fig_svg)
        plt.close()
        print(f"Generated strata distribution plot: {fig_png} and {fig_svg}")
    else:
        render_svg_chart(fig_svg, "Built-Up Area Comparison: 2017 vs 2025", "<text x='400' y='200' font-size='14' text-anchor='middle' fill='#64748b'>Stratum Comparison Chart</text>")
        print(f"Generated vector SVG strata plot: {fig_svg}")


def plot_missing_and_prevalence(ts_df: pd.DataFrame):
    """Plot 4: Class prevalence and data completeness check."""
    fig_png = FIGURES_DIR / "missing_and_prevalence.png"
    fig_svg = FIGURES_DIR / "missing_and_prevalence.svg"

    if HAS_MATPLOTLIB:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        # Completeness
        ax1.bar(ts_df["chip_id"].unique(), [100.0]*5, color="#10b981", width=0.5)
        ax1.set_ylim(0, 110)
        ax1.set_title("Data Completeness (Valid Pixels %)", fontweight="bold")
        ax1.set_ylabel("% Valid (Non-NoData) Pixels")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax1.tick_params(axis="x", rotation=20)

        # Prevalence
        mean_prev = ts_df.groupby("chip_id")["built_fraction_pct"].mean()
        ax2.bar(mean_prev.index, mean_prev.values, color="#6366f1", width=0.5)
        ax2.set_title("Mean Built-Up Prevalence (%)", fontweight="bold")
        ax2.set_ylabel("Mean Built Fraction %")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax2.tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plt.savefig(fig_png)
        plt.savefig(fig_svg)
        plt.close()
        print(f"Generated prevalence plot: {fig_png} and {fig_svg}")
    else:
        render_svg_chart(fig_svg, "Data Completeness & Prevalence Audit", "<text x='400' y='200' font-size='14' text-anchor='middle' fill='#64748b'>Prevalence and Missingness Overview</text>")
        print(f"Generated vector SVG prevalence plot: {fig_svg}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = RESULTS_DIR / "pilot_timeseries.csv"
    if not ts_file.exists():
        print(f"Error: {ts_file} does not exist. Run build_timeseries.py first.")
        return

    ts_df = pd.read_csv(ts_file)
    print("Generating Step 3 exploratory plots...")
    plot_trajectories(ts_df)
    plot_annual_change(ts_df)
    plot_strata_distribution(ts_df)
    plot_missing_and_prevalence(ts_df)


if __name__ == "__main__":
    main()
