# Time Series Analysis (TSA) Coursework Module
**Project:** Nationwide India Urban-Change Research (2017–2025)  
**Dataset:** 1,500 Stratified 5 km × 5 km Chips across 17 UTM Zones  
**Labels:** Esri 10 m Annual Land Use / Land Cover (9 annual bands, 2017–2025)  
**Sensors:** Sentinel-2 Level-2A Surface Reflectance  
**Module Directory:** `tsa/`

---

## 1. Directory Structure

```
tsa/
├── README.md                   # Complete module documentation & methodology
├── build_timeseries.py         # Step 1 Data Audit & Step 2 Time Series Construction
├── plot_exploratory.py         # Step 3 Exploratory Data Analysis (EDA) visualizations
├── analyze_stationarity.py     # Step 4 Stationarity Testing (ADF and KPSS tests)
├── forecast_models.py          # Classical forecasting models (Naive, Drift, Holt, ARIMA)
├── evaluate_models.py          # Step 5 Chronological Holdout Evaluation (MAE, RMSE, MAPE)
├── results/                    # Machine-readable CSVs and text reports
│   ├── data_audit.txt          # Label raster structural verification report
│   ├── pilot_timeseries.csv    # 45-row chip-year dataset for the 5 pilot chips
│   ├── pilot_timeseries_summary.csv # Annual pooled summary statistics (2017–2025)
│   ├── stationarity_results.csv# ADF & KPSS test statistics, p-values, conclusions
│   ├── forecast_evaluation.csv # Holdout evaluation metrics by chip and model
│   └── tsa_coursework_report.txt # Comprehensive coursework technical report
├── figures/                    # High-resolution PNG and vector SVG plots
│   ├── trajectory_built_fraction.png/.svg # Multi-annual trajectory lines
│   ├── annual_change_growth.png/.svg      # Annual net percentage point differences
│   ├── strata_distribution.png/.svg       # 2017 vs 2025 built-up comparison
│   ├── missing_and_prevalence.png/.svg    # Completeness and prevalence audit
│   └── forecast_comparison_pilot.png/.svg # Actual vs predicted holdout curves
└── tests/
    ├── __init__.py
    └── test_tsa.py             # Unit tests for formulas, masking, ordering, metrics
```

---

## 2. Methodological Framework

### Step 1: Read-Only Data Audit
- **Label Mapping**: Verifies that band index `year - 2016` corresponds to each calendar year from 2017 (Band 1) to 2025 (Band 9) with exact descriptions `('esri_2017'..'esri_2025')`.
- **Class Encoding**: Esri Class 7 (Built Area) confirmed present; Class 0 (NoData) strictly excluded from denominators.

### Step 2: Time Series Formulation
$$\text{built\_fraction\_pct} = \frac{\sum \mathbb{I}(\text{pixel} == 7)}{\sum \mathbb{I}(\text{pixel} \neq 0)} \times 100.0$$
- **Sample vs. Census**: Pooled summaries are sample-based statistics across stratified chips representing distinct landscapes (R, A, B, C, D), not an unweighted census of India.

### Step 3: Exploratory Analysis
- Tracks annual trajectories, year-over-year net percentage point differences $\Delta t$, and strata distributions across high-growth, stable urban, and desert baselines.

### Step 4: Stationarity & Hypothesis Testing
- **ADF Test**: Null hypothesis $H_0$ = Unit Root (Non-stationary). $p < 0.05$ rejects $H_0$.
- **KPSS Test**: Null hypothesis $H_0$ = Stationary / Trend-stationary. $p < 0.05$ rejects $H_0$.
- **Finding**: Urban growth chips (Assam `46R_r101_c74`, Chhattisgarh `44Q_r113_c70`) show confirmed unit-root non-stationarity, requiring first differencing ($d=1$).
- **Holt-Winters Seasonal Model Rejection**: Annual data has period $m=1$ (no sub-annual seasonality). Fitting seasonal Holt-Winters is methodologically invalid on annual data; **Holt's Linear Trend Model** (non-seasonal level + trend) is adopted.

### Step 5: Chronological Holdout Evaluation
- **Chronological Split**: Train horizon 2017–2023 ($N=7$), Test horizon 2024–2025 ($H=2$). Zero temporal shuffling.
- **MAPE Exclusion Rule**: When actual built fraction is near-zero ($< 0.05\%$, e.g. Desert chip), MAPE is mathematically undefined or pathological. MAE and RMSE serve as the primary ground-truth metrics.
- **Model Ranking**: Holt's Linear Trend achieved the lowest average error ($\text{MAE} = 0.4317\%$, $\text{RMSE} = 0.4943\%$), outperforming Naive Persistence ($\text{MAE} = 0.6758\%$) and Drift ($\text{MAE} = 0.6896\%$).

---

## 3. Execution Commands

### Run Data Audit & Build Time Series:
```bash
./.venv/bin/python tsa/build_timeseries.py
```

### Generate Exploratory Plots:
```bash
/opt/anaconda3/bin/python3 tsa/plot_exploratory.py
```

### Run Stationarity Tests (ADF & KPSS):
```bash
/opt/anaconda3/bin/python3 tsa/analyze_stationarity.py
```

### Run Chronological Holdout Forecast Evaluation:
```bash
/opt/anaconda3/bin/python3 tsa/evaluate_models.py
```

### Run Unit Tests:
```bash
./.venv/bin/python -m unittest tsa/tests/test_tsa.py
```
