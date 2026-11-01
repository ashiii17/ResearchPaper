# Sentinel-2 India Urban-Change PyTorch Baseline Pipeline

A reproducible, leakage-safe deep learning training and evaluation baseline for satellite urban-change research across India (2017–2025).

---

## 1. Directory Structure

```
baseline/
├── README.md               # Pipeline architecture & operational documentation
├── dataset_spec.md         # Detailed task, target, and data quality specifications
├── audit_splits.py         # Split integrity and data leakage audit script
├── split_audit.txt         # Machine-generated audit report for 1,500 chips
├── dataset.py              # Lazy rasterio-backed PyTorch Dataset and DataLoader
├── models.py               # CompactUNet, ChangeDetectionUNet, and masked losses
├── train_baseline.py       # Deterministic training engine & smoke test runner
├── evaluate.py             # Masked metrics (IoU, F1, Precision, Recall, Confusion Matrix)
├── pilot_smoke_test.txt    # Results of the 45-raster pilot smoke test
└── tests/                  # Unit and integration test suite
    ├── __init__.py
    ├── run_all_tests.py    # Test runner executing all tests
    ├── test_dataset.py     # Tests for lazy loading, shapes, dtypes, masking
    ├── test_splits.py      # Tests for disjoint splits and strata coverage
    ├── test_models.py      # Tests for reproducible seeds and model specs
    └── test_metrics.py     # Tests for masked metric computation and baselines
```

---

## 2. Core Methodology & Tasks

The pipeline strictly decouples three distinct learning tasks to prevent methodological conflation:

1. **Task A: Single-Year Built-Up Semantic Segmentation**
   - Inputs: 6 annual median surface reflectance bands (`B2, B3, B4, B8, B11, B12`) + normalized observation count channel (`n_obs / 100.0`) $\rightarrow$ $(7, H, W)$.
   - Target: Built Area ($1$) vs. Non-Built ($0$) derived from Esri 10 m class 7.
   - Masking: Class 0 (NoData) and pixels where $n_{\text{obs}} == 0$ are assigned `ignore_index = 255`.

2. **Task B: One-Year Urban Change Detection ($t \rightarrow t+1$)**
   - Inputs: Bi-temporal concatenation of consecutive annual composites + absolute spectral difference $\Delta = |X_{t+1} - X_t|$ $\rightarrow$ $(18, H, W)$.
   - Target: Newly developed urban land ($1$) transitioning from non-built to built between $t$ and $t+1$.
   - Masking: Any pixel invalid in year $t$ or year $t+1$ is assigned `ignore_index = 255`.

3. **Task C: Spatiotemporal Urban-Change Forecasting**
   - Historical multi-year sequence forecasting future built-up spatial masks or chip-level built fractions.
   - Strictly enforces causal temporal masking to prevent future data leakage.

---

## 3. Leakage Prevention Guardrails

- **Training-Only Normalization**: Spectral channels are standardized strictly using mean and standard deviation computed on the training split:
  $$\text{Mean} = [1353, 1540, 1690, 2850, 2450, 1850], \quad \text{Std} = [650, 700, 850, 1100, 1000, 900]$$
- **Disjoint Spatial Splits**: Verified zero chip ID overlap across train ($1,033$ chips, $68.9\%$), validation ($147$ chips, $9.8\%$), and test ($320$ chips, $21.3\%$).
- **Test Quarantine**: The test split is quarantined and never used for hyperparameter tuning, threshold selection, or normalization.
- **Pilot Statistical Boundary**: The 5-chip pilot contains only 4 training chips, 1 validation chip, and 0 test chips. It is certified strictly for software integrity and pipeline smoke testing, not statistical model ranking.

---

## 4. Usage Instructions

### Run Split Audit
```bash
./.venv/bin/python baseline/audit_splits.py
```

### Run Full Test Suite
```bash
./.venv/bin/python baseline/tests/run_all_tests.py
```

### Run Pilot Smoke Test
```bash
./.venv/bin/python baseline/train_baseline.py --smoke-test
```

### Model Training (when nationwide chips are available)
```bash
# Task A: Segmentation
./.venv/bin/python baseline/train_baseline.py --task segmentation --epochs 10 --batch-size 8 --patch-size 256

# Task B: Change Detection
./.venv/bin/python baseline/train_baseline.py --task change --epochs 10 --batch-size 4 --patch-size 256
```
