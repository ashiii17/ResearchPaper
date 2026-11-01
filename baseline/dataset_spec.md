# Satellite Urban-Change Dataset Specification
**Project:** Nationwide India Urban-Change Research (2017–2025)  
**Study Area:** India (1,500 stratified 5 km × 5 km chips across 17 UTM tiles)  
**Sensor:** Sentinel-2 L2A Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`)  
**Label Source:** Esri 10 m Annual Land Use / Land Cover (2017–2025)

---

## 1. Input Features: Sentinel-2 Annual Imagery

Each chip is a $500 \times 500$ pixel raster at 10-meter native spatial resolution ($5\text{ km} \times 5\text{ km}$) in native UTM projection (EPSG:32642–32646).

### 1.1 Spectral Channels (Bands 1–6)
Annual surface reflectance values are computed as the pixel-wise median of cloud-masked Level-2A observations over each calendar year (Jan 1 to Dec 31):
- **Band 1 (B2)**: Blue ($490\text{ nm}$, $10\text{ m}$) — atmospheric scattering, water, built contrast
- **Band 2 (B3)**: Green ($560\text{ nm}$, $10\text{ m}$) — vegetation peak reflectance
- **Band 3 (B4)**: Red ($665\text{ nm}$, $10\text{ m}$) — chlorophyll absorption, built-up contrast
- **Band 4 (B8)**: NIR ($842\text{ nm}$, $10\text{ m}$) — biomass structure, high vegetation reflectance
- **Band 5 (B11)**: SWIR 1 ($1610\text{ nm}$, $20\text{ m}$ resampled to $10\text{ m}$) — moisture, impervious surfaces, soil
- **Band 6 (B12)**: SWIR 2 ($2190\text{ nm}$, $20\text{ m}$ resampled to $10\text{ m}$) — mineralogy, urban impervious differentiation

*Storage & Scale:* Stored as `uint16` with scale factor $10,000$ (reflectance in $[0, 1.0]$ maps to integer $[0, 10,000]$). Value $0$ indicates unobserved/missing data (NoData).

### 1.2 Data Quality & Observation Channel (Band 7: `n_obs`)
- **Band 7 (`n_obs`)**: Observation count band (`uint16`).
- Counts the number of clear, cloud-free, shadow-free Sentinel-2 observations passing QA60 bitmask (bits 10 & 11) and SCL filtering (excluding classes 3, 8, 9, 10) during the year.
- **Quality Rule**: If $n_{\text{obs}} == 0$, the pixel had zero cloud-free observations during the entire year; all spectral bands are unmasked to $0$, and the pixel **must be masked out** (`ignore_index = 255`) during loss calculation and metric computation.
- **Feature Use**: $n_{\text{obs}}$ can also be normalized ($n_{\text{obs}} / 100.0$) and supplied as an explicit observation-reliability feature to the model.

---

## 2. Label Derivation: Esri 10 m LULC Stack

For each chip, `labels/<chip_id>.tif` contains 9 annual bands matching calendar years 2017 through 2025:
- Band 1: `esri_2017`
- Band 2: `esri_2018`
- ...
- Band 9: `esri_2025`
- Band index formula: $\text{band\_idx} = \text{year} - 2016$.

### Esri Classes
- **Class 7**: Built Area (structures, paved surfaces, residential, commercial, industrial).
- **Class 0**: NoData / Unclassified. Must be ignored in all training loss and evaluation.
- **Other Classes**: 1 (Water), 2 (Trees), 4 (Flooded Vegetation), 5 (Crops), 8 (Bare Ground), 9 (Snow/Ice), 10 (Clouds), 11 (Rangeland).

---

## 3. Separation of Tasks & Target Formulations

To prevent methodological conflation, this project strictly decouples three distinct prediction tasks:

### Task A: Single-Year Built-Up Segmentation
- **Objective**: Given Sentinel-2 imagery $X_t \in \mathbb{R}^{C \times H \times W}$ at year $t$, predict the binary spatial segmentation of built-up areas $Y_t \in \{0, 1\}^{H \times W}$.
- **Ground Truth Target**:
  $$Y_t(i, j) = \begin{cases} 1 & \text{if } \text{Esri}_t(i, j) == 7 \\ 0 & \text{if } \text{Esri}_t(i, j) \in \{1, 2, 4, 5, 8, 9, 10, 11\} \\ 255 & \text{if } \text{Esri}_t(i, j) == 0 \text{ or } n_{\text{obs}, t}(i, j) == 0 \quad (\text{ignore})\end{cases}$$
- **Loss**: Binary Cross-Entropy with Logits or Focal Loss computed strictly on valid pixels where $Y_t \neq 255$.

### Task B: One-Year Binary Urban Change Detection
- **Objective**: Given paired bi-temporal Sentinel-2 imagery $(X_t, X_{t+1})$, detect newly developed urban land transitioning from non-built to built between consecutive years $t$ and $t+1$.
- **Change Target**:
  $$\Delta Y_{t \rightarrow t+1}(i, j) = \begin{cases} 
  1 & \text{if } \text{Esri}_t(i, j) \neq 7 \text{ and } \text{Esri}_{t+1}(i, j) == 7 \quad (\text{Newly Built}) \\
  0 & \text{if } \text{No transition to built (stable non-built or stable built)} \\
  255 & \text{if } \text{Esri}_t(i, j) == 0 \text{ or } \text{Esri}_{t+1}(i, j) == 0 \text{ or } n_{\text{obs}, t}(i, j) == 0 \text{ or } n_{\text{obs}, t+1}(i, j) == 0
  \end{cases}$$
- **Extreme Class Imbalance**: Annual new urban growth typically represents $< 0.5\%$ of pixels in a chip. Masked evaluation against persistence baseline is mandatory.

### Task C: Spatiotemporal Urban-Change Forecasting
- **Objective**: Given a historical sequence of annual observations $(X_{t-k}, \dots, X_t)$ and historical labels $(Y_{t-k}, \dots, Y_t)$, forecast future urban built-up fraction $f_{t+1} \in [0, 1]$ or future spatial mask $Y_{t+1}$.
- **Chip-Level Target**:
  $$f_{t+1} = \frac{\sum_{(i, j) \in \text{valid}} \mathbb{I}(\text{Esri}_{t+1}(i, j) == 7)}{N_{\text{valid}}}$$
- **Constraint**: Must use causal temporal masking; future years $t > t_{\text{obs}}$ are strictly forbidden from input tensors.

---

## 4. Limitations of Esri-Derived Pseudo-Labels

Researchers must acknowledge and document the following critical limitations when training deep models on Esri 10 m LULC:
1. **Model-Generated Pseudo-Labels**: Esri 10 m maps are produced by an ensemble deep learning model (Impact Observatory / Microsoft AI for Earth) applied to Sentinel-2, not human-audited ground truth.
2. **Label Noise & Temporal Flickering**: Unconstrained annual inference can exhibit spurious transitions (e.g. built $\rightarrow$ non-built $\rightarrow$ built) caused by seasonal sun angle, dry/wet soil reflectance shifts, or crop phenology confusion.
3. **Fringe & Peri-Urban Bias**: Rural settlements, unpaved roads, and low-density tin/clay roofs have lower classification certainty in the Esri product compared to high-density concrete cores.
4. **Evaluation Benchmark Role**: While suitable for large-scale pretraining and architectural benchmarking across India's diverse agro-climatic zones, definitive claims of real-world change detection accuracy require validation against high-resolution aerial/commercial imagery (e.g. WorldView/Planet) or verified cadastral ground truth.
