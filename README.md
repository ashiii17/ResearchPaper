# Sentinel-2 10m Land Use/Land Cover (LULC) Time Series — India

This repository contains the complete dataset, metadata, and processing pipeline for downloading and processing annual 10-meter Land Use/Land Cover tiles for **India** from the Esri / Impact Observatory / Microsoft Sentinel-2 10m Land Cover Time Series.

---

## 1. Dataset Overview & Attribution

- **Dataset Name:** Sentinel-2 10m Land Use/Land Cover Time Series
- **Creators:** Esri, Impact Observatory, and Microsoft
- **Source Resolution:** Native 10 meters ($10\text{ m} \times 10\text{ m}$ per pixel)
- **Coordinate Reference System:** Native Universal Transverse Mercator (UTM WGS84 zones 42N to 47N)
- **Available Years:** **2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025**
- **Authoritative Source Endpoints:**
  - **ArcGIS Living Atlas (Primary, 2017–2025):** `https://lulctimeseries.blob.core.windows.net/lulctimeseriesv003`
    - Years 2024+: `lc{year}/{tile}_{year}0101-{year}1231.tif`
    - Years <2024: `lc{year}/{tile}_{year}0101-{year+1}0101.tif`
  - **AWS Open Data Registry (Fallback, 2017–2024 only):** `https://io-10m-annual-lulc.s3.amazonaws.com/{tile}_{year}.tif`
- **License & Attribution:**
  > *This data is provided under the Creative Commons Attribution 4.0 International License (CC BY 4.0).*  
  > **Attribution:** "Esri, Impact Observatory, and Microsoft - Sentinel-2 10m Land Use/Land Cover Time Series".

---

## 2. Land Cover Class Legend

The original 9-class integer values and definitions are strictly preserved without reclassification or resampling:

| Class Value | Class Name | Description |
|:-----------:|:-----------|:------------|
| **0** | **NoData** | Outside raster extent / unclassified |
| **1** | **Water** | Rivers, lakes, oceans, ponds, coastal waters |
| **2** | **Trees** | Dense clustering of tall ($\ge 15\text{ ft}$) canopy forest |
| **4** | **Flooded vegetation** | Mangroves, salt marshes, inundated wetlands |
| **5** | **Crops** | Human planted cereals, grasses, croplands |
| **7** | **Built Area** | Impervious surfaces, human settlements, roads, urban infrastructure |
| **8** | **Bare ground** | Sand dunes, exposed rock, salt flats, deserts |
| **9** | **Snow/Ice** | Permanent snow, glaciers, alpine ice sheets |
| **10** | **Clouds** | Cloud cover / persistent obstruction |
| **11** | **Rangeland** | Meadows, grasslands, scrubland, and savannahs |

---

## 3. Corrected Tile List (17 UTM Grid Tiles)

The tile list is geometrically derived from UTM zone longitudes $(n-31) \times 6^\circ$ to $(n-30) \times 6^\circ$ and 8-degree latitude bands ($N=0\text{--}8^\circ\text{N}$, $P=8\text{--}16^\circ\text{N}$, $Q=16\text{--}24^\circ\text{N}$, $R=24\text{--}32^\circ\text{N}$, $S=32\text{--}40^\circ\text{N}$, $T=40\text{--}48^\circ\text{N}$), intersected with `india_boundary.geojson`:

| Tile | Longitude | Latitude | India Area (sq deg) | Status | Covered Indian Region |
|:---:|:---:|:---:|:---:|:---:|:---|
| **42Q** | 66–72°E | 16–24°N | 8.433 | Valid (200 OK) | Gujarat coast, Saurashtra, Kutch |
| **42R** | 66–72°E | 24–32°N | 6.969 | Valid (200 OK) | Western Rajasthan, Kutch border |
| **43P** | 72–78°E | 8–16°N | 21.178 | Valid (200 OK) | Kerala, Tamil Nadu, Lakshadweep, Karnataka |
| **43Q** | 72–78°E | 16–24°N | 42.476 | Valid (200 OK) | Maharashtra, Goa, Madhya Pradesh |
| **43R** | 72–78°E | 24–32°N | 41.864 | Valid (200 OK) | Delhi, NCR, Haryana, Punjab, Rajasthan, UP |
| **43S** | 72–78°E | 32–40°N | 10.419 | Valid (200 OK) | Himachal Pradesh, Jammu & Kashmir, Western Ladakh |
| **44P** | 78–84°E | 8–16°N | 13.323 | Valid (200 OK) | Tamil Nadu, Andhra Pradesh, Kanyakumari |
| **44Q** | 78–84°E | 16–24°N | 44.780 | Valid (200 OK) | Telangana, Andhra Pradesh, Odisha, Chhattisgarh |
| **44R** | 78–84°E | 24–32°N | 31.675 | Valid (200 OK) | UP, Bihar, northern MP, Nepal border |
| **44S** | 78–84°E | 32–40°N | 2.553 | Valid (200 OK) | Eastern Ladakh, Aksai Chin border |
| **45Q** | 84–90°E | 16–24°N | 17.666 | Valid (200 OK) | Odisha, West Bengal, Sundarbans |
| **45R** | 84–90°E | 24–32°N | 14.065 | Valid (200 OK) | West Bengal, Sikkim, western Assam, Meghalaya |
| **46N** | 90–96°E | 0–8°N | 0.105 | Valid (200 OK) | Nicobar Islands & Indira Point (6.75°N) |
| **46P** | 90–96°E | 8–16°N | 0.525 | Valid (200 OK) | Andaman Islands & Port Blair (11.62°N) |
| **46Q** | 90–96°E | 16–24°N | 2.356 | Valid (200 OK) | Mizoram, Tripura, southern Manipur |
| **46R** | 90–96°E | 24–32°N | 18.648 | Valid (200 OK) | Assam, Arunachal Pradesh, Nagaland, Meghalaya |
| **47R** | 96–102°E | 24–32°N | 1.668 | Valid (200 OK) | Far-eastern Arunachal Pradesh (Kibithu up to 97.42°E) |

### Non-India / Ocean Tiles Excluded:
- **`42S`**: 0.00 sq deg intersection (Pakistan/Afghanistan only). Deleted from disk.
- **`44N`**: 0.00 sq deg intersection (Sri Lanka and Indian Ocean; Kanyakumari starts at 8.08°N in Band P).
- **`45P`**: 0.00 sq deg intersection (open ocean in central Bay of Bengal; Azure returns 404).
- **`47N`**: 0.00 sq deg intersection (Sumatra, Indonesia / Malacca Strait).
- **`47P`**: 0.00 sq deg intersection (Thailand / Myanmar / Andaman Sea).

---

## 4. Benchmark Coordinates Spot Checks

| Location | Coordinates | Tile | 2017 Class | 2025 Class |
|:---|:---:|:---:|:---|:---|
| **Delhi** | 77.2090°E, 28.6140°N | `43R` | **Class 7: Built Area** | **Class 7: Built Area** |
| **Bengaluru** | 77.5946°E, 12.9716°N | `43P` | **Class 7: Built Area** | **Class 7: Built Area** |
| **Hyderabad** | 78.4867°E, 17.3850°N | `44Q` | **Class 7: Built Area** | **Class 7: Built Area** |
| **Port Blair** | 92.7265°E, 11.6234°N | `46P` | **Class 7: Built Area** | **Class 7: Built Area** |
| **Itanagar** | 93.6200°E, 27.0800°N | `46R` | **Class 2: Trees** | **Class 2: Trees** |

---

## 5. Storage & Disk Safety

- **Size per Year (17 tiles):** `2.20 GiB` (`2.35 GB` on disk).
- **Projected Total Size (9 years, 2017–2025):** `19.8 GiB` (`21.2 GB`).
- **Disk Safety Limit:** Maintained above `8.0 GB` free at all times.
