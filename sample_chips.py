#!/usr/bin/env python3
"""
Draw a stratified, India-wide sample of 5 km x 5 km chips from the Esri tiles.

Why: Sentinel-2 imagery for ALL of India at 10 m for 9 years is roughly 4 TB (rough
estimate), which is not practical. The Esri label maps already cover all of India, so
this script chooses where to export Sentinel-2 imagery so the sample is spread across
every tile and every kind of landscape.

Reads tiles only (never modifies them). Run from the india_lulc folder:
    python3 sample_chips.py --tiles 43P          # quick test on one tile first
    python3 sample_chips.py                      # all tiles (can take 1-3 hours; cached)
Output: chips.csv  (upload it to Google Drive for the Colab exporter)

Strata (fractions are of the 500x500 px cell):
  R random        any eligible cell        -> use these for the UNBIASED test numbers
  A growth        net built gain 2017->2025 >= 2% of the cell
  B urban_stable  built in 2017 >= 20% and growth < 2%
  C low_built     built in 2025 >= 1%, not A or B
  D empty         built < 0.1% in both years (negatives)
Selection by Esri growth inherits Esri's noise, so report headline results on stratum R.

Split: whole 50 km blocks go to train/val/test (70/10/20 by hash) so neighbouring
chips never straddle splits.
"""
import argparse
import glob
import json
import os
import zlib

import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import unary_union

CELL_PX = 500                     # 500 px x 10 m = 5 km
MIN_VALID_FRAC = 0.98             # skip coastal/edge cells with NoData
BUILT, NODATA = 7, 0
TILES_DIR = "tiles"
BOUNDARY = "india_boundary.geojson"
CACHE_DIR = "cell_cache"
DEFAULT_COUNTS = {"R": 400, "A": 500, "B": 250, "C": 250, "D": 100}


def tile_epsg(tile):
    return 32600 + int(tile[:2])  # all India tiles are in the northern hemisphere


def scan_tile(tile):
    """Per-cell built counts for 2017 and 2025 (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{tile}.csv.gz")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    p17 = os.path.join(TILES_DIR, "2017", f"{tile}_2017.tif")
    p25 = os.path.join(TILES_DIR, "2025", f"{tile}_2025.tif")
    rows = []
    with rasterio.open(p17) as a, rasterio.open(p25) as b:
        assert a.shape == b.shape and a.transform == b.transform, f"grid mismatch {tile}"
        H, W = a.shape
        nr, nc = H // CELL_PX, W // CELL_PX
        t = a.transform
        for r in range(nr):
            win = Window(0, r * CELL_PX, nc * CELL_PX, CELL_PX)
            x = a.read(1, window=win)
            y = b.read(1, window=win)

            def agg(m):
                return m.reshape(CELL_PX, nc, CELL_PX).sum(axis=(0, 2), dtype=np.int64)

            valid = agg((x != NODATA) & (y != NODATA))
            b17 = agg(x == BUILT)
            b25 = agg(y == BUILT)
            for c in np.nonzero(valid)[0]:
                rows.append((tile, r, int(c), t.c + c * CELL_PX * t.a, t.f + r * CELL_PX * t.e,
                             int(valid[c]), int(b17[c]), int(b25[c])))
    df = pd.DataFrame(rows, columns=["tile", "r", "c", "x0", "y1", "valid", "b17", "b25"])
    df.to_csv(cache, index=False)
    return df


def in_india(df, geom):
    out = np.zeros(len(df), dtype=bool)
    for tile, idx in df.groupby("tile").groups.items():
        tr = Transformer.from_crs(tile_epsg(tile), 4326, always_xy=True)
        sub = df.loc[idx]
        lon, lat = tr.transform((sub.x0 + CELL_PX * 5).values, (sub.y1 - CELL_PX * 5).values)
        df.loc[idx, "lon"], df.loc[idx, "lat"] = lon, lat
        out[df.index.get_indexer(idx)] = shapely.contains_xy(geom, lon, lat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", help="limit to these tiles (test run)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all stratum counts")
    args = ap.parse_args()

    tiles = args.tiles or sorted({os.path.basename(p)[:3] for p in glob.glob(f"{TILES_DIR}/2017/*.tif")})
    print("tiles:", tiles)
    df = pd.concat([scan_tile(t) for t in tiles], ignore_index=True)

    gj = json.load(open(BOUNDARY))
    geom = unary_union([shape(f["geometry"]) for f in gj["features"]]).simplify(0.005)
    df["lon"] = np.nan
    df["lat"] = np.nan
    df = df[in_india(df, geom)].copy()
    df = df[df.valid >= MIN_VALID_FRAC * CELL_PX ** 2].copy()
    n2 = CELL_PX ** 2
    df["built17_frac"] = df.b17 / n2
    df["built25_frac"] = df.b25 / n2
    df["growth_frac"] = df.built25_frac - df.built17_frac
    print(f"eligible cells inside India: {len(df):,}")

    masks = {
        "A": df.growth_frac >= 0.02,
        "B": (df.built17_frac >= 0.20) & (df.growth_frac < 0.02),
        "C": (df.built25_frac >= 0.01) & (df.growth_frac < 0.02) & (df.built17_frac < 0.20),
        "D": (df.built25_frac < 0.001) & (df.built17_frac < 0.001),
    }
    rng = np.random.default_rng(args.seed)
    picked, taken = [], pd.Index([])
    # R first (random over everything), then the others from what is left
    plan = [("R", pd.Series(True, index=df.index))] + [(k, masks[k]) for k in "ABCD"]
    for name, mask in plan:
        pool = df[mask & ~df.index.isin(taken)]
        n = min(int(DEFAULT_COUNTS[name] * args.scale), len(pool))
        if n < DEFAULT_COUNTS[name] * args.scale:
            print(f"warning: stratum {name} has only {len(pool)} cells, taking {n}")
        pick = pool.sample(n=n, random_state=int(rng.integers(1 << 31)))
        pick = pick.assign(stratum=name)
        taken = taken.union(pick.index)
        picked.append(pick)
    out = pd.concat(picked)

    out["epsg"] = out.tile.map(tile_epsg)
    out["chip_id"] = out.tile + "_r" + out.r.astype(str) + "_c" + out.c.astype(str)
    blk = ((out.x0 // 50000).astype(int).astype(str) + "_" + (out.y1 // 50000).astype(int).astype(str)
           + "_" + out.tile)
    h = blk.map(lambda s: zlib.crc32(s.encode()) % 10)
    out["split"] = np.where(h <= 6, "train", np.where(h == 7, "val", "test"))
    cols = ["chip_id", "tile", "epsg", "x0", "y1", "lon", "lat", "stratum", "split",
            "built17_frac", "built25_frac", "growth_frac"]
    out[cols].to_csv("chips.csv", index=False)
    print(f"wrote chips.csv with {len(out)} chips")
    print(out.groupby(["stratum", "split"]).size().unstack(fill_value=0))
    print(out.groupby("tile").size().to_string())


if __name__ == "__main__":
    main()
