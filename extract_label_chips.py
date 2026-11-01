#!/usr/bin/env python3
"""
Cut a 9-year Esri label stack (2017..2025) for every chip in chips.csv.

Run from the india_lulc folder (needs tiles/ and chips.csv). Reads tiles only.
    python3 extract_label_chips.py              # all chips
    python3 extract_label_chips.py --limit 20   # quick test

Output:
  labels/<chip_id>.tif    9 bands (2017..2025), uint8, 500 x 500 px, 10 m, deflate-compressed,
                          same CRS and pixel grid as the Sentinel-2 chips exported to Drive
  labels_manifest.csv     per chip: built fraction (class 7) per year + number of NoData pixels
At the end it re-computes the 2017 and 2025 built fractions and compares them with chips.csv
(they must match exactly); a mismatch means the window is misaligned.
Expected size: roughly 1 GB or less for 1,500 chips (rough estimate).
After this you no longer need the 153 raw tiles to build the training set, so they can be
moved to an external drive or Google Drive to free space.
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

YEARS = list(range(2017, 2026))
CHIP_PX = 500
BUILT = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chips", default="chips.csv")
    args = ap.parse_args()

    chips = pd.read_csv(args.chips)
    if args.limit:
        chips = chips.head(args.limit)
    os.makedirs("labels", exist_ok=True)
    rows = []

    for tile, grp in chips.groupby("tile"):
        srcs = {y: rasterio.open(f"tiles/{y}/{tile}_{y}.tif") for y in YEARS}
        t = srcs[2017].transform
        for y in YEARS:
            assert srcs[y].transform == t and srcs[y].shape == srcs[2017].shape, \
                f"{tile}: grid differs between years"
        for r in grp.itertuples():
            col, row = (r.x0 - t.c) / t.a, (t.f - r.y1) / -t.e
            if col != int(col) or row != int(row):
                raise SystemExit(f"{r.chip_id}: chip origin is not on the tile's 10 m grid")
            win = Window(int(col), int(row), CHIP_PX, CHIP_PX)
            stack = np.stack([srcs[y].read(1, window=win) for y in YEARS])
            assert stack.shape == (len(YEARS), CHIP_PX, CHIP_PX), f"{r.chip_id}: bad shape {stack.shape}"
            path = f"labels/{r.chip_id}.tif"
            with rasterio.open(path, "w", driver="GTiff", height=CHIP_PX, width=CHIP_PX,
                               count=len(YEARS), dtype="uint8", crs=f"EPSG:{int(r.epsg)}",
                               transform=from_origin(r.x0, r.y1, 10, 10), compress="deflate",
                               nodata=0) as dst:
                dst.write(stack)
                dst.descriptions = [f"esri_{y}" for y in YEARS]
            rec = {"chip_id": r.chip_id, "nodata_px": int((stack == 0).sum())}
            for i, y in enumerate(YEARS):
                rec[f"built_{y}"] = float((stack[i] == BUILT).mean())
            rows.append(rec)
        for s in srcs.values():
            s.close()
        print(f"{tile}: {len(grp)} chips")

    man = pd.DataFrame(rows)
    man.to_csv("labels_manifest.csv", index=False)

    # consistency check against the sampler's own numbers
    m = man.merge(chips[["chip_id", "built17_frac", "built25_frac"]], on="chip_id")
    d17 = (m.built_2017 - m.built17_frac).abs().max()
    d25 = (m.built_2025 - m.built25_frac).abs().max()
    size_mb = sum(os.path.getsize(f"labels/{c}.tif") for c in man.chip_id) / 1e6
    print(f"wrote {len(man)} label chips, {size_mb:.0f} MB total")
    print(f"max |built fraction difference| vs chips.csv: 2017 {d17:.2e}, 2025 {d25:.2e}")
    print(f"chips with any NoData pixel: {(man.nodata_px > 0).sum()}")
    if max(d17, d25) > 1e-9:
        print("WARNING: fractions do not match chips.csv - the windows are probably misaligned")


if __name__ == "__main__":
    main()
