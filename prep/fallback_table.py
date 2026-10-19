"""Precompute the "recorded brain" table used when a browser has no WebGPU.

A grid over the four taste rates; each cell holds the outcome of 30 reference trials of 250 ms.
The app interpolates between cells and labels the result "recorded brain" instead of "live brain".

    python prep/fallback_table.py      # writes web/public/brain/recorded.json
"""

from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import refsim

SUGAR = [0, 30, 60, 90, 120, 150, 180, 210]  # up to 150 Hz × the "starving" gain of 1.4
OTHER = [0, 37.5, 75, 112.5, 150]
TRIALS, T_MS = 30, 250.0


def main():
    grid = list(itertools.product(SUGAR, OTHER, OTHER, OTHER))
    jobs = [(f"{s}/{b}/{l}/{w}", {"sugar": s, "bitter": b, "salt": l, "water": w}, TRIALS, T_MS, 50_000 + i)
            for i, (s, b, l, w) in enumerate(grid)]
    refsim._init()
    t0 = time.time()
    with ProcessPoolExecutor(14, initializer=refsim._init) as ex:
        results = list(ex.map(refsim.run_condition, jobs, chunksize=4))
    out = {
        "axes": {"sugar": SUGAR, "bitter": OTHER, "salt": OTHER, "water": OTHER},
        "order": "sugar, bitter, salt, water (last varies fastest)",
        "trials": TRIALS,
        "t_ms": T_MS,
        "ate": [r["ate"] for r in results],
        "mn9_hz": [round(r["mn9_hz_mean"], 1) for r in results],
        "fled": [r["dnp01_trials"] for r in results],
    }
    (refsim.BRAIN / "recorded.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(grid)} cells in {time.time() - t0:.0f} s; ate>0 in {np.count_nonzero(out['ate'])}")


if __name__ == "__main__":
    main()
