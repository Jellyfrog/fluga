"""Phase 5: judge WebGPU validation results against reference.json. Exits non-zero on any miss.

    python prep/compare.py web/tools/out/validation-active.json [more.json ...]

Criteria (from the plan):
  1. Sugar curve: GPU mean MN9 rate at 20/50/100/150 Hz sugar lies inside the 95% interval of the
     reference trials, including silence at 20 Hz.  (Also reported: a Welch z-score of the means.)
  2. Known effects: bitter and Ir94e input lower sugar-driven MN9; water input alone drives MN9.
  3. Active set: plain and active runs with the same seed agree (they should be bit-identical,
     because skipping neurons at exact rest is exact).
  4. No input, no spikes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF = {c["name"]: c for c in json.loads((ROOT / "web/public/brain/reference.json").read_text())["conditions"]}


def hz(counts, t_ms):
    return [c * 1e3 / t_ms for c in counts]


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / max(1, len(x) - 1))


def main(paths):
    runs = []
    for p in paths:
        runs += json.loads(Path(p).read_text())["results"]
    by = {}
    for r in runs:
        by.setdefault((r["name"], r["mode"]), r)
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {label}  {detail}")

    def gpu(name):
        r = by.get((name, "active")) or by.get((name, "plain"))
        return r, mean(hz(r["mn9"], r["t_ms"])) if r else None

    print("condition                                  GPU MN9 Hz        ref MN9 Hz [95% of trials]   z     ate gpu/ref")
    for name, ref in REF.items():
        r, m = gpu(name)
        if r is None:
            continue
        g = hz(r["mn9"], r["t_ms"])
        se = math.sqrt(sd(g) ** 2 / len(g) + ref["mn9_hz_sd"] ** 2 / ref["trials"]) or 1.0
        z = (m - ref["mn9_hz_mean"]) / se
        print(f"{name:42s} {m:6.1f} ± {sd(g):5.1f}    {ref['mn9_hz_mean']:6.1f} ± {ref['mn9_hz_sd']:5.1f} "
              f"[{ref['mn9_hz_p025']:5.1f}, {ref['mn9_hz_p975']:5.1f}]  {z:+5.2f}   {r['ate']:2d}/{ref['ate']:2d}")
        if any(x["overflow"] for x in [r]):
            check(f"{name}: no buffer overflow", False)

    print()
    for lvl in (20, 50, 100, 150):
        name = f"sugar_{lvl}"
        r, m = gpu(name)
        if r is None:
            print(f"skip  sugar curve {lvl:3d} Hz  not in these results")
            continue
        ref = REF[name]
        inside = ref["mn9_hz_p025"] <= m <= ref["mn9_hz_p975"]
        if lvl == 20:
            inside &= m == 0
        check(f"sugar curve {lvl:3d} Hz", inside, f"GPU {m:.1f} Hz, ref interval [{ref['mn9_hz_p025']}, {ref['mn9_hz_p975']}]")

    s150 = gpu("sugar_150")[1]
    for mix, label in (("sugar_150+bitter_100", "bitter lowers sugar-driven MN9"),
                       ("sugar_150+salt_150", "Ir94e lowers sugar-driven MN9")):
        m = gpu(mix)[1]
        if m is not None and s150 is not None:
            check(label, m < s150, f"{s150:.1f} → {m:.1f} Hz")
    w = gpu("water_150")[1]
    if w is not None:
        check("water alone drives MN9", w > 0, f"water 150 Hz → {w:.1f} Hz")

    pairs = [(n, by[(n, "plain")], by[(n, "active")]) for (n, mode) in by if mode == "plain" and (n, "active") in by]
    for n, p, a in pairs:
        same = p["mn9"] == a["mn9"] and p["spikes"] == a["spikes"]
        check(f"active set == plain: {n}", same,
              "bit-identical" if same else f"spikes plain {sum(p['spikes'])} vs active {sum(a['spikes'])}")
    if not pairs:
        print("note  no plain/active pairs in these files")

    r = by.get(("silent", "active")) or by.get(("silent", "plain"))
    if r:
        check("no input, no spikes", sum(r["spikes"]) == 0, f"{sum(r['spikes'])} spikes")

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
