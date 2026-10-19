"""Spot-check refsim.py against the original Brian2 model (model.py equations, built once).

Runs a few reference conditions with Brian2 and prints MN9 rates next to reference.json.
Agreement is statistical (different random streams), so this checks the mean MN9 rate falls
inside the reference's spread.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, mV, ms, Hz, prefs, defaultclock, seed

ROOT = Path(__file__).resolve().parent.parent
BRAIN = ROOT / "web" / "public" / "brain"
RAW = ROOT / "data" / "brain"
prefs.codegen.target = "cython"

P = dict(v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms)
EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""
CONDS = ["sugar_100", "sugar_150", "sugar_150+bitter_100", "sugar_150+salt_150"]
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
T_RUN = 250 * ms


def main():
    rowptr = np.fromfile(RAW / "rowptr.bin", dtype=np.uint32).astype(np.int64)
    post = np.fromfile(RAW / "post.bin", dtype=np.uint32).astype(np.int64)
    weight = np.fromfile(RAW / "weight.bin", dtype=np.int16).astype(np.float64)
    groups = json.loads((BRAIN / "groups.json").read_text())["groups"]
    ref = {c["name"]: c for c in json.loads((BRAIN / "reference.json").read_text())["conditions"]}
    n = len(rowptr) - 1
    pre = np.repeat(np.arange(n), np.diff(rowptr))
    mn9 = groups["mn9"]["indices"][0]

    for cond in CONDS:  # a fresh network per condition: Brian2 cannot add inputs to a simulated group
        neu = NeuronGroup(n, EQS, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                          refractory="rfc", namespace=P)
        syn = Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms)
        syn.connect(i=pre, j=post)
        syn.w = weight * 0.275 * mV
        mon = SpikeMonitor(neu[mn9:mn9 + 1])
        rates = ref[cond]["rates"]
        neu.v = -52 * mV
        neu.g = 0 * mV
        neu.rfc = 2.2 * ms
        inputs = []
        for ch, r in rates.items():
            if r <= 0:
                continue
            for i in groups[ch]["indices"]:
                neu[i].rfc = 0 * ms
                inputs.append(PoissonInput(neu[i:i + 1], "v", N=1, rate=r * Hz, weight=250 * 0.275 * mV))
        net = Network(neu, syn, mon, *inputs)
        net.store("init")
        counts = []
        for t in range(N_TRIALS):
            net.restore("init")
            seed(10_000 + t)
            before = mon.num_spikes
            net.run(T_RUN)
            counts.append(mon.num_spikes - before)
        hz = np.array(counts) * 1e3 / float(T_RUN / ms)
        r = ref[cond]
        inside = r["mn9_hz_p025"] <= hz.mean() <= r["mn9_hz_p975"]
        print(f"{cond:28s} brian2 MN9 {hz.mean():6.1f} ± {hz.std(ddof=1):5.1f} Hz | refsim {r['mn9_hz_mean']:6.1f} ± "
              f"{r['mn9_hz_sd']:5.1f} [{r['mn9_hz_p025']}, {r['mn9_hz_p975']}] {'OK' if inside else 'MISS'}", flush=True)


if __name__ == "__main__":
    main()
