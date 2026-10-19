"""Reference simulator: event-driven LIF on the binaries written by build_brain.py.

A float64 port of flight-or-bite's `event` backend, which its authors verified spike-for-spike
against Brian2 on the full connectome. Per 0.1 ms step, in Brian2's order:

    1. state update   exact exponential solution, skipped while refractory
    2. threshold      v > v_th and not refractory
    3. synapses       spikes emitted 18 steps ago add g += count * w_syn; refractory targets drop it;
                      then Poisson kicks add 68.75 mV to v
    4. reset          v = v_rst, g = 0 for neurons that fired in step 2

Usage:
    python prep/refsim.py            # writes web/public/brain/reference.json
    python prep/refsim.py --gate     # M0 gate: sugar curve from the reloaded binaries
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent.parent
BRAIN = ROOT / "web" / "public" / "brain"
RAW = ROOT / "data" / "brain"

DT = 0.1
V0 = VRST = -52.0
VTH = -45.0
T_MBR, TAU = 20.0, 5.0
REFRACTORY_STEPS = 22  # 2.2 ms
DELAY = 18  # 1.8 ms
W_SYN = 0.275
KICK = 250 * W_SYN
A_V = math.exp(-DT / T_MBR)
A_G = math.exp(-DT / TAU)
C_G = TAU / (TAU - T_MBR) * (A_G - A_V)

NEVER = -(2**40)


def load_brain():
    rowptr = np.fromfile(RAW / "rowptr.bin", dtype=np.uint32).astype(np.int64)
    post = np.fromfile(RAW / "post.bin", dtype=np.uint32).astype(np.int32)
    weight = np.fromfile(RAW / "weight.bin", dtype=np.int16).astype(np.int32)
    groups = json.loads((BRAIN / "groups.json").read_text())["groups"]
    return rowptr, post, weight, groups


@njit(cache=True)
def simulate(rowptr, post, weight, kick_ptr, kick_neuron, refr, n_steps, watch):
    """Returns (spikes per neuron, spike steps of each watched neuron as (k, step) pairs, active count)."""
    n = rowptr.shape[0] - 1
    v = np.full(n, V0)
    g = np.zeros(n)
    last = np.full(n, NEVER, dtype=np.int64)
    active = np.empty(n, dtype=np.int32)
    is_active = np.zeros(n, dtype=np.bool_)
    n_active = 0
    ring = np.empty((DELAY + 1, n), dtype=np.int32)
    ring_n = np.zeros(DELAY + 1, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int32)
    watch_k = np.empty(1 << 16, dtype=np.int32)
    watch_step = np.empty(1 << 16, dtype=np.int32)
    n_watch = 0
    watch_of = np.full(n, -1, dtype=np.int32)
    for k in range(watch.shape[0]):
        watch_of[watch[k]] = k

    for step in range(n_steps):
        slot = step % (DELAY + 1)
        fired = ring[slot]
        n_fired = 0
        for a in range(n_active):
            i = active[a]
            if step - last[i] >= refr[i]:
                v[i] = V0 + (v[i] - V0) * A_V + g[i] * C_G
                g[i] *= A_G
                if v[i] > VTH:
                    fired[n_fired] = i
                    n_fired += 1
        ring_n[slot] = n_fired
        if step >= DELAY:
            src_slot = (step - DELAY) % (DELAY + 1)
            src = ring[src_slot]
            for k in range(ring_n[src_slot]):
                j = src[k]
                for e in range(rowptr[j], rowptr[j + 1]):
                    t = post[e]
                    if step - last[t] >= refr[t]:
                        g[t] += weight[e] * W_SYN
                        if not is_active[t]:
                            is_active[t] = True
                            active[n_active] = t
                            n_active += 1
        for k in range(kick_ptr[step], kick_ptr[step + 1]):
            t = kick_neuron[k]
            v[t] += KICK
            if not is_active[t]:
                is_active[t] = True
                active[n_active] = t
                n_active += 1
        for k in range(n_fired):
            i = fired[k]
            v[i] = VRST
            g[i] = 0.0
            last[i] = step
            counts[i] += 1
            w = watch_of[i]
            if w >= 0 and n_watch < watch_k.shape[0]:
                watch_k[n_watch] = w
                watch_step[n_watch] = step
                n_watch += 1
    return counts, watch_k[:n_watch], watch_step[:n_watch], n_active


def kick_schedule(channels, n_steps, rng):
    """channels: list of (neuron indices, rate Hz). Bernoulli(rate*dt) per neuron per step."""
    steps_all, neurons_all = [], []
    for neurons, rate in channels:
        if rate <= 0 or len(neurons) == 0:
            continue
        p = rate * DT * 1e-3
        steps, cols = np.nonzero(rng.random((n_steps, len(neurons))) < p)
        steps_all.append(steps)
        neurons_all.append(np.asarray(neurons)[cols])
    if not steps_all:
        return np.zeros(n_steps + 1, np.int64), np.empty(0, np.int32)
    steps = np.concatenate(steps_all)
    neurons = np.concatenate(neurons_all).astype(np.int32)
    order = np.argsort(steps, kind="stable")
    kick_ptr = np.zeros(n_steps + 1, dtype=np.int64)
    np.cumsum(np.bincount(steps, minlength=n_steps), out=kick_ptr[1:])
    return kick_ptr, neurons[order]


def ate_it(mn9_steps, k=4, window_ms=50.0):
    """At least k MN9 spikes within any window of window_ms."""
    s = np.sort(mn9_steps)
    w = round(window_ms / DT)
    return bool(len(s) >= k and np.any(s[k - 1:] - s[: len(s) - k + 1] < w))


_BRAIN = None


def _init():
    global _BRAIN
    _BRAIN = load_brain()


def run_condition(args):
    name, rates, n_trials, t_ms, seed0 = args
    rowptr, post, weight, groups = _BRAIN
    n = len(rowptr) - 1
    n_steps = round(t_ms / DT)
    mn9 = np.array(groups["mn9"]["indices"], np.int32)
    dnp01 = np.array(groups["dnp01"]["indices"], np.int32)
    watch = np.concatenate([mn9, dnp01]).astype(np.int32)
    stimulated = np.concatenate([groups[c]["indices"] for c, r in rates.items() if r > 0] or [[]]).astype(np.int64)
    refr = np.full(n, REFRACTORY_STEPS, dtype=np.int64)
    refr[stimulated] = 0  # Poisson targets have no refractory period (model.py `poi`)
    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng([seed0, t])
        channels = [(groups[c]["indices"], r) for c, r in rates.items()]
        kick_ptr, kick_neuron = kick_schedule(channels, n_steps, rng)
        t0 = time.perf_counter()
        counts, wk, ws, n_active = simulate(rowptr, post, weight, kick_ptr, kick_neuron, refr, n_steps, watch)
        mn9_steps = ws[wk < len(mn9)]
        dn_steps = ws[wk >= len(mn9)]
        trials.append({
            "mn9": int(len(mn9_steps)),
            "dnp01": int(len(dn_steps)),
            "ate": ate_it(mn9_steps),
            "spikes": int(counts.sum()),
            "fired": int(np.count_nonzero(counts)),
            "touched": int(n_active),
            "wall_s": round(time.perf_counter() - t0, 3),
        })
    mn9_hz = np.array([tr["mn9"] for tr in trials]) * 1e3 / t_ms
    return {
        "name": name,
        "rates": rates,
        "t_ms": t_ms,
        "trials": n_trials,
        "mn9_hz_mean": round(float(mn9_hz.mean()), 3),
        "mn9_hz_sd": round(float(mn9_hz.std(ddof=1)) if n_trials > 1 else 0.0, 3),
        "mn9_hz_p025": round(float(np.percentile(mn9_hz, 2.5)), 3),
        "mn9_hz_p975": round(float(np.percentile(mn9_hz, 97.5)), 3),
        "ate": int(sum(tr["ate"] for tr in trials)),
        "dnp01_trials": int(sum(tr["dnp01"] > 0 for tr in trials)),
        "spikes_mean": round(float(np.mean([tr["spikes"] for tr in trials])), 1),
        "fired_max": int(max(tr["fired"] for tr in trials)),
        "touched_max": int(max(tr["touched"] for tr in trials)),
        "per_trial_mn9": [tr["mn9"] for tr in trials],
    }


def conditions():
    z = {"sugar": 0, "bitter": 0, "salt": 0, "water": 0}
    out = [("silent", dict(z))]
    for c in ("sugar", "bitter", "salt", "water"):
        for r in (20, 50, 100, 150):
            out.append((f"{c}_{r}", {**z, c: r}))
    for b in (50, 100, 150):
        out.append((f"sugar_150+bitter_{b}", {**z, "sugar": 150, "bitter": b}))
    out.append(("sugar_150+salt_150", {**z, "sugar": 150, "salt": 150}))
    out.append(("sugar_100+water_100", {**z, "sugar": 100, "water": 100}))
    out.append(("sugar_100+bitter_100+salt_50+water_50", {**z, "sugar": 100, "bitter": 100, "salt": 50, "water": 50}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="M0 gate: sugar curve only")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--t-ms", type=float, default=250.0)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()

    conds = conditions()
    if a.gate:
        conds = [c for c in conds if c[0] == "silent" or c[0].startswith("sugar_") and "+" not in c[0]]
    jobs = [(name, rates, a.trials, a.t_ms, 1000 + i) for i, (name, rates) in enumerate(conds)]
    _init()
    simulate(*[x for x in (_BRAIN[0], _BRAIN[1], _BRAIN[2])], np.zeros(11, np.int64), np.empty(0, np.int32),
             np.zeros(len(_BRAIN[0]) - 1, np.int64), 10, np.zeros(1, np.int32))  # compile once before forking
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs, initializer=_init) as ex:
        results = list(ex.map(run_condition, jobs))
    for r in results:
        print(f"{r['name']:42s} MN9 {r['mn9_hz_mean']:6.1f} ± {r['mn9_hz_sd']:5.1f} Hz  ate {r['ate']:2d}/{r['trials']}"
              f"  DNp01 {r['dnp01_trials']:2d}  fired≤{r['fired_max']}  touched≤{r['touched_max']}")
    print(f"{time.time() - t0:.0f} s")
    if a.gate:
        curve = {r["name"]: r["mn9_hz_mean"] for r in results}
        ok = curve["silent"] == 0 and curve["sugar_20"] < 2 and curve["sugar_50"] < curve["sugar_100"] < curve["sugar_150"] * 1.15
        print("M0 gate", "PASS" if ok else "FAIL", curve)
        raise SystemExit(0 if ok else 1)
    out = {
        "generator": "prep/refsim.py (float64 event-driven port of the Shiu et al. model)",
        "connectome": 783,
        "params": {"dt_ms": DT, "t_run_ms": a.t_ms, "eat_rule": "≥4 MN9 spikes within 50 ms"},
        "conditions": results,
    }
    (BRAIN / "reference.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
