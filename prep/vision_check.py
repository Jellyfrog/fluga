"""Reference runs of the vision drive on the float64 simulator, to check the WebGPU path against.

    python prep/vision_check.py [trials=12]     # writes web/public/brain/vision_reference.json

The stimulus is synthetic and fixed: a disk of ~44 right-eye columns in front of the fly gets ON drive
at 40 Hz from 50 to 200 ms, then OFF drive at 40 Hz from 200 to 350 ms; ON/OFF types get 40 Hz throughout.
Semantics match the GPU kernels: driven neurons are Poisson sources with no refractory period for the
whole trial; each 0.1 ms step kicks with probability prob[frame, slot]; frames are 10 ms.
Outputs, per condition, the mean and sd over trials of spikes per LC family, per DN/MN and in total.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refsim import DT, REFRACTORY_STEPS, load_brain, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VISION = ROOT / 'web/public/brain/vision.json'
OUT = ROOT / 'web/public/brain/vision_reference.json'
T_MS, FRAME_STEPS = 400.0, 100


def slots(v):
    """Slot = 3 * column + (0 ON, 1 OFF, 2 both), as in web/src/vision."""
    d = v['drive']
    neurons = np.array(d['on'] + d['off'] + d['both'], np.int64)
    sl = np.array([3 * c for c in d['onCol']] + [3 * c + 1 for c in d['offCol']] + [3 * c + 2 for c in d['bothCol']], np.int64)
    return neurons, sl, 3 * len(v['columns']['az'])


def stimulus(v, n_slots):
    az, el = np.array(v['columns']['az']), np.array(v['columns']['el'])
    right = np.array(v['columns']['right'])
    disk = np.where(right & (np.hypot(az - 20, el) < 18))[0]
    n_frames = int(np.ceil(T_MS / DT / FRAME_STEPS))
    prob = np.zeros((n_frames, n_slots), np.float32)
    p = 40 * DT * 1e-3
    prob[5:20, 3 * disk] = p
    prob[20:35, 3 * disk + 1] = p
    prob[5:35, 3 * disk + 2] = p
    return prob, disk


def kicks(neurons, sl, prob, n_steps, rng):
    p_step = prob[np.arange(n_steps) // FRAME_STEPS][:, sl]  # (steps, driven neurons)
    steps, cols = np.nonzero(rng.random(p_step.shape) < p_step)
    order = np.argsort(steps, kind='stable')
    kick_ptr = np.zeros(n_steps + 1, np.int64)
    np.cumsum(np.bincount(steps, minlength=n_steps), out=kick_ptr[1:])
    return kick_ptr, neurons[cols[order]].astype(np.int32)


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    v = json.loads(VISION.read_text())
    rowptr, post, weight, _ = load_brain()
    n = len(rowptr) - 1
    neurons, sl, n_slots = slots(v)
    prob, disk = stimulus(v, n_slots)
    n_steps = round(T_MS / DT)
    refr = np.full(n, REFRACTORY_STEPS, np.int64)
    refr[neurons] = 0
    lc = np.array(v['lc']['index'], np.int64)
    fam = np.array(v['lc']['family'])
    groups = {f'lc_{f}': lc[fam == f] for f in sorted(set(fam))}
    for name, sides in v['dn'].items():
        groups[name] = np.array(sides['left'] + sides['right'], np.int64)
    per = {g: [] for g in groups}
    total = []
    for t in range(trials):
        rng = np.random.default_rng([7, t])
        kp, kn = kicks(neurons, sl, prob, n_steps, rng)
        counts, _, _, _ = simulate(rowptr, post, weight, kp, kn, refr, n_steps, np.zeros(0, np.int32))
        for g, idx in groups.items():
            per[g].append(int(counts[idx].sum()))
        total.append(int(counts.sum()))
        print(f'trial {t}: {total[-1]} spikes, ' + ', '.join(f'{g} {per[g][-1]}' for g in groups if per[g][-1]), flush=True)
    stat = lambda x: {'mean': round(float(np.mean(x)), 3), 'sd': round(float(np.std(x, ddof=1)), 3)}
    nz = np.nonzero(prob)
    OUT.write_text(json.dumps({
        't_ms': T_MS, 'frame_steps': FRAME_STEPS, 'n_slots': n_slots, 'trials': trials,
        'prob_sparse': {'frame': nz[0].tolist(), 'slot': nz[1].tolist(), 'p': prob[nz].tolist()},
        'groups': {g: idx.tolist() for g, idx in groups.items()},
        'spikes': {g: stat(x) for g, x in per.items()} | {'total': stat(total)},
    }))
    print(f'disk of {len(disk)} columns; {trials} trials -> {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
