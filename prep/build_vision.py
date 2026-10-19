"""Vision mode data: eye columns, the neurons a picture drives, and what is read out.

    python prep/build_vision.py        # writes web/public/brain/vision.json

Inputs (data/vision/, see data/vision_research.md for sources and licence):
  column_assignment.csv.gz              FlyWire v783 Codex: columnar neurons with eye-column hex
                                        coordinates (x posterior, y dorsal) per hemisphere
  Supplemental_file1_neuron_annotations.tsv   Schlegel et al. 2024 cell types and sides
  data/raw/Drosophila_brain_model/{Completeness,Connectivity}_783.*   the model's neuron order and edges

Output:
  columns   per eye column: hemisphere and an APPROXIMATE viewing direction (azimuth, elevation, degrees).
            The files give lattice coordinates, not measured angles; x and y are mapped linearly so
            each eye spans about 160 deg x 150 deg with 15 deg of frontal binocular overlap.
  drive     columnar neurons with their column, by pathway: ON (Mi1, Mi4, Tm3, T4a-d), OFF (Tm1, Tm2, Tm4,
            Tm9, Mi9, T5a-d) and both (T2, T2a, T3, Tm20, Tm21). L1-5, R7/8, C2/3 and T1 are not driven.
            Photoreceptors and lamina are not driven: their synapses are under-detected in FlyWire and the
            model gets their signs wrong (histamine, glutamatergic L1), so input does not get through.
  lc        visual projection neurons (LC4, LPLC2, LC11, LPLC1, LC12, LC17, LC18) with a receptive-field
            centre: the synapse-weighted mean direction of their column-assigned inputs.
  dn        descending neurons with published behaviour links, per side, plus MN9.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data/raw/Drosophila_brain_model'
VIS = ROOT / 'data/vision'
OUT = ROOT / 'web/public/brain/vision.json'

# Medulla and lobula-plate columnar types, by the polarity of the change they report.
ON_TYPES = ['Mi1', 'Mi4', 'Tm3', 'T4a', 'T4b', 'T4c', 'T4d']
OFF_TYPES = ['Tm1', 'Tm2', 'Tm4', 'Tm9', 'Mi9', 'T5a', 'T5b', 'T5c', 'T5d']
BOTH_TYPES = ['T2', 'T2a', 'T3', 'Tm20', 'Tm21']  # respond to brightening and darkening
LC_FAMILIES = {'LC4': 'looming', 'LPLC2': 'looming', 'LC11': 'object', 'LPLC1': 'object',
               'LC12': 'other', 'LC17': 'other', 'LC18': 'other'}
DNS = ['DNp01', 'DNa01', 'DNa02', 'MDN', 'DNp09', 'DNp02', 'DNp03', 'DNp04', 'DNp06', 'DNp11']

AZ_SPAN, EL_SPAN, OVERLAP = 160.0, 150.0, 15.0


def direction(hemisphere, x, y, xr, yr):
    """Approximate (azimuth, elevation): azimuth > 0 is to the fly's right, x = posterior, y = dorsal."""
    a = -OVERLAP + (x - xr[0]) / (xr[1] - xr[0]) * AZ_SPAN
    az = np.where(hemisphere == 'right', a, -a)
    el = (y - (yr[0] + yr[1]) / 2) / (yr[1] - yr[0]) * EL_SPAN
    return az, el


def main():
    ids = pd.read_csv(RAW / 'Completeness_783.csv', index_col=0).index.to_numpy(np.int64)
    index = pd.Series(np.arange(len(ids)), index=ids)

    cols = pd.read_csv(VIS / 'column_assignment.csv.gz')
    cols = cols[cols.root_id.isin(index.index)]
    xr, yr = (cols.x.min(), cols.x.max()), (cols.y.min(), cols.y.max())
    lattice = cols.drop_duplicates(['hemisphere', 'column_id'])[['hemisphere', 'column_id', 'x', 'y']].reset_index(drop=True)
    lattice['az'], lattice['el'] = direction(lattice.hemisphere.to_numpy(), lattice.x.to_numpy(), lattice.y.to_numpy(), xr, yr)
    col_of = {(h, c): i for i, (h, c) in enumerate(zip(lattice.hemisphere, lattice.column_id))}
    cols['col'] = [col_of[(h, c)] for h, c in zip(cols.hemisphere, cols.column_id)]
    cols['index'] = index[cols.root_id].to_numpy()

    def driven(types):
        d = cols[cols.type.isin(types)]
        return d['index'].astype(int).tolist(), d['col'].astype(int).tolist()

    on_i, on_c = driven(ON_TYPES)
    off_i, off_c = driven(OFF_TYPES)
    both_i, both_c = driven(BOTH_TYPES)

    ann = pd.read_csv(VIS / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', usecols=['root_id', 'cell_type', 'side'], low_memory=False)
    ann = ann[ann.root_id.isin(index.index)]

    # LC receptive fields from their columnar inputs.
    conn = pd.read_parquet(RAW / 'Connectivity_783.parquet', columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Connectivity'])
    lcs = ann[ann.cell_type.isin(LC_FAMILIES)].copy()
    lcs['index'] = index[lcs.root_id].to_numpy()
    colpos = cols.drop_duplicates('index').set_index('index')[['col']]
    e = conn[conn.Postsynaptic_Index.isin(lcs['index']) & conn.Presynaptic_Index.isin(colpos.index)].copy()
    e['col'] = colpos.loc[e.Presynaptic_Index, 'col'].to_numpy()
    e['az'] = lattice.az.to_numpy()[e.col]
    e['el'] = lattice.el.to_numpy()[e.col]
    w = e.Connectivity.astype(float)
    rf = (e.assign(waz=e.az * w, wel=e.el * w, w=w).groupby('Postsynaptic_Index')[['waz', 'wel', 'w']].sum())
    rf['az'], rf['el'] = rf.waz / rf.w, rf.wel / rf.w
    lcs = lcs[lcs['index'].isin(rf.index)]
    print(f"LCs with a receptive field: {len(lcs)} ({lcs.cell_type.value_counts().to_dict()})")

    dn = {}
    for t in DNS:
        rows = ann[ann.cell_type == t]
        dn[t] = {s: sorted(index[rows[rows.side == s].root_id].astype(int).tolist()) for s in ('left', 'right')}
    groups = json.loads((ROOT / 'web/public/brain/groups.json').read_text())['groups']
    dn['MN9'] = {'left': [], 'right': groups['mn9']['indices']}
    missing = [t for t, s in dn.items() if not (s['left'] or s['right'])]
    assert not missing, missing

    r1 = lambda a: [round(float(v), 1) for v in a]
    out = {
        'version': 1,
        'note': 'Approximate viewing directions (lattice coordinates mapped linearly); see prep/build_vision.py.',
        'columns': {'right': [h == 'right' for h in lattice.hemisphere], 'az': r1(lattice.az), 'el': r1(lattice.el)},
        'drive': {'on': on_i, 'onCol': on_c, 'off': off_i, 'offCol': off_c, 'both': both_i, 'bothCol': both_c},
        'lc': {'index': lcs['index'].astype(int).tolist(), 'type': lcs.cell_type.tolist(),
               'family': [LC_FAMILIES[t] for t in lcs.cell_type],
               'az': r1(rf.loc[lcs['index'], 'az']), 'el': r1(rf.loc[lcs['index'], 'el'])},
        'dn': dn,
    }
    OUT.write_text(json.dumps(out, separators=(',', ':')))
    print(f'{len(lattice)} columns, {len(on_i)} ON + {len(off_i)} OFF + {len(both_i)} ON/OFF driven neurons, {len(lcs)} LCs, '
          f'{sum(len(s["left"]) + len(s["right"]) for s in dn.values())} DN/MN neurons -> {OUT.relative_to(ROOT)} '
          f'({OUT.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
