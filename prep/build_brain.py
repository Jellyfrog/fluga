"""Phase 0: turn the Shiu et al. v783 connectome into compact binaries the browser loads.

Outputs (data/brain/*.bin, gzipped copies in web/public/brain/*.bin.gz):
    rowptr.bin   Uint32[N+1]   outgoing edge ranges per presynaptic neuron
    post.bin     Uint32[E]     postsynaptic index per edge (sorted within each row)
    weight.bin   Int16[E]      signed synapse count ("Excitatory x Connectivity")
Outputs (web/public/brain/):
    groups.json  model indices of the taste input and motor read-out neurons
    manifest.json sizes, counts and a content hash used as the cache version

Group ids come from the model authors' notebooks (v630 ids). Ids that no longer exist in v783
are dropped and listed under "missing" instead of being guessed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "Drosophila_brain_model"
OUT = ROOT / "web" / "public" / "brain"  # what the app downloads: *.gz and JSON
RAW_OUT = ROOT / "data" / "brain"  # uncompressed binaries for the Python tools

# philshiu/Drosophila_brain_model figures.ipynb (v630 ids, right labellum unless noted)
GROUPS_V630 = {
    "sugar": [
        720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
        720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
        720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
        720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
        720575940611875570,
    ],
    "bitter": [
        720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
        720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
        720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
        720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
        720575940604027168,
    ],
    "salt": [  # Ir94e
        720575940614211295, 720575940638218173, 720575940628832256, 720575940626016017, 720575940621375231,
        720575940612920386, 720575940614273292, 720575940628198503, 720575940626241636, 720575940619387814,
        720575940624604560, 720575940615274425, 720575940610683315, 720575940627265265, 720575940624079544,
        720575940629211607, 720575940615089369, 720575940631082124,
    ],
    "water": [
        720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053, 720575940622902535,
        720575940616177458, 720575940660292225, 720575940622486922, 720575940613786774, 720575940629852866,
        720575940625861168, 720575940613996959, 720575940617857694, 720575940644965399, 720575940625203504,
        720575940630553415, 720575940635172191, 720575940634796536,
    ],
    # read-outs: figures.ipynb `ids_mn9` (left, right); DNp01 from flywire_annotations v783
    "mn9": [720575940660219265, 720575940645521262],
    "dnp01": [720575940622838154, 720575940632499757],
}
INPUTS = ("sugar", "bitter", "salt", "water")


def load():
    ids = pd.read_csv(RAW / "Completeness_783.csv", index_col=0).index.to_numpy(np.int64)
    df = pd.read_parquet(
        RAW / "Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = df["Presynaptic_Index"].to_numpy(np.int64)
    post = df["Postsynaptic_Index"].to_numpy(np.int64)
    w = df["Excitatory x Connectivity"].to_numpy(np.int64)
    return ids, pre, post, w


def build_csr(n, pre, post, w):
    order = np.lexsort((post, pre))  # by pre, then post inside each row
    rowptr = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum(np.bincount(pre, minlength=n), out=rowptr[1:])
    return rowptr, post[order].astype(np.uint32), w[order]


def resolve_groups(ids):
    index = {int(f): i for i, f in enumerate(ids)}
    groups = {}
    for name, flyids in GROUPS_V630.items():
        present = [f for f in flyids if f in index]
        missing = [f for f in flyids if f not in index]
        groups[name] = {
            "indices": [index[f] for f in present],
            "flywire": [str(f) for f in present],
            "missing": [str(f) for f in missing],
        }
    return groups


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    ids, pre, post, w = load()
    n, e = len(ids), len(pre)
    assert w.min() >= -32768 and w.max() <= 32767, "weights must fit Int16"
    assert not np.any(w == 0), "zero-weight edge"
    rowptr, post_s, w_s = build_csr(n, pre, post, w)

    files = {
        "rowptr.bin": rowptr.tobytes(),
        "post.bin": post_s.tobytes(),
        "weight.bin": w_s.astype(np.int16).tobytes(),
    }
    digest = hashlib.sha256()
    sizes = {}
    for name, blob in files.items():
        (RAW_OUT / name).write_bytes(blob)
        gz = gzip.compress(blob, compresslevel=9, mtime=0)
        (OUT / (name + ".gz")).write_bytes(gz)
        digest.update(blob)
        sizes[name] = {"raw": len(blob), "gzip": len(gz)}

    groups = resolve_groups(ids)
    (OUT / "groups.json").write_text(json.dumps({"version": 783, "inputs": list(INPUTS), "groups": groups}, indent=1))

    out_deg = np.diff(rowptr.astype(np.int64))
    manifest = {
        "version": 783,
        "neurons": int(n),
        "edges": int(e),
        "hash": digest.hexdigest()[:16],
        "weightRange": [int(w.min()), int(w.max())],
        "inhibitoryFraction": round(float(np.mean(w < 0)), 4),
        "maxOutDegree": int(out_deg.max()),
        "meanOutDegree": round(float(out_deg.mean()), 1),
        "files": sizes,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    for k, g in groups.items():
        print(f"{k:7s} {len(g['indices']):3d} present, missing {g['missing']}")


if __name__ == "__main__":
    main()
