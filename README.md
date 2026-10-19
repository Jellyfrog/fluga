# FlyPick

A joke web app with a real core: upload photos of dishes, and a simulated fruit fly brain — the
full FlyWire v783 connectome, 138,639 neurons and 15.1 million connections — tastes each one in
your browser on the GPU (WebGPU) and picks the dish it would eat.

A food recognizer turns each photo into a taste profile (sweet, bitter, salty, watery); the fly
brain only ever receives taste. Its preference comes from the connectome, run with the leaky
integrate-and-fire model of Shiu et al. (2024).

## Layout

| Path | What |
| --- | --- |
| `prep/build_brain.py` | Phase 0: connectome → `web/public/brain/{rowptr,post,weight}.bin(.gz)`, `groups.json`, `manifest.json` |
| `prep/refsim.py` | float64 event-driven reference simulator (port of flight-or-bite's Brian2-identical backend) → `reference.json`; `--gate` runs the M0 sugar-curve gate |
| `prep/brian2_check.py` | spot-checks `refsim.py` against the original Brian2 model |
| `prep/fallback_table.py` | precomputed "recorded brain" grid for browsers without WebGPU → `recorded.json` |
| `prep/compare.py` | Phase 5 pass/fail of WebGPU results against `reference.json` |
| `web/src/sim/` | WebGPU simulator (`shaders.ts`, `gpu-sim.ts`), brain loader, worker, service with fallbacks |
| `web/src/taste/` | on-device Food-101 classifier, `tastes.json`, taste → rates and the decision rule |
| `web/src/flyeye/` | fly-eye view (WebGL2 fragment shader) |
| `web/src/ui/`, `web/src/main.ts` | the four screens, spike raster, share card |
| `web/validate.html`, `web/tools/validate.mjs` | headless validation page and its Playwright runner |

## Build the data (once)

```sh
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model data/raw/Drosophila_brain_model
uv venv .venv && uv pip install -p .venv numpy pandas pyarrow numba
.venv/bin/python prep/build_brain.py        # ~1 min
.venv/bin/python prep/refsim.py --gate      # M0 gate
.venv/bin/python prep/refsim.py             # reference.json
.venv/bin/python prep/fallback_table.py     # recorded.json (~5 min on 14 cores)
```

## Run

```sh
cd web
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in web/dist (deploy to Cloudflare Pages / GitHub Pages)
```

WebGPU needs a secure context (https or localhost). Browsers without WebGPU get the recorded brain.

## Validate (Phase 5)

```sh
cd web && npm run dev &
node tools/validate.mjs tools/out/validation.json "trials=30&mode=both"
../.venv/bin/python ../prep/compare.py tools/out/validation.json
```

On a Linux box without a GPU or browser libraries, `source web/tools/env.sh` first (local libs,
SwiftShader); expect it to be slow.

## Notes on the port

- Brian2's step order is update → threshold → synapses/Poisson → reset. Synaptic input to a
  refractory neuron is dropped and input in the step a neuron fires is wiped by the reset; both
  are reproduced exactly. Input is accumulated as integer synapse counts (`atomic<i32>`), so there
  is no fixed-point scale to choose and no overflow.
- Only neurons that have left exact rest are updated (a neuron at v = v₀, g = 0 is a fixed
  point), so the "active set" mode is exact, not an approximation: plain and active runs with the
  same seed are bit-identical.
- v630 → v783: the model authors' neuron lists are v630 IDs. IDs missing in v783 are dropped and
  listed in `groups.json` (1 sugar, 1 bitter GRN, and the right MN9); the sugar-curve gate and the
  Brian2 cross-check pass without them.
