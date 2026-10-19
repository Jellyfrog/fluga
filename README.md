# Fluga

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
| `web/src/taste/` | on-device zero-shot dish recognizer (TinyCLIP), the dish table and search, taste → rates and the decision rule |
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

## Dishes and tastes

`prep/dishes/*.json` lists about 1,400 dishes (all 101 Food-101 classes plus world cuisines, sweets,
produce, snacks and drinks). Each dish is one USDA FNDDS food or a short recipe of FNDDS foods with
gram weights. `prep/usda_tastes.py` turns nutrients per 100 g into the four channels with saturating
curves: sugars → sweet, sodium → salty, caffeine + theobromine + bitter-ingredient share → bitter,
water above 55 g → watery. Photos are matched zero-shot by TinyCLIP ViT-39M/16 (MIT) against text
embeddings of each dish's name, aliases and a short visual description, and a recognized dish's taste is
the score-weighted blend of the top 5 guesses.

```sh
curl -sSLO https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip   # into data/usda, unzip
.venv/bin/python prep/usda_tastes.py && .venv/bin/python prep/dishes.py check
# separate venv with torch, transformers, onnx, onnxruntime:
python prep/classifier/export_clip.py wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M <dir>   # + 4-bit variants
python prep/classifier/embed_dishes.py <dir>
python prep/classifier/eval_vocab.py <dir> <dir>/vision_q4_b32a.onnx food101val:<imgs> mmfood:data/eval/mmfood
python prep/classifier/build_web.py <dir> <dir>/vision_q4_b32a.onnx                     # -> web/public/models/dishes
```

Measured (`eval_vocab.py`, mean absolute taste error over the 4 channels, lower is better):

| Photos | New: TinyCLIP over 1,372 dishes | Old: MobileNetV2 over Food-101's 101 |
| --- | --- | --- |
| 400 phone photos from MM-Food-100K, 210 dishes | 0.076 (top-1 dish 48%, top-5 72%) | 0.197 |
| … of which the 79 Food-101 dishes | 0.067 | 0.080 |
| 800 Food-101 validation photos | 0.058 (top-1 56%, top-5 79%) | 0.033 (the old model was trained on this set) |

The 4-bit image model (27 MB) costs about 2 points of top-1 against 8-bit (42 MB) or full precision.

## Vision mode ("Show it a picture", experimental)

`prep/build_vision.py` writes `web/public/brain/vision.json` from FlyWire Codex's column assignment and
the Schlegel et al. annotations (downloads and sources in `data/vision_research.md`): 1,581 eye columns with
approximate viewing directions, ~31,000 medulla/lobula-plate columnar neurons to drive (ON: Mi1, Mi4, Tm3,
T4; OFF: Tm1, Tm2, Tm4, Tm9, Mi9, T5; either: T2, T2a, T3, Tm20, Tm21), 1,425 LC/LPLC visual projection
neurons with receptive-field centres derived from their columnar inputs, and DNs with published behaviour
links. In the browser (`web/src/vision/`), a picture approaches or passes by; brightness changes per column
and 10 ms frame become Poisson rates (up to 60 Hz) on those neurons. Photoreceptors and lamina are not
driven: their synapses are under-detected in FlyWire and their signs are wrong in the model, so input
does not get through. Nothing about vision was validated in this model; the page says so.

`prep/vision_check.py` runs a fixed synthetic stimulus on the float64 reference simulator, and
`validate.html?vision=1` runs it on WebGPU for comparison.

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
