// On-device dish recognizer: zero-shot TinyCLIP (ViT-39M/16, MIT licence, 4-bit weights) matches a photo
// against text embeddings of ~1,400 dish names and descriptions computed at build time
// (prep/classifier/embed_dishes.py). Run by onnxruntime-web. Photos never leave the page.

import * as ort from 'onnxruntime-web/webgpu';
import { DISHES_BASE, loadDishes, type DishInfo } from './dishes';

export interface Guess { label: string; name: string; score: number; dish: DishInfo } // label = dish id
export interface Classification { top: Guess[] } // top-5, sorted, softmax scores over the whole vocabulary

// No ort.env.wasm.wasmPaths on purpose: the webgpu bundle inlines its JS glue and locates its only
// runtime file, ort-wasm-simd-threaded.jsep.wasm (21.7 MB, ~5 MB gzipped), with
// new URL(..., import.meta.url). Vite serves that from node_modules in dev (onnxruntime-web is excluded
// from optimizeDeps) and emits it as a hashed asset in builds, so a copy under public/ would only duplicate it.
// Without COOP/COEP there is no SharedArrayBuffer; ask for one thread up front instead of a warning.
if (!globalThis.crossOriginIsolated) ort.env.wasm.numThreads = 1;

interface Preprocess { size: number; crop: number | null; mean: number[]; std: number[] }
const LOGIT_SCALE = 100; // CLIP's learned temperature: softmax(100 * cosine)
const TOP_K = 5;

let loading: Promise<void> | null = null;
let session: ort.InferenceSession;
let pre: Preprocess;
let textEmb: Float32Array; // dishes x dim, unit rows
let dim: number;
let dishes: DishInfo[];
let queue: Promise<unknown> = Promise.resolve(); // ort sessions must not run() concurrently

/**
 * Downloads the model (reporting byte progress in [0,1]) and creates the session, preferring WebGPU
 * and falling back to wasm. Safe to call repeatedly; a failed load can be retried.
 */
export function loadClassifier(onProgress?: (frac: number) => void): Promise<void> {
  loading ??= (async () => {
    const [bytes, prep, emb, table] = await Promise.all([
      fetchWithProgress(DISHES_BASE + 'vision.onnx', onProgress),
      fetch(DISHES_BASE + 'preprocess.json').then((r) => (r.ok ? r.json() : Promise.reject(new Error(`preprocess: HTTP ${r.status}`)))),
      fetchWithProgress(DISHES_BASE + 'text_emb.bin'),
      loadDishes(),
    ]);
    pre = prep;
    dishes = table.dishes;
    dim = table.dim;
    textEmb = dequantize(emb, dishes.length, dim);
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu', 'wasm'] });
  })().catch((err) => {
    loading = null;
    throw err;
  });
  return loading;
}

/** int8 rows followed by one float32 scale per row (see build_web.py). */
function dequantize(bytes: Uint8Array, rows: number, cols: number): Float32Array {
  if (bytes.byteLength !== rows * cols + 4 * rows) throw new Error('text_emb.bin does not match dishes.json');
  const q = new Int8Array(bytes.buffer, bytes.byteOffset, rows * cols);
  const scale = new Float32Array(bytes.slice(rows * cols).buffer);
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * scale[r];
  return out;
}

/**
 * Top-5 dishes for an image. The caller owns orientation: decode files with
 * createImageBitmap(file, { imageOrientation: 'from-image' }) so EXIF rotation is applied.
 */
export async function classify(img: ImageBitmap | HTMLImageElement | HTMLCanvasElement): Promise<Classification> {
  await loadClassifier();
  const side = pre.crop ?? pre.size;
  const input = new ort.Tensor('float32', toTensorData(img, side), [1, 3, side, side]);
  const run = queue.then(() => session.run({ [session.inputNames[0]]: input }));
  queue = run.catch(() => {});
  const out = await run;
  const e = out[session.outputNames[0]].data as Float32Array;
  const logits = new Float32Array(dishes.length);
  for (let r = 0; r < dishes.length; r++) {
    let s = 0;
    for (let c = 0; c < dim; c++) s += e[c] * textEmb[r * dim + c];
    logits[r] = LOGIT_SCALE * s;
  }
  return { top: topK(softmax(logits), TOP_K) };
}

/** Center square covering crop/size of the short side, resampled to side x side, as normalized NCHW floats. */
function toTensorData(img: ImageBitmap | HTMLImageElement | HTMLCanvasElement, size: number): Float32Array {
  const w = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const h = img instanceof HTMLImageElement ? img.naturalHeight : img.height;
  if (!w || !h) throw new Error('classify: image has no pixels (not loaded yet?)');
  let side = Math.min(w, h) * ((pre.crop ?? pre.size) / pre.size);
  let src: CanvasImageSource = img;
  let sx = (w - side) / 2, sy = (h - side) / 2;
  // Halve in steps until within 2x of the target: a single big drawImage downscale aliases, while
  // the model was trained on PIL's area-weighted resize.
  while (side > 2 * size) {
    const half = Math.round(side / 2);
    const step = context2d(half);
    step.drawImage(src, sx, sy, side, side, 0, 0, half, half);
    src = step.canvas;
    side = half; sx = sy = 0;
  }
  const ctx = context2d(size);
  ctx.drawImage(src, sx, sy, side, side, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) data[c * n + i] = (rgba[4 * i + c] / 255 - pre.mean[c]) / pre.std[c];
  }
  return data;
}

/** A square 2D context with high-quality smoothing, offscreen when the browser allows. */
function context2d(size: number): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(size, size)
    : Object.assign(document.createElement('canvas'), { width: size, height: size });
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
    CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  if (!ctx) throw new Error('classify: 2D canvas unavailable');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  return ctx;
}

function softmax(z: Float32Array): Float32Array {
  const max = z.reduce((a, b) => Math.max(a, b), -Infinity);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

function topK(p: Float32Array, k: number): Guess[] {
  return Array.from(p, (score, i) => ({ score, i }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k)
    .map(({ score, i }) => ({ label: dishes[i].id, name: dishes[i].name, score, dish: dishes[i] }));
}

/** fetch() into one buffer, reporting received/Content-Length; progress is skipped when the length is unknown. */
async function fetchWithProgress(url: string, onProgress?: (frac: number) => void): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url.split('/').pop()}: HTTP ${res.status}`);
  // Content-Length is the encoded size; with transfer compression it undercounts, hence the clamp.
  const total = Number(res.headers.get('Content-Length')) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress?.(Math.min(1, got / total));
  }
  onProgress?.(1);
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  return bytes;
}
