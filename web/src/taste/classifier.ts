// On-device Food-101 dish recognizer: MobileNetV2 (torchvision, fine-tuned by AlexKoff88, ~76% top-1)
// with int8 per-channel weights, run by onnxruntime-web. Photos never leave the page.

import * as ort from 'onnxruntime-web/webgpu';
import { prettyName } from './tastes';

export interface Guess { label: string; name: string; score: number } // label = snake_case, name = "Caesar salad"
export interface Classification { top: Guess[] } // top-5, sorted, softmax scores

const BASE = new URL(import.meta.env.BASE_URL, document.baseURI);
const MODEL_URL = new URL('models/food101/model.onnx', BASE).href;
const LABELS_URL = new URL('models/food101/labels.json', BASE).href;

// No ort.env.wasm.wasmPaths on purpose: the webgpu bundle inlines its JS glue and locates its only
// runtime file, ort-wasm-simd-threaded.jsep.wasm (21.7 MB, ~5 MB gzipped), with
// new URL(..., import.meta.url). Vite serves that from node_modules in dev (onnxruntime-web is excluded
// from optimizeDeps) and emits it as a hashed asset in builds, so a copy under public/ would only duplicate it.
// Without COOP/COEP there is no SharedArrayBuffer; ask for one thread up front instead of a warning.
if (!globalThis.crossOriginIsolated) ort.env.wasm.numThreads = 1;

// torchvision eval transform: Resize(256) on the short side, CenterCrop(224), ImageNet mean/std, RGB, NCHW.
const SIZE = 224;
const CROP = 224 / 256;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const TOP_K = 5;

let loading: Promise<void> | null = null;
let session: ort.InferenceSession;
let labels: string[];
let queue: Promise<unknown> = Promise.resolve(); // ort sessions must not run() concurrently

/**
 * Downloads the model (reporting byte progress in [0,1]) and creates the session, preferring WebGPU
 * and falling back to wasm. Safe to call repeatedly; a failed load can be retried.
 */
export function loadClassifier(onProgress?: (frac: number) => void): Promise<void> {
  loading ??= (async () => {
    const [bytes, names] = await Promise.all([
      fetchWithProgress(MODEL_URL, onProgress),
      fetch(LABELS_URL).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`labels: HTTP ${r.status}`)))),
    ]);
    labels = names as string[];
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu', 'wasm'] });
  })().catch((err) => {
    loading = null;
    throw err;
  });
  return loading;
}

/**
 * Top-5 Food-101 guesses for an image. The caller owns orientation: decode files with
 * createImageBitmap(file, { imageOrientation: 'from-image' }) so EXIF rotation is applied.
 */
export async function classify(img: ImageBitmap | HTMLImageElement | HTMLCanvasElement): Promise<Classification> {
  await loadClassifier();
  const input = new ort.Tensor('float32', toTensorData(img), [1, 3, SIZE, SIZE]);
  const run = queue.then(() => session.run({ [session.inputNames[0]]: input }));
  queue = run.catch(() => {});
  const out = await run;
  const logits = out[session.outputNames[0]].data as Float32Array;
  return { top: topK(softmax(logits), TOP_K) };
}

/** Center square covering 224/256 of the short side, resampled to 224x224, as normalized NCHW floats. */
function toTensorData(img: ImageBitmap | HTMLImageElement | HTMLCanvasElement): Float32Array {
  const w = img instanceof HTMLImageElement ? img.naturalWidth : img.width;
  const h = img instanceof HTMLImageElement ? img.naturalHeight : img.height;
  if (!w || !h) throw new Error('classify: image has no pixels (not loaded yet?)');
  let side = Math.min(w, h) * CROP;
  let src: CanvasImageSource = img;
  let sx = (w - side) / 2, sy = (h - side) / 2;
  // Halve in steps until within 2x of the target: a single big drawImage downscale aliases, while
  // the model was trained on PIL's area-weighted resize.
  while (side > 2 * SIZE) {
    const half = Math.round(side / 2);
    const step = context2d(half);
    step.drawImage(src, sx, sy, side, side, 0, 0, half, half);
    src = step.canvas;
    side = half; sx = sy = 0;
  }
  const ctx = context2d(SIZE);
  ctx.drawImage(src, sx, sy, side, side, 0, 0, SIZE, SIZE);
  const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) data[c * n + i] = (rgba[4 * i + c] / 255 - MEAN[c]) / STD[c];
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
    .map(({ score, i }) => ({ label: labels[i], name: prettyName(labels[i]), score }));
}

/** fetch() into one buffer, reporting received/Content-Length; progress is skipped when the length is unknown. */
async function fetchWithProgress(url: string, onProgress?: (frac: number) => void): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`model: HTTP ${res.status}`);
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
