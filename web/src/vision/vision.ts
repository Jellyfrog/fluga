// "Show the fly a picture": the photo moves in front of the fly (it approaches, or passes by), its change
// in brightness per eye column drives the medulla/lobula-plate columnar neurons of that column, and we read
// out the visual projection neurons (LCs) and descending neurons (DNs) it reaches.
//
// What is and isn't modelled (see prep/build_vision.py and data/vision_research.md): the connectome
// carries the activity from the columns to LCs and DNs; the change detection itself (ON/OFF, per 10 ms
// frame) happens here, outside the connectome, because the model has no temporal filtering. Viewing
// directions are approximate. None of the DN read-outs were validated for vision in this model.

import type { Drive, VisionMap } from '../sim/gpu-sim';
import { DT_MS } from '../sim/model';

export interface VisionData {
  columns: { right: boolean[]; az: number[]; el: number[] };
  drive: { on: number[]; onCol: number[]; off: number[]; offCol: number[]; both: number[]; bothCol: number[] };
  lc: { index: number[]; type: string[]; family: ('looming' | 'object' | 'other')[]; az: number[]; el: number[] };
  dn: Record<string, { left: number[]; right: number[] }>;
}

export type Motion = 'approach' | 'pass';

export const T_MS = 400; // per trial; motion runs from MOVE_START to MOVE_END
export const TRIALS = 10;
const FRAME_STEPS = 100; // 10 ms frames
const MOVE_START = 5, MOVE_END = 30; // frames
const MAX_HZ = 60;
const GAIN_HZ = 1200; // Hz per unit of brightness change per frame (0..1 scale)
const DECAY = 0.5; // a change keeps driving for a couple of frames, halving each frame
export const PHOTO_AZ = 45; // the photo spans +-45 deg azimuth in front of the fly at its resting size

let data: Promise<VisionData> | null = null;
export function loadVision(): Promise<VisionData> {
  data ??= fetch(`${import.meta.env.BASE_URL}brain/vision.json`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`vision: HTTP ${r.status}`))))
    .catch((err) => { data = null; throw err; });
  return data;
}

/** DN read-outs with their published behaviour links (not validated in this model). */
export const DN_INFO: { name: string; role: string }[] = [
  { name: 'DNp01', role: 'giant fiber: takeoff / escape jump' },
  { name: 'DNp02', role: 'looming-responsive' },
  { name: 'DNp03', role: 'looming-responsive' },
  { name: 'DNp04', role: 'looming-responsive' },
  { name: 'DNp06', role: 'looming-responsive' },
  { name: 'DNp11', role: 'looming-responsive' },
  { name: 'DNa01', role: 'steering' },
  { name: 'DNa02', role: 'steering' },
  { name: 'DNp09', role: 'forward walking' },
  { name: 'MDN', role: 'backward walking' },
  { name: 'MN9', role: 'proboscis (feeding)' },
];

/** Slots: 3 per column (0 ON, 1 OFF, 2 both); counted: all LCs, then each DN's left then right neurons. */
export function visionMap(v: VisionData): VisionMap & { dnSlices: { name: string; side: 'left' | 'right'; from: number; to: number }[] } {
  const d = v.drive;
  const counted = [...v.lc.index];
  const dnSlices: { name: string; side: 'left' | 'right'; from: number; to: number }[] = [];
  for (const { name } of DN_INFO) {
    for (const side of ['left', 'right'] as const) {
      const from = counted.length;
      counted.push(...v.dn[name][side]);
      dnSlices.push({ name, side, from, to: counted.length });
    }
  }
  return {
    driveNeuron: [...d.on, ...d.off, ...d.both],
    driveSlot: [...d.onCol.map((c) => 3 * c), ...d.offCol.map((c) => 3 * c + 1), ...d.bothCol.map((c) => 3 * c + 2)],
    counted,
    dnSlices,
  };
}

/** Grey-level image of the photo, sampled with an approximate 5 deg acceptance angle. */
function luminanceGrid(photo: HTMLCanvasElement, cols: number): { w: number; h: number; L: Float32Array } {
  const h = Math.max(1, Math.round((cols * photo.height) / photo.width));
  const c = document.createElement('canvas');
  c.width = cols; c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true })!;
  g.imageSmoothingQuality = 'high';
  g.drawImage(photo, 0, 0, cols, h);
  const px = g.getImageData(0, 0, cols, h).data;
  const L = new Float32Array(cols * h);
  for (let i = 0; i < L.length; i++) L[i] = (0.2126 * px[4 * i] + 0.7152 * px[4 * i + 1] + 0.0722 * px[4 * i + 2]) / 255;
  return { w: cols, h, L };
}

/** Where the photo is at frame f: centre azimuth and scale (1 = resting size). */
export function placement(motion: Motion, f: number): { az: number; scale: number } {
  const t = Math.min(1, Math.max(0, (f - MOVE_START) / (MOVE_END - MOVE_START)));
  if (motion === 'approach') return { az: 0, scale: 0.25 / (1 - 0.85 * t) }; // looming: 1/distance, ends 1.7x
  return { az: 70 - 140 * t, scale: 1 }; // passes from the fly's right to its left
}

/** Per-frame Poisson probabilities for every slot, from the photo's brightness changes. */
export function encode(v: VisionData, photo: HTMLCanvasElement, motion: Motion): Drive {
  const grid = luminanceGrid(photo, 18); // 90 deg / 18 = 5 deg per sample
  const elSpan = PHOTO_AZ * (photo.height / photo.width);
  const nCols = v.columns.az.length, nSlots = 3 * nCols;
  const nFrames = Math.ceil(T_MS / DT_MS / FRAME_STEPS);
  const prob = new Float32Array(nFrames * nSlots);
  const BG = 0.5;
  const sample = (f: number, c: number) => {
    const { az, scale } = placement(motion, f);
    const u = (v.columns.az[c] - az) / (PHOTO_AZ * scale), w = -v.columns.el[c] / (elSpan * scale);
    if (Math.abs(u) > 1 || Math.abs(w) > 1) return BG;
    const x = Math.min(grid.w - 1, Math.floor(((u + 1) / 2) * grid.w)), y = Math.min(grid.h - 1, Math.floor(((w + 1) / 2) * grid.h));
    return grid.L[y * grid.w + x];
  };
  const prev = Float32Array.from({ length: nCols }, (_, c) => sample(0, c));
  const on = new Float32Array(nCols), off = new Float32Array(nCols);
  const p = DT_MS * 1e-3;
  for (let f = 0; f < nFrames; f++) {
    for (let c = 0; c < nCols; c++) {
      const L = sample(f, c), d = L - prev[c];
      prev[c] = L;
      on[c] = Math.min(MAX_HZ, Math.max(DECAY * on[c], GAIN_HZ * d));
      off[c] = Math.min(MAX_HZ, Math.max(DECAY * off[c], -GAIN_HZ * d));
      const base = f * nSlots + 3 * c;
      prob[base] = on[c] * p;
      prob[base + 1] = off[c] * p;
      prob[base + 2] = Math.min(MAX_HZ, on[c] + off[c]) * p;
    }
  }
  return { frameSteps: FRAME_STEPS, nSlots, prob };
}

export interface VisionOutcome {
  lcHz: Float32Array; // mean rate per LC over the trial
  dn: { name: string; role: string; leftHz: number; rightHz: number }[];
  escapeTrials: number; // trials in which DNp01 fired
  trials: number;
  drivenColumns: number; // columns that got any drive
}

export function readOut(v: VisionData, map: ReturnType<typeof visionMap>, counts: number[], dnp01Steps: number[][], drive: Drive): VisionOutcome {
  const trials = dnp01Steps.length, perHz = 1000 / T_MS / trials;
  const lcHz = Float32Array.from(v.lc.index, (_, k) => counts[k] * perHz);
  const dn = DN_INFO.map(({ name, role }) => {
    const hz = (side: 'left' | 'right') => {
      const s = map.dnSlices.find((x) => x.name === name && x.side === side)!;
      const n = s.to - s.from;
      return n ? counts.slice(s.from, s.to).reduce((a, b) => a + b, 0) * perHz / n : NaN;
    };
    return { name, role, leftHz: hz('left'), rightHz: hz('right') };
  });
  let driven = 0;
  const nCols = v.columns.az.length;
  for (let c = 0; c < nCols; c++) {
    for (let f = 0; f * drive.nSlots < drive.prob.length; f++) if (drive.prob[f * drive.nSlots + 3 * c + 2] > 0) { driven++; break; }
  }
  return { lcHz, dn, escapeTrials: dnp01Steps.filter((s) => s.length > 0).length, trials, drivenColumns: driven };
}
