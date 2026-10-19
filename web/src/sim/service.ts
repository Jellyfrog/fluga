// One interface over the three ways to get a verdict: the live brain in a worker, the live brain on
// the main thread (browsers without WebGPU in workers), and the recorded table (no WebGPU at all).

import { CHANNELS, brainBaseUrl, loadBrain } from './brain';
import { GpuBrain, type ChunkUpdate, type Rates, type TastingRequest, type TastingResult } from './gpu-sim';
import type { FromWorker, ToWorker } from './worker';

export interface BrainService {
  kind: 'live' | 'recorded';
  where: string; // "worker", "main thread" or "recorded table"
  taste(req: TastingRequest, onChunk?: (u: ChunkUpdate) => void): Promise<TastingResult | RecordedResult>;
}

export interface RecordedResult { recorded: true; ate: number; trials: number; mn9Hz: number; fled: number }

class WorkerService implements BrainService {
  kind = 'live' as const;
  where = 'worker';
  private next = 1;
  private calls = new Map<number, { resolve: (r: TastingResult) => void; reject: (e: Error) => void; onChunk?: (u: ChunkUpdate) => void }>();
  constructor(private worker: Worker) {
    worker.addEventListener('message', (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type !== 'chunk' && m.type !== 'done' && m.type !== 'error') return;
      const call = m.id !== undefined ? this.calls.get(m.id) : undefined;
      if (!call) return;
      if (m.type === 'chunk') call.onChunk?.(m.update);
      else if (m.type === 'done') { this.calls.delete(m.id); call.resolve(m.result); }
      else { this.calls.delete(m.id!); call.reject(new Error(m.message)); }
    });
  }
  taste(req: TastingRequest, onChunk?: (u: ChunkUpdate) => void) {
    const id = this.next++;
    return new Promise<TastingResult>((resolve, reject) => {
      this.calls.set(id, { resolve, reject, onChunk });
      this.worker.postMessage({ type: 'run', id, req } satisfies ToWorker);
    });
  }
}

class MainThreadService implements BrainService {
  kind = 'live' as const;
  where = 'main thread';
  constructor(private sim: GpuBrain) {}
  taste(req: TastingRequest, onChunk?: (u: ChunkUpdate) => void) { return this.sim.run(req, onChunk); }
}

interface Table { axes: Record<string, number[]>; trials: number; ate: number[]; mn9_hz: number[]; fled: number[] }

/** Multilinear interpolation in the precomputed 4-D grid of reference outcomes. */
class RecordedService implements BrainService {
  kind = 'recorded' as const;
  where = 'recorded table';
  constructor(private t: Table) {}
  async taste(req: TastingRequest): Promise<RecordedResult> {
    const axes = CHANNELS.map((c) => this.t.axes[c]);
    const pos = CHANNELS.map((c, d) => {
      const a = axes[d], x = Math.min(a[a.length - 1], Math.max(0, req.rates[c]));
      let i = 0;
      while (i < a.length - 2 && x > a[i + 1]) i++;
      return { i, f: (x - a[i]) / (a[i + 1] - a[i]) };
    });
    const strides = [axes[1].length * axes[2].length * axes[3].length, axes[2].length * axes[3].length, axes[3].length, 1];
    let ate = 0, hz = 0, fled = 0;
    for (let corner = 0; corner < 16; corner++) {
      let w = 1, idx = 0;
      for (let d = 0; d < 4; d++) {
        const hi = (corner >> d) & 1;
        w *= hi ? pos[d].f : 1 - pos[d].f;
        idx += (pos[d].i + hi) * strides[d];
      }
      ate += w * this.t.ate[idx]; hz += w * this.t.mn9_hz[idx]; fled += w * this.t.fled[idx];
    }
    await new Promise((r) => setTimeout(r, 600)); // let the tasting screen breathe
    const scale = req.trials / this.t.trials;
    return { recorded: true, ate: Math.round(ate * scale), trials: req.trials, mn9Hz: hz, fled: Math.round(fled * scale) };
  }
}

export interface Startup { service: BrainService; note?: string }

/** Brings up the best available brain. onProgress gets the download fraction. */
export async function startBrain(maxTrials: number, onProgress: (frac: number) => void): Promise<Startup> {
  const recorded = async (why: string): Promise<Startup> => {
    const t = await (await fetch(`${brainBaseUrl()}recorded.json`)).json();
    onProgress(1);
    return { service: new RecordedService(t), note: why };
  };
  if (!('gpu' in navigator) || !(await navigator.gpu.requestAdapter().catch(() => null))) {
    return recorded('This browser has no WebGPU, so the fly answers from recorded runs of the same brain.');
  }
  // Prefer a worker; fall back to the main thread if WebGPU is missing there.
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  const viaWorker = await new Promise<Startup | 'nogpu'>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === 'progress') onProgress(m.frac);
      else if (m.type === 'nogpu') resolve('nogpu');
      else if (m.type === 'ready') { worker.onmessage = null; resolve({ service: new WorkerService(worker) }); }
      else if (m.type === 'error' && m.id === undefined) reject(new Error(m.message));
    };
    worker.onerror = () => resolve('nogpu');
    worker.postMessage({ type: 'init', maxTrials, brainBase: brainBaseUrl() } satisfies ToWorker);
  }).catch(async (err: Error) => {
    console.warn('live brain failed in worker:', err);
    worker.terminate();
    return recorded(`The live brain could not start here (${err.message}), so the fly answers from recorded runs.`);
  });
  if (viaWorker !== 'nogpu') return viaWorker;
  worker.terminate();
  try {
    const sim = await GpuBrain.create(await loadBrain(onProgress), maxTrials);
    return { service: new MainThreadService(sim) };
  } catch (err) {
    return recorded(`The live brain could not start here (${(err as Error).message}), so the fly answers from recorded runs.`);
  }
}

export function isRecorded(r: TastingResult | RecordedResult): r is RecordedResult { return 'recorded' in r; }
export type { Rates };
