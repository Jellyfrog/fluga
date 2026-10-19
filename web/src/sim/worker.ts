// Web Worker that owns the WebGPU brain so the UI stays smooth.

import { loadBrain } from './brain';
import { GpuBrain, type TastingRequest, type VisionMap } from './gpu-sim';

export type ToWorker = { type: 'init'; maxTrials: number; brainBase: string } | { type: 'run'; id: number; req: TastingRequest }
  | { type: 'vision'; map: VisionMap };
export type FromWorker =
  | { type: 'nogpu'; reason: string }
  | { type: 'progress'; frac: number }
  | { type: 'ready'; maxTrials: number; adapter: string }
  | { type: 'chunk'; id: number; update: import('./gpu-sim').ChunkUpdate }
  | { type: 'done'; id: number; result: import('./gpu-sim').TastingResult }
  | { type: 'error'; id?: number; message: string };

const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);
let sim: GpuBrain | null = null;

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === 'init') {
    try {
      const adapter = 'gpu' in navigator ? await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }) : null;
      if (!adapter) return post({ type: 'nogpu', reason: 'no WebGPU adapter in workers' });
      const brain = await loadBrain((frac) => post({ type: 'progress', frac }), m.brainBase);
      sim = await GpuBrain.create(brain, m.maxTrials, adapter);
      post({ type: 'ready', maxTrials: sim.maxTrials, adapter: adapter.info?.vendor ?? '' });
    } catch (err) {
      post({ type: 'error', message: String((err as Error)?.message ?? err) });
    }
  } else if (m.type === 'vision') {
    sim?.setVision(m.map); // queued on the GPU ahead of any later run
  } else if (m.type === 'run') {
    try {
      const result = await sim!.run(m.req, (update) => post({ type: 'chunk', id: m.id, update }));
      post({ type: 'done', id: m.id, result });
    } catch (err) {
      post({ type: 'error', id: m.id, message: String((err as Error)?.message ?? err) });
    }
  }
};
