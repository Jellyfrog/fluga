// Headless validation page: runs reference.json conditions on the WebGPU brain (Phase 5).
// Results land in window.__results; tools/validate.mjs collects them, prep/compare.py judges them.

import { loadBrain } from './sim/brain';
import { GpuBrain, type Rates } from './sim/gpu-sim';
import { ateIt } from './sim/model';

interface RefCondition { name: string; rates: Rates; t_ms: number; trials: number }
interface Out {
  name: string; mode: 'plain' | 'active'; rates: Rates; trials: number; t_ms: number;
  mn9: number[]; dnp01: number[]; ate: number; spikes: number[]; active: number[]; wall_ms: number; overflow: boolean;
}

const log = (s: string) => { document.getElementById('log')!.textContent += s + '\n'; console.log(s); };
declare global { interface Window { __results?: { done: boolean; error?: string; results: Out[]; adapter?: string } } }

/** vision=1: the synthetic stimulus of prep/vision_check.py, GPU vs float64 reference, per neuron group. */
async function vision(q: URLSearchParams, state: NonNullable<Window['__results']>) {
  const trials = Number(q.get('trials') ?? 24);
  const brain = await loadBrain();
  const ref = await (await fetch(`${import.meta.env.BASE_URL}brain/vision_reference.json`)).json();
  const v = await (await fetch(`${import.meta.env.BASE_URL}brain/vision.json`)).json();
  const sim = await GpuBrain.create(brain, trials);
  const d = v.drive;
  const groups = Object.entries(ref.groups as Record<string, number[]>);
  sim.setVision({
    driveNeuron: [...d.on, ...d.off, ...d.both],
    driveSlot: [...d.onCol.map((c: number) => 3 * c), ...d.offCol.map((c: number) => 3 * c + 1), ...d.bothCol.map((c: number) => 3 * c + 2)],
    counted: groups.flatMap(([, idx]) => idx),
  });
  const frames = Math.ceil(ref.t_ms / 0.1 / ref.frame_steps);
  const prob = new Float32Array(frames * ref.n_slots);
  ref.prob_sparse.frame.forEach((f: number, k: number) => { prob[f * ref.n_slots + ref.prob_sparse.slot[k]] = ref.prob_sparse.p[k]; });
  const modes = (q.get('mode') ?? 'active') === 'both' ? ['active', 'plain'] as const : [q.get('mode') as 'active' | 'plain' ?? 'active'];
  for (const mode of modes) {
    const r = await sim.run({ rates: { sugar: 0, bitter: 0, salt: 0, water: 0 }, trials, tMs: ref.t_ms, seed: 99, plain: mode === 'plain',
      drive: { frameSteps: ref.frame_steps, nSlots: ref.n_slots, prob } });
    let at = 0;
    const per: Record<string, number> = {};
    for (const [g, idx] of groups) { per[g] = r.counts.slice(at, at + idx.length).reduce((a, b) => a + b, 0) / trials; at += idx.length; }
    per.total = r.spikes.reduce((a, b) => a + b, 0) / trials;
    for (const [g, m] of Object.entries(per)) {
      const rs = ref.spikes[g];
      const se = Math.sqrt(rs.sd ** 2 / ref.trials + rs.sd ** 2 / trials) || 1;
      const z = (m - rs.mean) / se;
      log(`${mode} ${g.padEnd(12)} GPU ${m.toFixed(2).padStart(9)}  ref ${rs.mean.toFixed(2).padStart(9)} ± ${rs.sd.toFixed(2)}  z ${z.toFixed(2)}`);
      state.results.push({ name: g, mode, gpu: m, ref: rs.mean, z } as unknown as Out);
    }
    log(`${mode}: ${Math.round(r.wallMs)} ms for ${trials} trials${r.overflow ? '  OVERFLOW' : ''}`);
  }
  state.done = true;
  log('done');
}

async function main() {
  const q = new URLSearchParams(location.search);
  if (q.get('vision')) {
    window.__results = { done: false, results: [] };
    return vision(q, window.__results);
  }
  const trials = Number(q.get('trials') ?? 30);
  const modes = (q.get('mode') ?? 'both') === 'both' ? ['plain', 'active'] as const : [q.get('mode') as 'plain' | 'active'];
  const only = q.get('conds')?.replaceAll(' ', '+').split(','); // '+' in names arrives as a space
  const state: NonNullable<Window['__results']> = { done: false, results: [] };
  window.__results = state;

  const brain = await loadBrain();
  log(`brain: ${brain.manifest.neurons} neurons, ${brain.manifest.edges} edges`);
  const ref = await (await fetch(`${import.meta.env.BASE_URL}brain/reference.json`)).json();
  const adapter = await navigator.gpu.requestAdapter();
  state.adapter = `${adapter?.info.vendor} ${adapter?.info.architecture} ${adapter?.info.description}`;
  log(`adapter: ${state.adapter}`);
  const sim = await GpuBrain.create(brain, trials, adapter!);
  const conds: RefCondition[] = ref.conditions.filter((c: RefCondition) => !only || only.includes(c.name));

  for (const c of conds) {
    for (const mode of modes) {
      const r = await sim.run({ rates: c.rates, trials, tMs: c.t_ms, seed: 12345, plain: mode === 'plain' });
      const o: Out = {
        name: c.name, mode, rates: c.rates, trials, t_ms: c.t_ms,
        mn9: r.mn9Steps.map((s) => s.length), dnp01: r.dnp01Steps.map((s) => s.length),
        ate: r.mn9Steps.filter(ateIt).length, spikes: r.spikes, active: r.active,
        wall_ms: Math.round(r.wallMs), overflow: r.overflow,
      };
      state.results.push(o);
      const hz = o.mn9.reduce((a, b) => a + b, 0) / trials * 1000 / c.t_ms;
      log(`${c.name.padEnd(40)} ${mode.padEnd(6)} MN9 ${hz.toFixed(1).padStart(6)} Hz  ate ${o.ate}/${trials}  ` +
        `spikes/trial ${Math.round(o.spikes.reduce((a, b) => a + b, 0) / trials)}  ${o.wall_ms} ms${o.overflow ? '  OVERFLOW' : ''}`);
    }
  }
  state.done = true;
  log('done');
}

main().catch((e) => {
  log('ERROR ' + (e?.stack ?? e));
  window.__results = { ...(window.__results ?? { results: [] }), done: true, error: String(e) };
});
