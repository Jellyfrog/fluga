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

async function main() {
  const q = new URLSearchParams(location.search);
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
