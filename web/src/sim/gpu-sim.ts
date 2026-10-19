// The whole-brain simulator on WebGPU: many trials of one tasting side by side.

import type { Brain, Channel } from './brain';
import { CHANNELS } from './brain';
import * as M from './model';
import { initWGSL, integrateWGSL, propagateWGSL, MAX_B, MAX_COUNTED, MAX_SLOTS, RING_SLOTS } from './shaders';

export type Rates = Record<Channel, number>; // Hz per taste channel

export interface TastingRequest {
  rates: Rates;
  trials: number; // ≤ maxTrials
  tMs: number; // simulated time per trial
  seed: number;
  plain?: boolean; // update every neuron each step instead of only active ones (validation)
  drive?: Drive; // per-neuron Poisson input (vision mode); needs setVision first
}

/** Time-varying Poisson input to the slots of setVision: prob[frame * nSlots + slot] per 0.1 ms step. */
export interface Drive { frameSteps: number; nSlots: number; prob: Float32Array }

/** Which neurons get which drive slot, and which neurons' spikes are counted (in this order). */
export interface VisionMap { driveNeuron: ArrayLike<number>; driveSlot: ArrayLike<number>; counted: ArrayLike<number> }

/** Live progress after each chunk of steps. Raster spikes are from trial 0. */
export interface ChunkUpdate {
  step: number; nSteps: number;
  mn9: number[]; // spikes per trial so far
  spikes: number[]; // all spikes per trial so far
  active: number[]; // neurons touched per trial so far
  raster: { neuron: number; step: number }[]; // new raster spikes since the last update
}

export interface TastingResult {
  nSteps: number;
  mn9Steps: number[][]; // per trial, steps at which MN9 fired
  dnp01Steps: number[][];
  spikes: number[];
  active: number[];
  wallMs: number;
  overflow: boolean;
  counts: number[]; // spikes of each counted neuron (setVision), summed over trials
}

const RING_CAP = 8192;
const REC_CAP = 1 << 16;
const RAST_CAP = 1 << 15;
const STEPS_PER_SUBMIT = 100;
const STEP_STRIDE = 256; // minUniformBufferOffsetAlignment
const DRIVE_CAP = 48 * 8192; // frames x slots
const LIST_WORDS = RING_SLOTS * RING_CAP + REC_CAP + 2 * RAST_CAP + DRIVE_CAP;
const COUNT_BASE = 4 * MAX_B + 8; // counters word where the per-neuron counts start

export class GpuBrain {
  readonly n: number;
  readonly maxTrials: number;
  private device: GPUDevice;
  private bufs!: Record<'params' | 'steps' | 'neurons' | 'pending' | 'active' | 'graph' | 'weights' | 'lists'
    | 'counters' | 'ringArgs' | 'k1Args', GPUBuffer>;
  private pipes!: { init: GPUComputePipeline; integrate: GPUComputePipeline; propagate: GPUComputePipeline };
  private groups!: { init: GPUBindGroup; integrate: GPUBindGroup; propagate: GPUBindGroup; step: GPUBindGroup[][] };
  private maxSteps = 0;
  private busy = false;
  private info!: Uint32Array; // CPU copy of the graph's info words
  private nCounted = 0;

  private constructor(device: GPUDevice, n: number, maxTrials: number) {
    this.device = device;
    this.n = n;
    this.maxTrials = maxTrials;
  }

  static async create(brain: Brain, maxTrials = 30, adapter?: GPUAdapter): Promise<GpuBrain> {
    if (!('gpu' in navigator)) throw new Error('WebGPU is not available');
    adapter ??= (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ?? undefined;
    if (!adapter) throw new Error('no WebGPU adapter');
    const n = brain.manifest.neurons;
    const graphBytes = 4 * (2 * n + 1 + brain.manifest.edges);
    const trialBytes = n * (12 + 4 + 4) + 4 * Math.ceil(n / 32);
    // Fit the batch into what the adapter allows (and Safari will tolerate).
    const maxBind = adapter.limits.maxStorageBufferBindingSize;
    maxTrials = Math.min(maxTrials, MAX_B, Math.floor(Math.min(maxBind, adapter.limits.maxBufferSize) / (n * 12)));
    if (graphBytes > maxBind) throw new Error(`graph needs ${graphBytes} B bindings, adapter allows ${maxBind}`);
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.max(graphBytes, n * 12 * maxTrials),
        maxBufferSize: Math.max(graphBytes, n * 12 * maxTrials),
        maxStorageBuffersPerShaderStage: 8,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      },
    });
    device.lost.then((info) => console.warn('WebGPU device lost:', info.message));
    const sim = new GpuBrain(device, n, maxTrials);
    sim.upload(brain);
    await sim.build();
    console.info(`GpuBrain: ${maxTrials} trials, ${(trialBytes * maxTrials / 1e6).toFixed(0)} MB state`);
    return sim;
  }

  private upload(brain: Brain) {
    const d = this.device, n = this.n, B = this.maxTrials, E = brain.manifest.edges;
    const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;

    // graph = rowPtr | info | post
    const graph = d.createBuffer({ size: 4 * (2 * n + 1 + E), usage: S | CD, mappedAtCreation: true });
    const g = new Uint32Array(graph.getMappedRange());
    g.set(brain.rowPtr, 0);
    const info = g.subarray(n + 1, 2 * n + 1);
    CHANNELS.forEach((ch, c) => { for (const i of brain.groups[ch].indices) info[i] |= 1 << c; });
    for (const i of brain.groups.mn9.indices) info[i] |= 1 << 4;
    for (const i of brain.groups.dnp01.indices) info[i] |= 2 << 4;
    g.set(brain.post, 2 * n + 1);
    this.info = info.slice();
    graph.unmap();

    const weights = d.createBuffer({ size: 4 * Math.ceil(E / 2), usage: S, mappedAtCreation: true });
    new Int16Array(weights.getMappedRange()).set(brain.weight);
    weights.unmap();

    const listWords = LIST_WORDS;
    const counterWords = COUNT_BASE + MAX_COUNTED;
    this.bufs = {
      params: d.createBuffer({ size: 80 + 16 * MAX_B, usage: GPUBufferUsage.UNIFORM | CD }),
      steps: d.createBuffer({ size: 0, usage: GPUBufferUsage.UNIFORM | CD }), // sized in ensureSteps
      neurons: d.createBuffer({ size: 12 * n * B, usage: S }),
      pending: d.createBuffer({ size: 4 * n * B, usage: S | CD }),
      active: d.createBuffer({ size: 4 * (n * B + Math.ceil((n * B) / 32)), usage: S | CD }),
      graph,
      weights,
      lists: d.createBuffer({ size: 4 * listWords, usage: S | CS | CD }), // CD: the vision drive is written here
      counters: d.createBuffer({ size: 4 * counterWords, usage: S | CD | CS }),
      ringArgs: d.createBuffer({ size: 4 * 3 * RING_SLOTS, usage: S | GPUBufferUsage.INDIRECT | CD }),
      k1Args: d.createBuffer({ size: 16, usage: S | GPUBufferUsage.INDIRECT | CD }),
    };
  }

  private async build() {
    const d = this.device;
    const mk = (code: string, entryPoint: string) => d.createComputePipelineAsync({
      layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint },
    });
    const [init, integrate, propagate] = await Promise.all([
      mk(initWGSL, 'init'), mk(integrateWGSL, 'integrate'), mk(propagateWGSL, 'propagate')]);
    this.pipes = { init, integrate, propagate };
    const b = this.bufs;
    const bg = (p: GPUComputePipeline, entries: [number, GPUBuffer][]) => d.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
    this.groups = {
      init: bg(init, [[0, b.params], [1, b.neurons], [3, b.active], [5, b.counters], [6, b.graph], [8, b.k1Args]]),
      integrate: bg(integrate, [[0, b.params], [1, b.neurons], [2, b.pending], [3, b.active], [4, b.lists],
        [5, b.counters], [6, b.graph], [7, b.ringArgs]]),
      propagate: bg(propagate, [[0, b.params], [1, b.neurons], [2, b.pending], [3, b.active], [4, b.lists],
        [5, b.counters], [6, b.graph], [7, b.weights], [8, b.k1Args]]),
      step: [],
    };
  }

  /** Marks driven and counted neurons in the info words (vision mode). Replaces any earlier map. */
  setVision(map: VisionMap) {
    const n = this.n, info = this.info.map((w) => w & 0x3f);
    if (map.driveNeuron.length !== map.driveSlot.length) throw new Error('setVision: driveNeuron/driveSlot lengths differ');
    if (map.counted.length > MAX_COUNTED) throw new Error(`setVision: at most ${MAX_COUNTED} counted neurons`);
    for (let k = 0; k < map.driveNeuron.length; k++) {
      const i = map.driveNeuron[k], slot = map.driveSlot[k];
      if (i < 0 || i >= n || slot < 0 || slot >= MAX_SLOTS) throw new Error('setVision: drive index out of range');
      info[i] = (info[i] & ~(0x1fff << 6)) | ((slot + 1) << 6);
    }
    for (let k = 0; k < map.counted.length; k++) info[map.counted[k]] = (info[map.counted[k]] & ~(0xfff << 19)) | ((k + 1) << 19);
    this.info = info;
    this.nCounted = map.counted.length;
    this.device.queue.writeBuffer(this.bufs.graph, 4 * (n + 1), info);
  }

  /** One 16-byte uniform slot per step index, with a bind group per step and kernel. */
  private ensureSteps(nSteps: number) {
    if (nSteps <= this.maxSteps) return;
    const d = this.device;
    this.bufs.steps.destroy();
    const steps = d.createBuffer({ size: STEP_STRIDE * nSteps, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
    const u = new Uint32Array(steps.getMappedRange());
    for (let s = 0; s < nSteps; s++) u[(s * STEP_STRIDE) / 4] = s;
    steps.unmap();
    this.bufs.steps = steps;
    // 'auto' layouts allow no dynamic offsets, and are per pipeline: one bind group per step and kernel.
    const groups = (p: GPUComputePipeline) => Array.from({ length: nSteps }, (_, s) => d.createBindGroup({
      layout: p.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: steps, offset: s * STEP_STRIDE, size: 16 } }],
    }));
    this.groups.step = [groups(this.pipes.integrate), groups(this.pipes.propagate)];
    this.maxSteps = nSteps;
  }

  private writeParams(req: TastingRequest) {
    const buf = new ArrayBuffer(80 + 16 * MAX_B);
    const u = new Uint32Array(buf), f = new Float32Array(buf);
    u.set([this.n, req.trials, req.seed >>> 0, req.plain ? 1 : 0, RING_CAP, REC_CAP, RAST_CAP, M.REFRACTORY_STEPS]);
    f.set([M.V0, M.V_RST, M.V_TH, M.A_V, M.A_G, M.C_G, M.W_SYN, M.KICK], 8);
    u[16] = 0; // raster trial
    if (req.drive) {
      const { frameSteps, nSlots, prob } = req.drive;
      const nSteps = Math.round(req.tMs / M.DT_MS);
      if (nSlots < 1 || nSlots > MAX_SLOTS || frameSteps < 1) throw new Error('drive: bad nSlots or frameSteps');
      if (prob.length < Math.ceil(nSteps / frameSteps) * nSlots) throw new Error('drive: prob does not cover the trial');
      if (prob.length > DRIVE_CAP) throw new Error(`drive: at most ${DRIVE_CAP} frame x slot values`);
      if (prob.some((x) => !(x >= 0 && x <= 1))) throw new Error('drive: probabilities must be in [0, 1]');
      u.set([1, frameSteps, nSlots], 17);
      this.device.queue.writeBuffer(this.bufs.lists, 4 * (RING_SLOTS * RING_CAP + REC_CAP + 2 * RAST_CAP), prob);
    }
    CHANNELS.forEach((ch, c) => {
      const p = req.rates[ch] * M.DT_MS * 1e-3;
      if (p < 0 || p > 1) throw new Error(`${ch}: rate ${req.rates[ch]} Hz out of range`);
      for (let t = 0; t < req.trials; t++) f[20 + 4 * t + c] = p;
    });
    this.device.queue.writeBuffer(this.bufs.params, 0, buf);
  }

  async run(req: TastingRequest, onChunk?: (u: ChunkUpdate) => void): Promise<TastingResult> {
    if (this.busy) throw new Error('GpuBrain is busy');
    if (req.trials < 1 || req.trials > this.maxTrials) throw new Error(`trials must be 1..${this.maxTrials}`);
    this.busy = true;
    try {
      return await this.runInner(req, onChunk);
    } finally {
      this.busy = false;
    }
  }

  private async runInner(req: TastingRequest, onChunk?: (u: ChunkUpdate) => void): Promise<TastingResult> {
    const d = this.device, b = this.bufs, B = req.trials, n = this.n;
    const nSteps = Math.round(req.tMs / M.DT_MS);
    if (nSteps > 4096) throw new Error('at most 409.6 ms per trial (12-bit step field in the spike record)');
    this.ensureSteps(nSteps);
    this.writeParams(req);
    const t0 = performance.now();

    // Reset counters and indirect args; init kernel resets neurons and seeds active lists.
    const ring = new Uint32Array(3 * RING_SLOTS);
    for (let s = 0; s < RING_SLOTS; s++) ring.set([0, 1, 1], 3 * s);
    d.queue.writeBuffer(b.ringArgs, 0, ring);
    d.queue.writeBuffer(b.k1Args, 0, new Uint32Array([0, B, 1, 0]));
    {
      const enc = d.createCommandEncoder();
      enc.clearBuffer(b.counters);
      enc.clearBuffer(b.pending, 0, 4 * n * B);
      enc.clearBuffer(b.active);
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipes.init);
      pass.setBindGroup(0, this.groups.init);
      pass.dispatchWorkgroups(Math.ceil(n / 64), B);
      pass.end();
      d.queue.submit([enc.finish()]);
    }

    // Readback ring: two staging buffers so the GPU keeps working while we read the last chunk.
    const counterBytes = b.counters.size;
    const listBase = 4 * RING_SLOTS * RING_CAP;
    const listBytes = 4 * (REC_CAP + 2 * RAST_CAP);
    const staging = [0, 1].map(() => d.createBuffer({
      size: counterBytes + listBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    let rastSeen = 0;
    let last: { counters: Uint32Array; lists: Uint32Array } | null = null;

    const read = async (buf: GPUBuffer, step: number) => {
      await buf.mapAsync(GPUMapMode.READ);
      const all = new Uint32Array(buf.getMappedRange().slice(0));
      buf.unmap();
      const counters = all.subarray(0, counterBytes / 4);
      const lists = all.subarray(counterBytes / 4);
      last = { counters, lists };
      if (onChunk) {
        const rastCount = Math.min(counters[4 * MAX_B + 1], RAST_CAP);
        const raster = [];
        for (let r = rastSeen; r < rastCount; r++) raster.push({ neuron: lists[REC_CAP + 2 * r], step: lists[REC_CAP + 2 * r + 1] });
        rastSeen = rastCount;
        const slice = (o: number) => Array.from(counters.subarray(o * B, o * B + B));
        onChunk({ step, nSteps, active: slice(0), spikes: slice(1), mn9: slice(2), raster });
      }
    };

    const reads: (Promise<void> | null)[] = [null, null];
    for (let s0 = 0, chunk = 0; s0 < nSteps; s0 += STEPS_PER_SUBMIT, chunk++) {
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      const s1 = Math.min(nSteps, s0 + STEPS_PER_SUBMIT);
      for (let s = s0; s < s1; s++) {
        pass.setPipeline(this.pipes.integrate);
        pass.setBindGroup(0, this.groups.integrate);
        pass.setBindGroup(1, this.groups.step[0][s]);
        if (req.plain) pass.dispatchWorkgroups(Math.ceil(n / 64), B);
        else pass.dispatchWorkgroupsIndirect(b.k1Args, 0);
        pass.setPipeline(this.pipes.propagate);
        pass.setBindGroup(0, this.groups.propagate);
        pass.setBindGroup(1, this.groups.step[1][s]);
        pass.dispatchWorkgroupsIndirect(b.ringArgs, 12 * ((s + RING_SLOTS - M.DELAY_STEPS) % RING_SLOTS));
      }
      pass.end();
      const st = staging[chunk % 2];
      const isLast = s1 >= nSteps;
      if (onChunk || isLast) {
        await reads[chunk % 2]; // the staging buffer we are about to reuse must be unmapped
        enc.copyBufferToBuffer(b.counters, 0, st, 0, counterBytes);
        enc.copyBufferToBuffer(b.lists, listBase, st, counterBytes, listBytes);
      }
      d.queue.submit([enc.finish()]);
      if (onChunk || isLast) reads[chunk % 2] = read(st, s1);
    }
    await Promise.all(reads);
    staging.forEach((s) => s.destroy());

    const { counters, lists } = last!;
    const recCount = Math.min(counters[4 * MAX_B], REC_CAP);
    const mn9Steps: number[][] = Array.from({ length: B }, () => []);
    const dnp01Steps: number[][] = Array.from({ length: B }, () => []);
    for (let r = 0; r < recCount; r++) {
      const e = lists[r];
      const step = e & 0xfff, trial = (e >>> 12) & 0xffff, watch = e >>> 28;
      (watch === 1 ? mn9Steps : dnp01Steps)[trial].push(step);
    }
    const slice = (o: number) => Array.from(counters.subarray(o * B, o * B + B));
    return {
      nSteps, mn9Steps, dnp01Steps, spikes: slice(1), active: slice(0),
      wallMs: performance.now() - t0,
      overflow: counters[4 * MAX_B + 2] !== 0 || counters[4 * MAX_B] > REC_CAP,
      counts: Array.from(counters.subarray(COUNT_BASE, COUNT_BASE + this.nCounted)),
    };
  }

  destroy() {
    Object.values(this.bufs).forEach((b) => b.destroy());
    this.device.destroy();
  }
}
