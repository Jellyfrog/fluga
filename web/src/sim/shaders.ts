// WGSL kernels for the spike-driven LIF brain. One 0.1 ms step = integrate(s) then propagate(s).
//
// Brian2's order within step s is: state update -> threshold -> synapses (+ Poisson kicks) -> reset.
// Here the synaptic input and kicks of step s, and the reset of neurons that fired in step s, are
// applied at the start of integrate(s + 1). That is equivalent because nothing reads v or g in
// between, and it saves a full pass over the neurons per step:
//   - spikes of step s-18 are added to `pending` (integer synapse counts) by propagate(s), but only
//     for targets that are not refractory: Brian2 drops input to refractory neurons;
//   - a neuron that fired in step s (last == s) is reset at integrate(s+1), discarding that step's
//     pending input and kick, exactly as Brian2's reset after the synapses slot does.
//
// Buffers (k = trial * N + neuron):
//   neurons  {v, g, last}                     per k
//   pending  atomic<i32> synapse counts       per k
//   active   [B*N active lists | B*N/32 flag words]   neurons that have left exact rest, per trial
//   graph    [rowPtr (N+1) | info (N) | post (E)]
//            info = channel mask (bits 0-3) | watch (4-5) | drive slot + 1 (6-18) | count index + 1 (19-30)
//   weights  Int16 pairs packed in u32
//   lists    [ring 20 x RING_CAP | record REC_CAP | raster 2 x RAST_CAP | drive DRIVE_CAP]
//            drive = Poisson probability per step, frame-major (frame * nSlots + slot), as f32 bits
//   counters [activeCount B | spikes B | mn9 B | dnp01 B | (at 4*MAX_B) recCount | rastCount | overflow
//            | (at 4*MAX_B + 8) spike count per count index, summed over trials]
//   ringArgs 20 x (x, 1, 1) indirect dispatch args: spikes per ring slot
//   k1Args   (x, B, 1) indirect args of integrate in active mode

export const MAX_B = 64;
export const RING_SLOTS = 20;
export const MAX_SLOTS = 8191; // 13 info bits
export const MAX_COUNTED = 4095;

const common = /* wgsl */ `
struct Params {
  n: u32, b: u32, seed: u32, plain: u32,
  ringCap: u32, recCap: u32, rastCap: u32, refrSteps: u32,
  v0: f32, vrst: f32, vth: f32, av: f32,
  ag: f32, cg: f32, wsyn: f32, kick: f32,
  rasterTrial: u32, drive: u32, frameSteps: u32, nSlots: u32,  // drive: 1 = per-neuron input on
  prob: array<vec4f, ${MAX_B}>,   // Poisson probability per step: sugar, bitter, salt, water
}
struct Step { s: u32 }
struct Neuron { v: f32, g: f32, last: i32 }

@group(0) @binding(0) var<uniform> P: Params;
@group(1) @binding(0) var<uniform> S: Step;

fn driveSlot(info: u32) -> u32 { return (info >> 6u) & 0x1FFFu; }  // 0 = not driven, else slot + 1
fn countIndex(info: u32) -> u32 { return (info >> 19u) & 0xFFFu; } // 0 = not counted, else index + 1

// Neurons of a channel driven at rate > 0, and per-neuron driven neurons while the drive is on, are
// Poisson sources with no refractory period (the model's 'poi' neurons), even in frames at rate 0.
fn stimulated(info: u32, p: vec4f) -> bool {
  let on = vec4u(select(vec4u(0u), vec4u(1u), p > vec4f(0.0)));
  let m = vec4u(info & 1u, (info >> 1u) & 1u, (info >> 2u) & 1u, (info >> 3u) & 1u);
  return dot(on, m) > 0u || (P.drive == 1u && driveSlot(info) != 0u);
}
fn refractory(info: u32, p: vec4f) -> i32 {
  return select(i32(P.refrSteps), 0, stimulated(info, p));
}
`;

const counters = /* wgsl */ `
const C_SPIKES = 1u;  // offsets in units of B
const C_MN9 = 2u;
const C_DN = 3u;
fn cGlobal(i: u32) -> u32 { return 4u * ${MAX_B}u + i; }  // 0 = recCount, 1 = rastCount, 2 = overflow, 8+ = counts
`;

export const integrateWGSL = common + counters + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(2) var<storage, read_write> pending: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> act: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> lists: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> graph: array<u32>;
@group(0) @binding(7) var<storage, read_write> ringArgs: array<atomic<u32>>;

fn pcg(x: u32) -> u32 {
  let s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
// Counter-based uniform in [0, 1): independent stream per (seed, trial, neuron, step, channel).
fn rand(trial: u32, i: u32, step: u32, ch: u32) -> f32 {
  let h = pcg(P.seed ^ pcg((trial * 0x9E3779B9u) ^ pcg(i ^ pcg(step * 4u + ch))));
  return f32(h >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let trial = gid.y;
  let s = S.s;
  let si = i32(s);
  // Free the ring slot that propagate(s-1) consumed; integrate(s+1) writes it next.
  if (gid.x == 0u && trial == 0u) { atomicStore(&ringArgs[((s + 1u) % ${RING_SLOTS}u) * 3u], 0u); }

  var i: u32;
  if (P.plain == 1u) {
    i = gid.x;
    if (i >= P.n) { return; }
  } else {
    if (gid.x >= atomicLoad(&counters[trial])) { return; }
    i = atomicLoad(&act[trial * P.n + gid.x]);
  }
  let k = trial * P.n + i;
  var nr = neurons[k];
  let info = graph[P.n + 1u + i];
  let p = P.prob[trial];

  // End of step s-1: synaptic input and Poisson kicks, or the reset if it fired in s-1.
  let syn = atomicExchange(&pending[k], 0);
  if (nr.last == si - 1) {
    nr.v = P.vrst;
    nr.g = 0.0;
  } else {
    nr.g += f32(syn) * P.wsyn;
    if (s > 0u && stimulated(info, p)) {
      for (var c = 0u; c < 4u; c++) {
        if (((info >> c) & 1u) == 1u && rand(trial, i, s - 1u, c) < p[c]) { nr.v += P.kick; }
      }
      let slot = driveSlot(info);
      if (P.drive == 1u && slot != 0u) {
        let at = ${RING_SLOTS}u * P.ringCap + P.recCap + 2u * P.rastCap + ((s - 1u) / P.frameSteps) * P.nSlots + slot - 1u;
        if (rand(trial, i, s - 1u, 4u) < bitcast<f32>(lists[at])) { nr.v += P.kick; }
      }
    }
  }

  // Step s: exact update unless refractory, then threshold.
  if (si - nr.last >= refractory(info, p)) {
    nr.v = P.v0 + (nr.v - P.v0) * P.av + nr.g * P.cg;
    nr.g *= P.ag;
    if (nr.v > P.vth) {
      nr.last = si;
      let slot = s % ${RING_SLOTS}u;
      let idx = atomicAdd(&ringArgs[slot * 3u], 1u);
      if (idx < P.ringCap) {
        lists[slot * P.ringCap + idx] = (trial << 18u) | i;
      } else {
        atomicAdd(&ringArgs[slot * 3u], 0xFFFFFFFFu);  // keep the dispatch within the list
        atomicStore(&counters[cGlobal(2u)], 1u);
      }
      atomicAdd(&counters[C_SPIKES * P.b + trial], 1u);
      let ci = countIndex(info);
      if (ci != 0u) { atomicAdd(&counters[cGlobal(7u + ci)], 1u); }
      let watch = (info >> 4u) & 3u;
      if (watch != 0u) {
        atomicAdd(&counters[select(C_DN, C_MN9, watch == 1u) * P.b + trial], 1u);
        let r = atomicAdd(&counters[cGlobal(0u)], 1u);
        if (r < P.recCap) { lists[${RING_SLOTS}u * P.ringCap + r] = s | (trial << 12u) | (watch << 28u); }
      }
      if (trial == P.rasterTrial) {
        let r = atomicAdd(&counters[cGlobal(1u)], 1u);
        if (r < P.rastCap) {
          let base = ${RING_SLOTS}u * P.ringCap + P.recCap + 2u * r;
          lists[base] = i;
          lists[base + 1u] = s;
        }
      }
    }
  }
  neurons[k] = nr;
}
`;

export const propagateWGSL = common + /* wgsl */ `
@group(0) @binding(1) var<storage, read> neurons: array<Neuron>;
@group(0) @binding(2) var<storage, read_write> pending: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> act: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> lists: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> graph: array<u32>;
@group(0) @binding(7) var<storage, read> weights: array<u32>;
@group(0) @binding(8) var<storage, read_write> k1Args: array<atomic<u32>>;

fn weight(e: u32) -> i32 {
  let w = weights[e >> 1u] >> ((e & 1u) * 16u);
  return i32(w << 16u) >> 16u;  // sign-extend the Int16
}

// Puts neuron t of trial into its active list the first time it receives anything.
fn activate(trial: u32, t: u32, k: u32) {
  let word = P.b * P.n + (k >> 5u);
  let bit = 1u << (k & 31u);
  if ((atomicOr(&act[word], bit) & bit) == 0u) {
    let idx = atomicAdd(&counters[trial], 1u);
    atomicStore(&act[trial * P.n + idx], t);
    atomicMax(&k1Args[0], idx / 64u + 1u);
  }
}

// One workgroup per spike emitted 18 steps ago (dispatched indirectly with that slot's count).
@compute @workgroup_size(64)
fn propagate(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let s = S.s;
  let slot = (s + ${RING_SLOTS}u - ${18}u) % ${RING_SLOTS}u;
  let e = lists[slot * P.ringCap + wg.x];
  let trial = e >> 18u;
  let j = e & 0x3FFFFu;
  let start = graph[j];
  let end = graph[j + 1u];
  let postBase = 2u * P.n + 1u;
  let base = trial * P.n;
  let p = P.prob[trial];
  for (var x = start + lid; x < end; x += 64u) {
    let t = graph[postBase + x];
    let k = base + t;
    // Input to a refractory target is dropped (Brian2 gates g += w by not_refractory).
    if (i32(s) - neurons[k].last >= refractory(graph[P.n + 1u + t], p)) {
      atomicAdd(&pending[k], weight(x));
      activate(trial, t, k);
    }
  }
}
`;

/** Resets all per-trial state and seeds the active lists with the stimulated neurons. */
export const initWGSL = common + /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(3) var<storage, read_write> act: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> graph: array<u32>;
@group(0) @binding(8) var<storage, read_write> k1Args: array<atomic<u32>>;

@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let trial = gid.y;
  if (i >= P.n) { return; }
  let k = trial * P.n + i;
  neurons[k] = Neuron(P.v0, 0.0, -1000000);
  if (stimulated(graph[P.n + 1u + i], P.prob[trial])) {
    atomicOr(&act[P.b * P.n + (k >> 5u)], 1u << (k & 31u));
    let idx = atomicAdd(&counters[trial], 1u);
    atomicStore(&act[trial * P.n + idx], i);
    atomicMax(&k1Args[0], idx / 64u + 1u);
  }
}
`;
