// Shiu et al. (2024) leaky integrate-and-fire parameters, identical to model.py and prep/refsim.py.

export const DT_MS = 0.1;
export const V0 = -52;
export const V_RST = -52;
export const V_TH = -45;
export const T_MBR = 20;
export const TAU = 5;
export const REFRACTORY_STEPS = 22; // 2.2 ms
export const DELAY_STEPS = 18; // 1.8 ms
export const W_SYN = 0.275; // mV per synapse
export const KICK = 250 * W_SYN; // one Poisson event, mV on v (always forces a spike)

// Exact solution of the linear ODE over one step ("linear" method in Brian2).
export const A_V = Math.exp(-DT_MS / T_MBR);
export const A_G = Math.exp(-DT_MS / TAU);
export const C_G = (TAU / (TAU - T_MBR)) * (A_G - A_V);

/** Eat rule (as in flight-or-bite): at least 4 MN9 spikes within any 50 ms window. */
export const EAT_SPIKES = 4;
export const EAT_WINDOW_MS = 50;

export function ateIt(mn9Steps: number[]): boolean {
  if (mn9Steps.length < EAT_SPIKES) return false;
  const s = [...mn9Steps].sort((a, b) => a - b);
  const w = Math.round(EAT_WINDOW_MS / DT_MS);
  for (let i = EAT_SPIKES - 1; i < s.length; i++) if (s[i] - s[i - EAT_SPIKES + 1] < w) return true;
  return false;
}
