// Taste profile -> firing rates of the four gustatory channels, and the eat/decline decision.

import type { Rates } from '../sim/gpu-sim';
import type { Taste } from './tastes';

export const MAX_RATE_HZ = 150; // the range the reference model was tested at
export const TRIALS = 30;
export const TRIAL_MS = 250;
export const MIN_ATE = 5; // below this for every dish, the fly declines the whole menu

export type Mood = 'full' | 'peckish' | 'starving';
export const MOOD_SUGAR_GAIN: Record<Mood, number> = { full: 0.6, peckish: 1, starving: 1.4 };

export function tasteToRates(t: Taste, mood: Mood): Rates {
  const c = (x: number) => Math.min(1, Math.max(0, x));
  return {
    sugar: c(t.sweet) * MAX_RATE_HZ * MOOD_SUGAR_GAIN[mood],
    bitter: c(t.bitter) * MAX_RATE_HZ,
    salt: c(t.salty) * MAX_RATE_HZ,
    water: c(t.watery) * MAX_RATE_HZ,
  };
}

export interface DishOutcome {
  ate: number; // trials with ≥ 4 MN9 spikes within 50 ms
  trials: number;
  mn9Hz: number; // mean MN9 rate
  fled: boolean; // DNp01 (giant fibre) fired
}

export interface Ranked<T> { item: T; outcome: DishOutcome; rank: number }

/** Most "ate it" trials wins; ties go to the higher mean MN9 rate. Fled dishes rank last. */
export function rank<T>(dishes: { item: T; outcome: DishOutcome }[]): { ranked: Ranked<T>[]; winner: Ranked<T> | null } {
  const score = (o: DishOutcome) => (o.fled ? -1 : o.ate);
  const ranked = [...dishes]
    .sort((a, b) => score(b.outcome) - score(a.outcome) || b.outcome.mn9Hz - a.outcome.mn9Hz)
    .map((d, i) => ({ ...d, rank: i + 1 }));
  const top = ranked[0];
  return { ranked, winner: top && !top.outcome.fled && top.outcome.ate >= MIN_ATE ? top : null };
}

/** A one-liner built from the result and the taste that went in. */
export function oneLiner(name: string, t: Taste, o: DishOutcome, isWinner: boolean,
  winner?: { name: string; outcome: DishOutcome }): string {
  const dish = name.toLowerCase();
  if (o.fled) return `Fled from the ${dish}: the giant fibre fired before the proboscis did.`;
  if (isWinner) {
    if (o.ate === o.trials) return `Devoured the ${dish}: MN9 fired in every single tasting.`;
    if (t.bitter >= 0.3) return `Ate the ${dish} anyway: the sugar outvoted the bitterness.`;
    if (t.sweet < 0.3 && t.watery >= 0.5) return `Drank the ${dish}: water neurons alone opened the proboscis.`;
    return `Chose the ${dish}: sweet enough to extend the proboscis ${o.ate} times out of ${o.trials}.`;
  }
  if (o.ate === 0) {
    if (t.sweet < 0.25 && t.watery < 0.6) return `Ignored the ${dish}: nothing sweet reached the sugar neurons.`;
    if (t.bitter >= 0.3 && t.bitter >= t.salty) return `Refused the ${dish}: bitterness overruled everything.`;
    if (t.salty >= 0.4) return `Refused the ${dish}: the salt neurons shut the proboscis.`;
    return `Refused the ${dish}: not enough to cross MN9's threshold.`;
  }
  if (winner && o.ate === winner.outcome.ate) {
    const why = t.bitter >= 0.3 ? ', bitterness took the edge off' : '';
    return `Ate the ${dish} ${o.ate} of ${o.trials} times too, but MN9 fired harder for the ${winner.name.toLowerCase()} `
      + `(${winner.outcome.mn9Hz.toFixed(0)} vs ${o.mn9Hz.toFixed(0)} Hz)${why}.`;
  }
  if (t.bitter >= 0.3) return `Tempted by the ${dish} (${o.ate}/${o.trials}), but the bitterness held it back.`;
  if (t.salty >= 0.4) return `Liked the ${dish} (${o.ate}/${o.trials}), but it was too salty to commit.`;
  return `Tasted the ${dish} and ate it ${o.ate} times out of ${o.trials}.`;
}
