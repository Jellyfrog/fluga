// Hand-written Food-101 taste table: what a fly's gustatory receptor neurons would plausibly get from each dish.

import table from './tastes.json';

/** Drive of the four fly gustatory channels, each 0..1. */
export interface Taste { sweet: number; bitter: number; salty: number; watery: number }

type Entry = Taste & { notes: string };
const TASTES: Record<string, Entry> = table;

/** Mildly salty, slightly moist: an unremarkable savoury dish. */
const NEUTRAL: Entry = { sweet: 0.1, bitter: 0.1, salty: 0.3, watery: 0.2, notes: 'Unknown dish: a neutral, mildly savoury guess.' };

/** Taste profile for a Food-101 label (snake_case); unknown labels get a neutral profile. */
export function tasteFor(label: string): Entry {
  return { ...(TASTES[label] ?? NEUTRAL) };
}

/** "caesar_salad" -> "Caesar salad". */
export function prettyName(label: string): string {
  const s = label.replace(/_/g, ' ').trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}
