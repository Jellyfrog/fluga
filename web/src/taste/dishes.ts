// The dish vocabulary (prep/dishes/*.json, built by prep/classifier/build_web.py): about 1,400 dishes,
// each with a taste computed from USDA FNDDS nutrients (sugars, sodium, water, caffeine, theobromine,
// bitter ingredients). Shared by the photo recognizer and the typed-name search.

/** Drive of the four fly gustatory channels, each 0..1. */
export interface Taste { sweet: number; bitter: number; salty: number; watery: number }

export interface DishInfo {
  id: string;
  name: string;
  aliases: string[];
  taste: Taste;
  per100g: { sugar_g: number; sodium_mg: number; water_g: number; caffeine_mg: number; theobromine_mg: number; bitter_share: number };
  usda: [description: string, grams: number][];
}

export const DISHES_BASE = new URL(`${import.meta.env.BASE_URL}models/dishes/`, document.baseURI).href;

let table: Promise<{ dim: number; dishes: DishInfo[]; byId: Map<string, DishInfo> }> | null = null;

export function loadDishes() {
  table ??= fetch(DISHES_BASE + 'dishes.json')
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`dishes: HTTP ${r.status}`))))
    .then((t: { dim: number; dishes: DishInfo[] }) => ({ ...t, byId: new Map(t.dishes.map((d) => [d.id, d])) }))
    .catch((err) => { table = null; throw err; });
  return table;
}

/** Gram-free weighted mean of dish tastes, e.g. the recognizer's top guesses weighted by score. */
export function blendTaste(parts: { taste: Taste; weight: number }[]): Taste {
  const total = parts.reduce((s, p) => s + p.weight, 0) || 1;
  const out: Taste = { sweet: 0, bitter: 0, salty: 0, watery: 0 };
  for (const p of parts) for (const k of Object.keys(out) as (keyof Taste)[]) out[k] += (p.taste[k] * p.weight) / total;
  return out;
}

/** "USDA, per 100 g: 18.7 g sugar · 173 mg sodium · 34 mg caffeine". Only what moves a channel. */
export function nutrientLine(d: DishInfo): string {
  const n = d.per100g;
  const parts = [`${n.sugar_g} g sugar`, `${n.sodium_mg} mg sodium`, `${Math.round(n.water_g)} g water`];
  if (n.caffeine_mg) parts.push(`${n.caffeine_mg} mg caffeine`);
  if (n.theobromine_mg) parts.push(`${n.theobromine_mg} mg theobromine`);
  if (n.bitter_share >= 0.02) parts.push(`${Math.round(n.bitter_share * 100)}% bitter ingredients`);
  return `USDA, per 100 g: ${parts.join(' · ')}`;
}

/** "Tiramisu" or "Pancakes, plain (150 g) + Syrup (30 g)". */
export function sourceLine(d: DishInfo): string {
  if (d.usda.length === 1) return `USDA food: ${d.usda[0][0]}`;
  return `Built from USDA foods: ${d.usda.map(([desc, g]) => `${desc} (${g} g)`).join(' + ')}`;
}

const norm = (s: string) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9 ]/g, ' ').replace(/\s+/g, ' ').trim();

/** Dishes whose name or alias matches the typed text: prefix matches first, then word and substring matches. */
export async function searchDishes(query: string, limit = 8): Promise<DishInfo[]> {
  const q = norm(query);
  if (!q) return [];
  const { dishes } = await loadDishes();
  const scored: { d: DishInfo; s: number }[] = [];
  for (const d of dishes) {
    let best = Infinity;
    for (const [i, n] of [d.name, ...d.aliases].map(norm).entries()) {
      const alias = i > 0 ? 0.5 : 0;
      if (n === q) best = Math.min(best, 0 + alias);
      else if (n.startsWith(q)) best = Math.min(best, 1 + alias);
      else if (n.includes(` ${q}`)) best = Math.min(best, 2 + alias);
      else if (n.includes(q)) best = Math.min(best, 3 + alias);
    }
    if (best < Infinity) scored.push({ d, s: best + d.name.length / 1000 });
  }
  return scored.sort((a, b) => a.s - b.s).slice(0, limit).map((x) => x.d);
}
