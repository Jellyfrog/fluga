// Fluga app: wake the fly, add dishes, watch it taste, read the verdict.

import { FlyEye, flyEyeSupported, FLY_EYE_LEGEND } from './flyeye/flyeye';
import { startBrain, isRecorded, type BrainService } from './sim/service';
import { ateIt, DT_MS } from './sim/model';
import { classify, loadClassifier, type Guess } from './taste/classifier';
import { blendTaste, nutrientLine, searchDishes, sourceLine, type DishInfo, type Taste } from './taste/dishes';
import {
  MIN_ATE, TRIALS, TRIAL_MS, oneLiner, rank, tasteToRates, type DishOutcome, type Mood,
} from './taste/encoder';
import { Raster } from './ui/raster';
import { initLook } from './vision/look';
import { renderShareCard, shareOrDownload } from './ui/share';

const MAX_DISHES = 8;
// Softmax over ~1,400 dishes: below 0.25 the top guess is right about half the time, so ask
// (flags ~20% of photos; prep/classifier/eval_vocab.py).
const LOW_CONFIDENCE = 0.25;
const PHOTO_MAX = 768;
const N_NEURONS = 138_639;

interface Dish {
  id: number;
  photo: HTMLCanvasElement; // downscaled source, reused by classifier, fly eye and share card
  thumb: string; // data URL for lists
  guesses: Guess[] | null;
  label: string | null; // dish id
  name?: string;
  skipped: boolean;
  taste: Taste;
  notes: string;
  el: HTMLElement;
  eye?: FlyEye;
  outcome?: DishOutcome;
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (n: number) => n.toLocaleString('en-US');

function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> & { class?: string } = {},
  ...kids: (Node | string | null | undefined)[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  const { class: cls, ...rest } = props;
  if (cls) el.className = cls;
  Object.assign(el, rest);
  for (const k of kids) if (k != null) el.append(k);
  return el;
}

// ---------- state ----------
const state = {
  service: null as BrainService | null,
  mood: 'peckish' as Mood,
  dishes: [] as Dish[],
  nextId: 1,
  classifier: null as Promise<void> | null,
  mn9Index: -1,
  tasting: false,
};

// ---------- navigation ----------
type Screen = 'wake' | 'dishes' | 'tasting' | 'verdict' | 'look';
function go(screen: Screen) {
  if (screen !== 'wake' && !state.service) screen = 'wake';
  if (state.tasting && screen !== 'tasting') return;
  for (const s of ['wake', 'dishes', 'tasting', 'verdict', 'look'] as Screen[]) $(`screen-${s}`).hidden = s !== screen;
  if (screen === 'dishes') ensureClassifier();
  window.scrollTo({ top: 0 });
}
document.addEventListener('click', (e) => {
  const t = (e.target as HTMLElement).closest<HTMLElement>('[data-go]');
  if (t) go(t.dataset.go as Screen);
});

// ---------- 1. wake ----------
$('wake-btn').addEventListener('click', async () => {
  $('wake-idle').hidden = true;
  $('wake-loading').hidden = false;
  const bar = $('wake-bar'), count = $('wake-count');
  try {
    const groups = fetch(`${import.meta.env.BASE_URL}brain/groups.json`).then((r) => r.json());
    const { service, note } = await startBrain(TRIALS, (frac) => {
      bar.style.width = `${(frac * 100).toFixed(1)}%`;
      count.textContent = frac < 1 ? `${fmt(Math.floor(frac * N_NEURONS))} of ${fmt(N_NEURONS)} neurons`
        : 'Wiring 15,091,983 connections on the GPU…';
    });
    state.mn9Index = (await groups).groups.mn9.indices[0];
    state.service = service;
    const badge = $('brain-badge');
    badge.hidden = false;
    badge.className = `badge ${service.kind}`;
    badge.textContent = service.kind === 'live' ? 'live brain' : 'recorded brain';
    badge.title = service.kind === 'live' ? `Simulated live with WebGPU (${service.where})` : 'Precomputed runs of the same model';
    $('about-tech').textContent = service.kind === 'live'
      ? `This device runs the live brain with WebGPU (${service.where}).`
      : 'This device has no WebGPU, so verdicts come from a table of recorded runs of the same model.';
    $('wake-note').textContent = note ?? 'The fly is awake: 138,639 neurons ready to taste.';
    $('wake-loading').hidden = true;
    $('wake-ready').hidden = false;
  } catch (err) {
    $('wake-loading').hidden = true;
    $('wake-idle').hidden = false;
    const e = $('wake-error');
    e.hidden = false;
    e.textContent = `The fly would not wake up: ${(err as Error).message}`;
  }
});
document.querySelectorAll<HTMLInputElement>('input[name=mood]').forEach((r) =>
  r.addEventListener('change', () => { if (r.checked) state.mood = r.value as Mood; }));

// ---------- 2. dishes ----------
function ensureClassifier() {
  if (state.classifier) return state.classifier;
  const status = $('classifier-status');
  status.textContent = 'Loading the dish recognizer…';
  state.classifier = loadClassifier((f) => { status.textContent = `Loading the dish recognizer… ${Math.round(f * 100)}%`; })
    .then(() => { status.textContent = ''; })
    .catch((err) => {
      status.textContent = 'Dish recognizer unavailable; set the taste bars by hand.';
      console.warn(err);
    });
  return state.classifier;
}

async function toPhoto(file: File): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  // Halve in steps: one large drawImage downscale aliases badly and confuses the classifier.
  let src: CanvasImageSource = bmp, w = bmp.width, hgt = bmp.height;
  const target = Math.min(1, PHOTO_MAX / Math.max(w, hgt));
  for (;;) {
    const s = Math.max(target * (bmp.width / w), 0.5);
    const c = h('canvas', { width: Math.round(w * s), height: Math.round(hgt * s) });
    const g = c.getContext('2d')!;
    g.imageSmoothingQuality = 'high';
    g.drawImage(src, 0, 0, c.width, c.height);
    src = c; w = c.width; hgt = c.height;
    if (s > 0.5 || w <= PHOTO_MAX && hgt <= PHOTO_MAX) break;
  }
  bmp.close();
  return src as HTMLCanvasElement;
}

function thumbOf(photo: HTMLCanvasElement): string {
  const t = h('canvas', { width: 160, height: 160 });
  const s = Math.min(photo.width, photo.height);
  t.getContext('2d')!.drawImage(photo, (photo.width - s) / 2, (photo.height - s) / 2, s, s, 0, 0, 160, 160);
  return t.toDataURL('image/jpeg', 0.8);
}

async function addFiles(files: FileList | null) {
  if (!files) return;
  for (const file of Array.from(files)) {
    if (state.dishes.length >= MAX_DISHES) break;
    if (!file.type.startsWith('image/')) continue;
    let photo: HTMLCanvasElement;
    try { photo = await toPhoto(file); } catch { continue; }
    const dish: Dish = {
      id: state.nextId++, photo, thumb: thumbOf(photo), guesses: null, label: null, skipped: false,
      taste: { sweet: 0, bitter: 0, salty: 0, watery: 0 }, notes: '', el: h('article', { class: 'card dish' }),
    };
    state.dishes.push(dish);
    $('dishes').append(dish.el);
    renderDish(dish);
    recognize(dish);
  }
  updateDishesUi();
}
$<HTMLInputElement>('file-camera').addEventListener('change', (e) => { addFiles((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ''; });
$<HTMLInputElement>('file-upload').addEventListener('change', (e) => { addFiles((e.target as HTMLInputElement).files); (e.target as HTMLInputElement).value = ''; });

async function recognize(dish: Dish) {
  await ensureClassifier();
  try {
    const { top } = await classify(dish.photo);
    dish.guesses = top;
    if (top[0].score >= LOW_CONFIDENCE) chooseDish(dish, top[0].dish, top);
  } catch (err) {
    console.warn('classify failed', err);
    dish.guesses = [];
  }
  renderDish(dish);
  updateDishesUi();
}

/**
 * Picks the dish. When the recognizer chose it, the taste blends the top guesses by score, so a
 * near-miss (prime rib vs. steak) changes little; a dish the user picked or typed counts alone.
 */
function chooseDish(dish: Dish, info: DishInfo, guesses?: Guess[]) {
  dish.label = info.id;
  dish.name = info.name;
  dish.skipped = false;
  const lines = [sourceLine(info), nutrientLine(info)];
  if (guesses && guesses.length > 1) {
    dish.taste = blendTaste(guesses.map((g) => ({ taste: g.dish.taste, weight: g.score })));
    const total = guesses.reduce((s, g) => s + g.score, 0);
    const shown = guesses.map((g) => ({ g, pct: Math.round((100 * g.score) / total) })).filter((x) => x.pct >= 1);
    lines.push(`Taste blended from the top guesses: ${shown.map(({ g, pct }) => `${g.name} ${pct}%`).join(', ')}.`);
  } else {
    dish.taste = { ...info.taste };
  }
  dish.notes = lines.join('\n');
}

function dishName(d: Dish) { return d.name ?? 'Mystery dish'; }

/** "Not it? Type the dish": search the vocabulary by name or alias and pick a match. */
function dishSearch(dish: Dish): HTMLElement {
  const input = h('input', { type: 'search', placeholder: 'Not it? Type the dish…', ariaLabel: 'Type the dish name', className: 'search' });
  const results = h('div', { class: 'guesses' });
  let seq = 0;
  input.addEventListener('input', async () => {
    const mine = ++seq;
    const found = await searchDishes(input.value, 6);
    if (mine !== seq) return;
    results.replaceChildren(...found.map((d) => {
      const b = h('button', { class: 'chip' }, d.name);
      b.addEventListener('click', () => { chooseDish(dish, d); renderDish(dish); updateDishesUi(); });
      return b;
    }));
    if (input.value.trim() && !found.length) results.append(h('span', { class: 'fine' }, 'No dish by that name; pick the closest or set the bars by hand.'));
  });
  return h('div', { class: 'dish-search' }, input, results);
}

function renderDish(dish: Dish) {
  const el = dish.el;
  el.classList.toggle('skipped', dish.skipped);
  const keep = el.querySelector('.view');
  const view = (keep as HTMLElement) ?? buildView(dish);
  el.replaceChildren(view);

  if (dish.eye) {
    const slider = h('input', { type: 'range', min: '0', max: '1', step: '0.01', value: '1', ariaLabel: 'Human to fly view' });
    const cur = el.dataset.mix ?? '1';
    slider.value = cur;
    slider.addEventListener('input', () => { el.dataset.mix = slider.value; dish.eye!.render(Number(slider.value)); });
    el.append(h('div', { class: 'eye' }, 'You', slider, 'Fly'));
  }

  const body = h('div', { class: 'body' });
  if (!dish.guesses) {
    body.append(h('p', { class: 'name' }, 'Recognizing…'));
  } else {
    const top = dish.guesses[0];
    const chosen = dish.guesses.find((g) => g.label === dish.label);
    body.append(h('p', { class: 'name' }, dish.skipped ? 'Skipped' : dishName(dish)));
    if (chosen) body.append(h('span', { class: 'conf' }, `${Math.round(chosen.score * 100)}% sure`));
    if (dish.guesses.length && (top.score < LOW_CONFIDENCE || !dish.label || dish.skipped)) {
      body.append(h('p', { class: 'fine' }, dish.skipped ? 'The fly sniffed it suspiciously and moved on.'
        : 'Not sure what this is. Pick the closest:'));
      const chips = h('div', { class: 'guesses' });
      for (const g of dish.guesses.slice(0, 3)) {
        const b = h('button', { class: `chip${g.label === dish.label ? ' on' : ''}` }, `${g.name} · ${Math.round(g.score * 100)}%`);
        b.addEventListener('click', () => { chooseDish(dish, g.dish); renderDish(dish); updateDishesUi(); });
        chips.append(b);
      }
      if (!dish.skipped) {
        const skip = h('button', { class: 'chip skip' }, 'Skip');
        skip.addEventListener('click', () => { dish.skipped = true; renderDish(dish); updateDishesUi(); });
        chips.append(skip);
      }
      body.append(chips);
    }
    if (!dish.skipped) body.append(dishSearch(dish));
    if (!dish.skipped && (dish.label || !dish.guesses.length)) {
      const bars = h('div', { class: 'tastes' });
      for (const [key, lbl] of [['sweet', 'Sweet'], ['bitter', 'Bitter'], ['salty', 'Salty'], ['watery', 'Watery']] as const) {
        const v = h('span', { class: 'v' }, dish.taste[key].toFixed(1));
        const input = h('input', { type: 'range', min: '0', max: '1', step: '0.05', value: String(dish.taste[key]), ariaLabel: `${lbl} 0 to 1` });
        input.addEventListener('input', () => { dish.taste[key] = Number(input.value); v.textContent = dish.taste[key].toFixed(1); });
        bars.append(h('label', { class: 'taste' }, lbl, input, v));
      }
      body.append(bars);
      if (dish.notes) body.append(h('div', { class: 'notes' }, ...dish.notes.split('\n').map((l) => h('p', {}, l))));
    }
  }
  el.append(body);
}

function buildView(dish: Dish): HTMLElement {
  const view = h('div', { class: 'view' });
  if (flyEyeSupported()) {
    const canvas = h('canvas');
    view.append(canvas);
    try {
      dish.eye = new FlyEye(canvas);
      dish.eye.setImage(dish.photo);
      requestAnimationFrame(() => dish.eye?.render(1));
    } catch { canvas.remove(); }
  }
  if (!dish.eye) view.append(h('img', { src: dish.thumb, alt: '' }));
  const rm = h('button', { class: 'remove', ariaLabel: 'Remove dish', title: 'Remove' }, '×');
  rm.addEventListener('click', () => {
    dish.eye?.dispose();
    state.dishes = state.dishes.filter((d) => d !== dish);
    dish.el.remove();
    updateDishesUi();
  });
  view.append(rm);
  return view;
}

function tasteable() { return state.dishes.filter((d) => !d.skipped && d.guesses !== null && (d.label || !d.guesses.length)); }

function updateDishesUi() {
  const n = tasteable().length, waiting = state.dishes.some((d) => d.guesses === null);
  const btn = $<HTMLButtonElement>('taste-btn');
  btn.disabled = n === 0 || waiting;
  btn.textContent = waiting ? 'Recognizing…' : n ? `Let the fly taste ${n} dish${n > 1 ? 'es' : ''}` : 'Let the fly taste';
  const full = state.dishes.length >= MAX_DISHES;
  $('file-camera').parentElement!.toggleAttribute('hidden', full);
  $('file-upload').parentElement!.toggleAttribute('hidden', full);
  let legend = document.querySelector('.legend');
  if (state.dishes.some((d) => d.eye) && !legend) {
    legend = h('p', { class: 'legend' }, `Fly view: ${FLY_EYE_LEGEND}. Display only; the fly decides by taste.`);
    $('dishes').after(legend);
  }
}
window.addEventListener('resize', () => state.dishes.forEach((d) => d.eye?.render(Number(d.el.dataset.mix ?? '1'))));

// ---------- 3. tasting ----------
$('taste-btn').addEventListener('click', () => runTasting());
$('again-btn').addEventListener('click', () => runTasting());

async function runTasting() {
  const service = state.service!, dishes = tasteable();
  if (!dishes.length || state.tasting) return;
  go('tasting');
  state.tasting = true;
  $('tasted').replaceChildren();
  const raster = new Raster($<HTMLCanvasElement>('raster'), state.mn9Index);
  $('raster').parentElement!.hidden = service.kind !== 'live';
  const thumb = $<HTMLCanvasElement>('tasting-thumb');

  try {
    for (const [i, dish] of dishes.entries()) {
      $('tasting-step').textContent = `Tasting ${i + 1} of ${dishes.length} · ${TRIALS} tastings of ${TRIAL_MS} ms`;
      $('tasting-name').textContent = dishName(dish);
      thumb.width = thumb.height = 192;
      const s = Math.min(dish.photo.width, dish.photo.height);
      thumb.getContext('2d')!.drawImage(dish.photo, (dish.photo.width - s) / 2, (dish.photo.height - s) / 2, s, s, 0, 0, 192, 192);
      const nSteps = Math.round(TRIAL_MS / DT_MS);
      raster.reset(nSteps);
      setLive(0, 0, 0, 0, 0);

      const rates = tasteToRates(dish.taste, state.mood);
      const r = await service.taste({ rates, trials: TRIALS, tMs: TRIAL_MS, seed: (Math.random() * 2 ** 32) >>> 0 }, (u) => {
        raster.add(u.raster, u.step);
        const ms = u.step * DT_MS;
        const meanMn9 = u.mn9.reduce((a, b) => a + b, 0) / u.mn9.length;
        setLive(raster.firing, Math.max(...u.active), ms, u.mn9.filter((c) => c >= 4).length, ms ? (meanMn9 * 1000) / ms : 0);
      });
      dish.outcome = isRecorded(r)
        ? { ate: r.ate, trials: r.trials, mn9Hz: r.mn9Hz, fled: r.fled > 0 }
        : {
          ate: r.mn9Steps.filter(ateIt).length, trials: TRIALS,
          mn9Hz: (r.mn9Steps.reduce((a, s) => a + s.length, 0) / TRIALS) * (1000 / TRIAL_MS),
          fled: r.dnp01Steps.some((s) => s.length > 0),
        };
      if (isRecorded(r)) setLive(0, 0, TRIAL_MS, r.ate, r.mn9Hz);
      const o = dish.outcome;
      $('tasted').append(h('li', {}, h('span', {}, dishName(dish)),
        h('span', { class: 'mono' }, o.fled ? 'fled' : `ate it ${o.ate}/${o.trials}`)));
    }
  } catch (err) {
    state.tasting = false;
    alert(`The tasting failed: ${(err as Error).message}`);
    go('dishes');
    return;
  }
  state.tasting = false;
  showVerdict(dishes);
}

function setLive(firing: number, touched: number, ms: number, ateSoFar: number, hz: number) {
  $('stat-firing').textContent = fmt(firing);
  $('stat-touched').textContent = fmt(touched);
  $('stat-time').textContent = ms.toFixed(0);
  $('mn9-fill').style.width = `${(ateSoFar / TRIALS) * 100}%`;
  $('mn9-hz').textContent = `${hz.toFixed(0)} Hz`;
}
document.querySelector<HTMLElement>('.meter .tick')!.style.left = `${(MIN_ATE / TRIALS) * 100}%`;

// ---------- 4. verdict ----------
let lastShare: Parameters<typeof renderShareCard>[0] | null = null;

function showVerdict(dishes: Dish[]) {
  const { ranked, winner } = rank(dishes.map((d) => ({ item: d, outcome: d.outcome! })));
  const hero = $('verdict-hero');
  hero.replaceChildren();
  hero.classList.toggle('has-photo', !!winner);
  const live = state.service!.kind === 'live';
  if (winner) {
    const d = winner.item, o = winner.outcome;
    hero.append(
      h('img', { class: 'photo', src: d.photo.toDataURL('image/jpeg', 0.85), alt: dishName(d) }),
      h('div', {},
        h('p', { class: 'stamp' }, `${live ? 'live' : 'recorded'} brain · ${state.mood} fly`),
        h('p', { class: 'big-line' }, `The fly picks the ${dishName(d).toLowerCase()}.`),
        h('p', { class: 'sub' }, `Ate it in ${o.ate} of ${o.trials} tastings.`)),
    );
  } else {
    hero.append(h('div', {},
      h('p', { class: 'stamp' }, `${live ? 'live' : 'recorded'} brain · ${state.mood} fly`),
      h('p', { class: 'big-line' }, 'The fly declines the whole menu.'),
      h('p', { class: 'sub' }, `No dish reached ${MIN_ATE} of ${TRIALS} tastings. Try something sweeter, or a hungrier fly.`)));
  }
  $('ranking').replaceChildren(...ranked.map(({ item: d, outcome: o, rank: pos }) => h('li', { class: d === winner?.item ? 'win' : '' },
    h('span', { class: 'pos' }, String(pos)),
    h('img', { src: d.thumb, alt: '' }),
    h('div', {}, h('div', { class: 't' }, dishName(d)), h('div', { class: 'why' }, oneLiner(dishName(d), d.taste, o, d === winner?.item,
      winner ? { name: dishName(winner.item), outcome: winner.outcome } : undefined))),
    h('div', { class: 'count' }, o.fled ? 'fled' : `${o.ate}/${o.trials}`, h('small', {}, `MN9 ${o.mn9Hz.toFixed(0)} Hz`)))));
  $('verdict-foot').textContent = `Each tasting is ${TRIAL_MS} ms of simulated fly time. "Ate it" = MN9, the proboscis motor neuron, fired 4 spikes within 50 ms.`;
  lastShare = {
    photo: winner?.item.photo ?? null,
    headline: winner ? `The fly picks the ${dishName(winner.item).toLowerCase()}.` : 'The fly declines the whole menu.',
    sub: winner ? `Ate it in ${winner.outcome.ate} of ${winner.outcome.trials} tastings.` : `No dish reached ${MIN_ATE} of ${TRIALS} tastings.`,
    lines: ranked.map(({ item, outcome }) => ({ name: dishName(item), ate: outcome.ate, trials: outcome.trials })),
    footer: `${live ? 'Live' : 'Recorded'} FlyWire brain · 138,639 neurons · ${state.mood} fly`,
  };
  go('verdict');
}

$('share-btn').addEventListener('click', async () => {
  if (!lastShare) return;
  const blob = await renderShareCard(lastShare);
  await shareOrDownload(blob, lastShare.headline);
});

// ---------- about ----------
$('about-open').addEventListener('click', () => $<HTMLDialogElement>('about').showModal());

// ---------- vision mode ----------
initLook(() => state.service);
