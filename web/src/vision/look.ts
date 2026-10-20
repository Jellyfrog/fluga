// The "What would a fly do?" screen: choose a picture and a motion, run the brain, show where its visual
// projection neurons fired and which command neurons it reached.

import type { BrainService } from '../sim/service';
import { isRecorded } from '../sim/service';
import {
  PHOTO_AZ, T_MS, TRIALS, encode, loadVision, placement, readOut, visionMap,
  type Motion, type VisionData, type VisionOutcome,
} from './vision';

const AZ_VIEW = 100, EL_VIEW = 70; // canvas spans +-100 deg azimuth, +-70 deg elevation
const FAMILY_COLOR = { looming: '#e8590c', object: '#1c7ed6', other: '#868e96' } as const;
const FAMILY_LABEL = { looming: 'looming detectors (LC4, LPLC2)', object: 'small-object detectors (LC11, LPLC1)', other: 'other (LC12, LC17, LC18)' } as const;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export function initLook(getService: () => BrainService | null) {
  let photo: HTMLCanvasElement | null = null;
  let vision: { v: VisionData; map: ReturnType<typeof visionMap> } | null = null;
  let mapSentTo: BrainService | null = null;
  let busy = false;
  const motion = () => (document.querySelector<HTMLInputElement>('input[name=motion]:checked')?.value ?? 'approach') as Motion;
  const status = (s: string) => { $('look-status').textContent = s; };

  let shown: VisionOutcome | undefined; // the result on screen, kept across redraws (resize, rotation)
  const redraw = () => drawField($('look-field') as HTMLCanvasElement, photo, motion(), vision?.v, shown);
  window.addEventListener('resize', () => redraw());
  document.querySelectorAll<HTMLInputElement>('input[name=motion]').forEach((r) => r.addEventListener('change', () => {
    shown = undefined; // the dots belong to the other motion
    $('look-result').hidden = true;
    redraw();
  }));
  redraw();

  $<HTMLInputElement>('look-file').addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    (e.target as HTMLInputElement).value = '';
    if (!file) return;
    try {
      photo = await loadPhoto(file);
    } catch {
      status('Could not read that picture.');
      return;
    }
    $('look-result').hidden = true;
    shown = undefined;
    redraw();
    const svc = getService();
    const live = svc && svc.kind === 'live';
    $<HTMLButtonElement>('look-btn').disabled = !live;
    status(live ? 'Ready. The picture will move for 250 ms of fly time.'
      : 'This needs the live brain (WebGPU); the recorded brain only knows taste.');
  });

  $('look-btn').addEventListener('click', async () => {
    const svc = getService();
    if (!svc || !photo || busy || svc.kind !== 'live' || !svc.setVision) return;
    busy = true;
    $<HTMLButtonElement>('look-btn').disabled = true;
    try {
      status('Loading the eye map…');
      if (!vision) { const v = await loadVision(); vision = { v, map: visionMap(v) }; }
      if (mapSentTo !== svc) { svc.setVision(vision.map); mapSentTo = svc; }
      const drive = encode(vision.v, photo, motion());
      const r = await svc.taste({ rates: { sugar: 0, bitter: 0, salt: 0, water: 0 }, trials: TRIALS, tMs: T_MS,
        seed: (Math.random() * 2 ** 31) | 0, drive },
      (u) => status(`Showing it… ${Math.round((u.step / u.nSteps) * T_MS)} of ${T_MS} ms, ${TRIALS} runs side by side`));
      if (isRecorded(r)) throw new Error('recorded brain');
      const outcome = readOut(vision.v, vision.map, r.counts, r.dnp01Steps, drive);
      shown = outcome;
      redraw();
      status(`Dots: visual projection neurons that fired, at the centre of what each one sees (${TRIALS} runs).`);
      renderResult(outcome, vision.v);
    } catch (err) {
      status(`The fly could not look: ${(err as Error).message}`);
    } finally {
      busy = false;
      $<HTMLButtonElement>('look-btn').disabled = false;
    }
  });
}

async function loadPhoto(file: File): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const s = Math.min(1, 512 / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * s); c.height = Math.round(bmp.height * s);
  const g = c.getContext('2d')!;
  g.imageSmoothingQuality = 'high';
  g.drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c;
}

function drawField(canvas: HTMLCanvasElement, photo: HTMLCanvasElement | null, motion: Motion, v?: VisionData, out?: VisionOutcome) {
  const w = canvas.clientWidth || 600, h = Math.round(w * (EL_VIEW / AZ_VIEW));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  canvas.style.height = `${h}px`;
  const g = canvas.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const X = (az: number) => ((az + AZ_VIEW) / (2 * AZ_VIEW)) * w, Y = (el: number) => ((EL_VIEW - el) / (2 * EL_VIEW)) * h;
  g.fillStyle = '#7f7f7f'; // the grey world around the picture
  g.fillRect(0, 0, w, h);
  if (photo) {
    // Where the picture is at the end of its motion (approach: closest; pass: about to leave).
    const { az, scale } = placement(motion, motion === 'approach' ? 30 : 15);
    const aw = PHOTO_AZ * scale, ah = aw * (photo.height / photo.width);
    g.drawImage(photo, X(az - aw), Y(ah), X(az + aw) - X(az - aw), Y(-ah) - Y(ah));
  }
  if (v) {
    g.fillStyle = 'rgb(255 255 255 / 18%)';
    for (let c = 0; c < v.columns.az.length; c++) g.fillRect(X(v.columns.az[c]) - 0.75, Y(v.columns.el[c]) - 0.75, 1.5, 1.5);
  }
  if (v && out) {
    const r = (4 / (2 * AZ_VIEW)) * w;
    for (let k = 0; k < out.lcHz.length; k++) {
      const hz = out.lcHz[k];
      if (hz <= 0) continue;
      g.globalAlpha = Math.min(1, 0.25 + hz / 20);
      g.fillStyle = FAMILY_COLOR[v.lc.family[k]];
      g.beginPath(); g.arc(X(v.lc.az[k]), Y(v.lc.el[k]), r, 0, 2 * Math.PI); g.fill();
    }
    g.globalAlpha = 1;
  }
  g.fillStyle = 'rgb(255 255 255 / 85%)';
  g.font = '12px system-ui, sans-serif';
  g.fillText("fly's left", 8, h - 8);
  g.textAlign = 'right'; g.fillText("fly's right", w - 8, h - 8); g.textAlign = 'left';
  g.strokeStyle = 'rgb(255 255 255 / 35%)'; g.setLineDash([4, 4]);
  g.beginPath(); g.moveTo(X(0), 0); g.lineTo(X(0), h); g.stroke(); g.setLineDash([]);
}

function renderResult(o: VisionOutcome, v: VisionData) {
  const el = $('look-result');
  el.hidden = false;
  const fam = (f: 'looming' | 'object' | 'other') => {
    let n = 0, sumHz = 0, left = 0, right = 0;
    o.lcHz.forEach((hz, k) => {
      if (v.lc.family[k] !== f || hz <= 0) return;
      n++; sumHz += hz;
      if (v.lc.az[k] < 0) left += hz; else right += hz;
    });
    return { n, sumHz, left, right };
  };
  const fams = (['looming', 'object', 'other'] as const).map((f) => ({ f, ...fam(f) }));
  const all = fams.reduce((s, x) => ({ left: s.left + x.left, right: s.right + x.right }), { left: 0, right: 0 });
  const side = all.left + all.right === 0 ? null : all.right > 2 * all.left ? 'right' : all.left > 2 * all.right ? 'left' : 'front';
  const esc = o.escapeTrials;
  const headline = esc > 0
    ? `Its escape neuron fired in ${esc} of ${o.trials} runs.`
    : side ? 'It noticed, but sent no escape command.' : 'Nothing got through to its visual feature detectors.';
  const sub = esc > 0
    ? 'In a real fly, spikes in the giant fiber (DNp01) trigger a takeoff jump. In this model it fires for most bright, changing patches, so read it as "startled", not as a judgement of danger.'
    : side ? `Its visual projection neurons fired, mostly ${side === 'front' ? 'on both sides' : `on its ${side}`}, but the giant fiber stayed silent.`
      : 'Try a picture with more contrast, or let it approach.';

  const fmt = (hz: number) => (Number.isNaN(hz) ? '–' : hz < 0.05 ? '0' : hz.toFixed(1));
  el.replaceChildren(
    h('p', { class: 'eyebrow' }, 'Live brain · experimental'),
    h('p', { class: 'big-line' }, headline),
    h('p', { class: 'fine' }, sub),
    h('h3', {}, 'Visual feature detectors'),
    h('ul', { class: 'lc-list' }, ...fams.map((x) => h('li', {},
      h('span', { class: 'dot', style: `background:${FAMILY_COLOR[x.f]}` }), `${FAMILY_LABEL[x.f]}: `,
      x.n ? `${x.n} fired, left ${x.left.toFixed(0)} Hz · right ${x.right.toFixed(0)} Hz (summed)` : 'silent'))),
    h('h3', {}, 'Command neurons'),
    h('table', { class: 'dn-table' },
      h('tr', {}, h('th', {}, 'Neuron'), h('th', {}, 'Known for'), h('th', {}, 'Left Hz'), h('th', {}, 'Right Hz')),
      ...o.dn.map((d) => h('tr', { class: d.leftHz > 0 || d.rightHz > 0 ? 'on' : '' },
        h('td', {}, d.name), h('td', {}, d.role), h('td', {}, fmt(d.leftHz)), h('td', {}, fmt(d.rightHz))))),
    h('p', { class: 'fine' }, `"Known for" is from experiments on real flies; none of these read-outs were validated in this model, `
      + 'which was tested on taste and grooming. Turning direction cannot be read from it.'),
    h('details', {}, h('summary', {}, 'What is real here'),
      h('p', { class: 'fine' }, `Real: the wiring from ${(v.drive.on.length + v.drive.off.length + v.drive.both.length).toLocaleString('en-US')} `
        + `medulla and lobula-plate neurons in ${o.drivenColumns} driven eye columns to the visual projection neurons and descending neurons, `
        + 'and the spiking model that runs it.'),
      h('p', { class: 'fine' }, 'Not modelled: photoreceptors and the lamina (their wiring is incomplete in the data), motion and change '
        + 'detection (done here, per 10 ms, outside the connectome), adaptation, colour, exact viewing angles (approximate), the body '
        + 'and ventral nerve cord, and internal state. A real fly would also be judging distance, speed and threat.'),
      h('p', { class: 'fine' }, `Dots mark each visual projection neuron that fired, at the centre of the eye columns that feed it: `
        + 'activity stays near where the picture changed, as the wiring predicts. It is not where the fly "looks".')),
  );
}

function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, string> = {}, ...kids: (Node | string)[]) {
  const el = document.createElement(tag);
  for (const [k, val] of Object.entries(props)) el.setAttribute(k, val);
  el.append(...kids);
  return el;
}

