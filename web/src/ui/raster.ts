// Live spike raster of the busiest neurons in one tasting, with MN9 pinned to the bottom row.

const ROWS = 36;

export class Raster {
  private ctx: CanvasRenderingContext2D;
  private spikes = new Map<number, number[]>(); // neuron -> steps
  private nSteps = 1;
  private now = 0;

  constructor(private canvas: HTMLCanvasElement, private mn9: number) {
    this.ctx = canvas.getContext('2d')!;
  }

  reset(nSteps: number) {
    this.spikes.clear();
    this.nSteps = nSteps;
    this.now = 0;
    this.draw();
  }

  add(raster: { neuron: number; step: number }[], now: number) {
    for (const { neuron, step } of raster) {
      let s = this.spikes.get(neuron);
      if (!s) this.spikes.set(neuron, (s = []));
      s.push(step);
    }
    this.now = now;
    this.draw();
  }

  get firing() { return this.spikes.size; }

  draw() {
    const c = this.canvas, dpr = Math.min(devicePixelRatio || 1, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const ctx = this.ctx, css = getComputedStyle(c);
    const ink = css.getPropertyValue('--spike').trim() || '#000';
    const accent = css.getPropertyValue('--accent').trim() || '#d9480f';
    const muted = css.getPropertyValue('--muted').trim() || '#888';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Busiest neurons first, then ordered by first spike so activity reads as a cascade.
    const others = [...this.spikes.entries()].filter(([n]) => n !== this.mn9)
      .sort((a, b) => b[1].length - a[1].length).slice(0, ROWS - 1)
      .sort((a, b) => a[1][0] - b[1][0]);
    const rowH = (h - 6) / ROWS;
    const x = (step: number) => (step / this.nSteps) * w;

    ctx.fillStyle = ink;
    others.forEach(([, steps], r) => {
      const y = 2 + r * rowH;
      for (const s of steps) ctx.fillRect(x(s), y, 1.5, Math.max(1, rowH - 1.5));
    });
    const y = h - rowH - 2;
    ctx.fillStyle = accent;
    ctx.fillRect(0, y - 2, w, 1);
    for (const s of this.spikes.get(this.mn9) ?? []) ctx.fillRect(x(s) - 0.5, y, 2.5, rowH);

    // Now-line
    ctx.fillStyle = muted;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(x(this.now), 0, 1, h);
    ctx.globalAlpha = 1;
  }
}
