// Renders the verdict to a shareable PNG card.

export interface ShareInput {
  photo: HTMLCanvasElement | null; // winner photo, or null when the fly declined
  headline: string;
  sub: string;
  lines: { name: string; ate: number; trials: number }[];
  footer: string;
}

export async function renderShareCard(s: ShareInput): Promise<Blob> {
  const W = 1080, H = 1350;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const g = c.getContext('2d')!;
  g.fillStyle = '#f6f3ec';
  g.fillRect(0, 0, W, H);

  const photoH = 640;
  if (s.photo) {
    const p = s.photo, scale = Math.max(W / p.width, photoH / p.height);
    const sw = W / scale, sh = photoH / scale;
    g.drawImage(p, (p.width - sw) / 2, (p.height - sh) / 2, sw, sh, 0, 0, W, photoH);
  } else {
    g.fillStyle = '#e2dccd';
    g.fillRect(0, 0, W, photoH);
    g.fillStyle = '#6b665a';
    g.font = '700 120px system-ui, sans-serif';
    g.textAlign = 'center';
    g.fillText('🪰  ✗', W / 2, photoH / 2 + 40);
    g.textAlign = 'left';
  }

  const pad = 72;
  let y = photoH + 96;
  g.fillStyle = '#d9480f';
  g.font = '600 30px ui-monospace, Menlo, monospace';
  g.fillText('FLUGA · A REAL FLY CONNECTOME DECIDED', pad, y);
  y += 84;
  g.fillStyle = '#1d1b16';
  g.font = '800 76px system-ui, sans-serif';
  y = wrap(g, s.headline, pad, y, W - 2 * pad, 84);
  g.fillStyle = '#6b665a';
  g.font = '400 38px system-ui, sans-serif';
  y = wrap(g, s.sub, pad, y + 12, W - 2 * pad, 48) + 24;

  g.font = '500 32px system-ui, sans-serif';
  for (const l of s.lines.slice(0, 5)) {
    g.fillStyle = '#1d1b16';
    g.fillText(l.name, pad, y);
    g.fillStyle = '#6b665a';
    g.textAlign = 'right';
    g.fillText(`${l.ate}/${l.trials}`, W - pad, y);
    g.textAlign = 'left';
    y += 46;
  }
  g.fillStyle = '#6b665a';
  g.font = '400 26px system-ui, sans-serif';
  g.fillText(s.footer, pad, H - 56);
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png'));
}

function wrap(g: CanvasRenderingContext2D, text: string, x: number, y: number, maxW: number, lh: number): number {
  let line = '';
  for (const word of text.split(' ')) {
    const t = line ? `${line} ${word}` : word;
    if (g.measureText(t).width > maxW && line) {
      g.fillText(line, x, y);
      line = word;
      y += lh;
    } else line = t;
  }
  if (line) g.fillText(line, x, y);
  return y + lh;
}

/** Native share sheet with the image where supported, otherwise a download. */
export async function shareOrDownload(blob: Blob, text: string) {
  const file = new File([blob], 'fluga.png', { type: 'image/png' });
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], text });
      return;
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
    }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'fluga.png';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
