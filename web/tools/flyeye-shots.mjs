// Screenshot the FlyEye demo at mix = 0, 0.5, 1.
// Usage: source tools/env.sh && node tools/flyeye-shots.mjs <outDir> [imgPath] [port]
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
const out = process.argv[2] || '.';
const img = process.argv[3] || 'tools/testimages/pizza.jpg';
const port = process.argv[4] || '5181';
const name = img.split('/').pop().replace(/\.\w+$/, '');
mkdirSync(out, { recursive: true });
const args = (process.env.FLYEYE_ARGS ||
  '--use-angle=swiftshader --enable-unsafe-swiftshader').split(' ');
const browser = await chromium.launch({ args, headless: true });
const page = await browser.newPage({ viewport: { width: 760, height: 640 }, deviceScaleFactor: Number(process.env.DPR || 1) });
page.on('console', (m) => { if (!/GPU stall|vite/.test(m.text())) console.log('[page]', m.text()); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
for (const mix of ['0', '0.5', '1']) {
  await page.goto(`http://localhost:${port}/flyeye-demo.html?img=/${img}&mix=${mix}`);
  await page.waitForSelector('body[data-ready="1"]', { timeout: 15000 });
  if (process.env.ASPECT) await page.addStyleTag({ content: `#view{aspect-ratio:${process.env.ASPECT} !important;width:400px !important}` });
  await page.waitForTimeout(200);
  const file = `${out}/${name}${process.env.ASPECT ? '-tall' : ''}-mix${mix}.png`;
  await page.locator('#view').screenshot({ path: file });
  console.log('wrote', file);
}
await browser.close();
