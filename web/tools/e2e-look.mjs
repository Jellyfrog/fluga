// End-to-end run of the vision mode ("Show it a picture") in headless Chromium with WebGPU.
// Usage: node tools/e2e-look.mjs <outDir> <image> [approach|pass]   (dev server on PORT, default 5173; or URL=https://...)
import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';

const [outDir, img, motion = 'approach'] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const port = process.env.PORT ?? 5173;
const args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan',
  '--disable-vulkan-surface', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({ args, headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3 });
page.on('console', (m) => { if (!/experimental/.test(m.text())) console.log('[page]', m.text()); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const t0 = Date.now();
await page.goto(process.env.URL ?? `http://localhost:${port}/`);
await page.click('#wake-btn');
await page.waitForSelector('#wake-ready:not([hidden])', { timeout: 300_000 });
const badge = (await page.textContent('#brain-badge')).trim();
if (badge !== 'live brain') { console.error(`FAIL: badge "${badge}"`); process.exit(1); }
await page.click('#look-go');
await page.setInputFiles('#look-file', img);
await page.click(`input[value=${motion}] + span`);
await page.waitForFunction(() => !document.querySelector('#look-btn').disabled, null, { timeout: 60_000 });
await page.screenshot({ path: `${outDir}/1-look.png`, fullPage: true });
await page.click('#look-btn');
await page.waitForSelector('#look-result:not([hidden])', { timeout: 1_500_000 });
console.log('result after', Date.now() - t0, 'ms:', await page.textContent('#look-result .big-line'));
console.log(await page.$$eval('#look-result .lc-list li', (els) => els.map((e) => e.textContent)));
console.log(await page.$$eval('.dn-table tr.on', (els) => els.map((e) => e.innerText.replace(/\t/g, ' | '))));
console.log('status:', await page.textContent('#look-status'));
await page.screenshot({ path: `${outDir}/2-result.png`, fullPage: true });
await browser.close();
