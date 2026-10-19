// End-to-end run of the app: wake, add dishes, taste, verdict. Screenshots into OUT.
// Usage: node tools/e2e.mjs <outDir> [image ...]   (dev server on PORT, default 5173; or URL=https://...)
import { chromium, devices } from 'playwright';
import { mkdirSync } from 'node:fs';

const out = process.argv[2] ?? 'tools/out/e2e';
const imgs = process.argv.slice(3).length ? process.argv.slice(3)
  : ['tools/testimages/donuts.jpg', 'tools/testimages/caesar_salad.jpg', 'tools/testimages/tiramisu.jpg'];
mkdirSync(out, { recursive: true });
const port = process.env.PORT ?? 5173;
const args = process.env.NO_GPU ? [] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan',
  '--disable-vulkan-surface', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({ args, headless: true });
const ctx = await browser.newContext({ ...devices['iPhone 13'], colorScheme: process.env.SCHEME ?? 'light' });
const page = await ctx.newPage();
page.on('console', (m) => { if (!m.text().includes('[vite]')) console.log('[page]', m.text()); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const t0 = Date.now();
const shot = (name) => page.screenshot({ path: `${out}/${name}.png`, fullPage: true });

await page.goto(process.env.URL ?? `http://localhost:${port}/`);
await shot('1-wake');
await page.click('#wake-btn');
await page.waitForSelector('#wake-ready:not([hidden])', { timeout: 300_000 });
const badge = (await page.textContent('#brain-badge')).trim();
console.log('awake after', Date.now() - t0, 'ms; badge:', badge);
const want = process.env.NO_GPU ? 'recorded brain' : 'live brain';
if (badge !== want) { console.error(`FAIL: badge "${badge}", expected "${want}"`); process.exit(1); }
await shot('2-ready');
await page.click('#wake-ready [data-go=dishes]');
await page.click('input[value=starving] + span');
await page.setInputFiles('#file-upload', imgs);
await page.waitForFunction(() => !document.querySelector('#taste-btn').disabled, null, { timeout: 300_000 });
await page.waitForTimeout(500);
console.log('dishes:', await page.$$eval('.dish .name', (els) => els.map((e) => e.textContent)));
await shot('3-dishes');
await page.click('#taste-btn');
await page.waitForFunction(() => Number(document.querySelector('#stat-time').textContent) > 120, null, { timeout: 300_000 });
await shot('4-tasting');
await page.waitForSelector('#screen-verdict:not([hidden])', { timeout: 900_000 });
console.log('verdict after', Date.now() - t0, 'ms:', await page.textContent('#verdict-hero .big-line'));
console.log(await page.$$eval('#ranking li', (els) => els.map((e) => e.innerText.replace(/\n/g, ' | '))));
await shot('5-verdict');
await page.click('#about-open');
await shot('6-about');
await browser.close();
