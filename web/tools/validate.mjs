// Runs validate.html in headless Chromium with WebGPU and writes the results to JSON.
// Usage: node tools/validate.mjs [out.json] [query]   (expects a dev or preview server on PORT, default 5173)
import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';

const out = process.argv[2] ?? 'validation.json';
const query = process.argv[3] ?? 'trials=30&mode=both';
const port = process.env.PORT ?? 5173;
const args = (process.env.GPU_ARGS ?? '--enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native --use-angle=vulkan --disable-vulkan-surface --ignore-gpu-blocklist').split(' ');
const browser = await chromium.launch({ args, headless: true });
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${port}/validate.html?${query}`);
await page.waitForFunction(() => window.__results?.done, null, { timeout: 0, polling: 2000 });
const res = await page.evaluate(() => window.__results);
writeFileSync(out, JSON.stringify(res, null, 1));
await browser.close();
if (res.error) { console.error(res.error); process.exit(1); }
