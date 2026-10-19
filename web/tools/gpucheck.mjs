import { chromium } from 'playwright';
const variants = {
  vulkan: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--use-angle=vulkan', '--disable-vulkan-surface', '--ignore-gpu-blocklist'],
  swiftshader: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'],
  default: ['--enable-unsafe-webgpu'],
};
for (const [name, args] of Object.entries(variants)) {
  const browser = await chromium.launch({ args, headless: true, channel: process.env.CH || undefined });
  const page = await browser.newPage();
  await page.goto('http://localhost:8765/');
  const r = await page.evaluate(async () => {
    if (!navigator.gpu) return 'no navigator.gpu';
    const a = await navigator.gpu.requestAdapter();
    if (!a) return 'no adapter';
    const info = a.info || {};
    const d = await a.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize, maxBufferSize: a.limits.maxBufferSize } });
    const buf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const m = d.createShaderModule({ code: '@group(0) @binding(0) var<storage, read_write> b: array<atomic<i32>>; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { atomicAdd(&b[0], 1); }' });
    const p = d.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    const bg = d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
    const rb = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = d.createCommandEncoder(); const c = e.beginComputePass(); c.setPipeline(p); c.setBindGroup(0, bg); c.dispatchWorkgroups(10); c.end();
    e.copyBufferToBuffer(buf, 0, rb, 0, 16); d.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    return JSON.stringify({ vendor: info.vendor, arch: info.architecture, desc: info.description, sum: new Int32Array(rb.getMappedRange())[0], maxSB: a.limits.maxStorageBufferBindingSize, maxBuf: a.limits.maxBufferSize, maxSBPerStage: a.limits.maxStorageBuffersPerShaderStage });
  }).catch(e => 'error ' + e.message);
  console.log(name, r);
  await browser.close();
}
