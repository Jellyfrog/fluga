// Loads the connectome binaries written by prep/build_brain.py, with progress and Cache Storage.

export interface Group { indices: number[]; flywire: string[]; missing: string[] }
export type Channel = 'sugar' | 'bitter' | 'salt' | 'water';
export const CHANNELS: Channel[] = ['sugar', 'bitter', 'salt', 'water'];

export interface Manifest {
  version: number; neurons: number; edges: number; hash: string;
  files: Record<string, { raw: number; gzip: number }>;
}

export interface Brain {
  manifest: Manifest;
  rowPtr: Uint32Array; // N+1
  post: Uint32Array; // E
  weight: Int16Array; // E, signed synapse counts
  groups: Record<Channel | 'mn9' | 'dnp01', Group>;
}

/** Absolute URL of the brain data. Resolve it on the page and pass it to workers: with a relative
 * Vite base ('./'), a worker would otherwise resolve it against its own script under assets/. */
export const brainBaseUrl = () => new URL(`${import.meta.env.BASE_URL}brain/`, location.href).href;
const FILES = ['rowptr.bin', 'post.bin', 'weight.bin'] as const;

async function getJson<T>(base: string, name: string): Promise<T> {
  const r = await fetch(base + name, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}

/** Fetches a file from Cache Storage (versioned by the data hash) or the network. */
async function cachedFetch(url: string, cacheName: string): Promise<Response> {
  let cache: Cache | undefined;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) return hit;
  } catch { /* no Cache Storage (private mode, insecure origin): plain fetch */ }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (cache) {
    try { await cache.put(url, r.clone()); } catch { /* quota: fine, just not cached */ }
  }
  return r;
}

async function readAll(r: Response, onBytes: (n: number) => void): Promise<Uint8Array> {
  const reader = r.body!.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
    onBytes(value.length);
  }
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** gzip files are decompressed here unless the server already applied Content-Encoding. */
async function gunzipIfNeeded(bytes: Uint8Array): Promise<Uint8Array> {
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** onProgress receives the loaded fraction of the compressed download (0..1). */
export async function loadBrain(onProgress?: (frac: number) => void, base = brainBaseUrl()): Promise<Brain> {
  const manifest = await getJson<Manifest>(base, 'manifest.json');
  const { groups } = await getJson<{ groups: Brain['groups'] }>(base, 'groups.json');
  const cacheName = `flypick-brain-${manifest.hash}`;
  const totalBytes = FILES.reduce((s, f) => s + manifest.files[f].gzip, 0);
  let loaded = 0;
  const bump = (n: number) => { loaded += n; onProgress?.(Math.min(1, loaded / totalBytes)); };

  const blobs = await Promise.all(FILES.map(async (f) => {
    const r = await cachedFetch(base + f + '.gz', cacheName);
    return gunzipIfNeeded(await readAll(r, bump));
  }));
  onProgress?.(1);
  // Drop caches of older brain versions.
  caches?.keys?.().then((ks) => ks.filter((k) => k.startsWith('flypick-brain-') && k !== cacheName)
    .forEach((k) => caches.delete(k))).catch(() => {});

  const [rp, po, we] = blobs;
  const brain: Brain = {
    manifest,
    rowPtr: new Uint32Array(rp.buffer, rp.byteOffset, rp.byteLength / 4),
    post: new Uint32Array(po.buffer, po.byteOffset, po.byteLength / 4),
    weight: new Int16Array(we.buffer, we.byteOffset, we.byteLength / 2),
    groups,
  };
  if (brain.rowPtr.length !== manifest.neurons + 1 || brain.post.length !== manifest.edges
      || brain.weight.length !== manifest.edges) throw new Error('brain files do not match manifest');
  return brain;
}
