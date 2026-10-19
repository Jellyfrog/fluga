"""Sample Food-101 validation images via HF datasets-server; writes <dir>/<idx>__<labelname>.jpg."""
import json, sys, os, random, urllib.request, concurrent.futures as cf
out = sys.argv[1]; random.seed(0)
import time
def get(u):
    for k in range(8):
        try: return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'FlugaDev/0.1'}), timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(5 * (k + 1))
    raise RuntimeError(u)
jobs = []
for off in random.sample(range(0, 25250 - 8, 8), 100):
    d = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=ethz/food101&config=default&split=validation&offset={off}&length=8"))
    time.sleep(1)
    names = d['features'][1]['type']['names']
    for r in d['rows']:
        jobs.append((r['row_idx'], names[r['row']['label']], r['row']['image']['src']))
def dl(j):
    i, n, u = j
    open(os.path.join(out, f"{i:05d}__{n}.jpg"), 'wb').write(get(u))
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(dl, jobs))
print(len(jobs), len({n for _, n, _ in jobs}))
