"""Pick MM-Food-100K rows whose dish_name matches a vocabulary dish (name or alias) and download them.

    python prep/classifier/select_mmfood.py <meta.jsonl> <out_dir> [max=400] [per_dish=3]

Writes <out_dir>/<row>__<dish_id>.jpg. Food-101 dishes are kept too; the eval reports both groups.
"""
import json, os, random, re, sys, urllib.request, concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dishes import load_all

meta, out = sys.argv[1], sys.argv[2]
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 400
per = int(sys.argv[4]) if len(sys.argv) > 4 else 3
os.makedirs(out, exist_ok=True)
norm = lambda s: re.sub(r'(es|s)$', '', re.sub(r'[^a-z ]', '', s.lower()).strip())
index = {}
for d in load_all():
    for n in [d['name'], *d.get('aliases', [])]:
        index.setdefault(norm(n), d['id'])
rows = [json.loads(l) for l in open(meta)]
random.seed(2); random.shuffle(rows)
picked, count = [], {}
for r in rows:
    if r['food_prob'] < 0.9: continue
    i = index.get(norm(r['dish_name']))
    if i and count.get(i, 0) < per:
        count[i] = count.get(i, 0) + 1
        picked.append((r['row'], i, r['image_url']))
    if len(picked) >= cap: break
def dl(p):
    row, i, u = p
    try:
        data = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'FlugaDev/0.1'}), timeout=60).read()
        open(os.path.join(out, f'{row}__{i}.jpg'), 'wb').write(data)
        return 1
    except Exception:
        return 0
with cf.ThreadPoolExecutor(8) as ex: ok = sum(ex.map(dl, picked))
print(f'{ok}/{len(picked)} images, {len(count)} dishes')
