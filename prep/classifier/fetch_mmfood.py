"""Random metadata sample of MM-Food-100K (phone photos with dish names) via HF datasets-server.

    python prep/classifier/fetch_mmfood.py <out.jsonl> [pages=60]
"""
import json, random, sys, time, urllib.request
out, pages = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
N = 100_000
random.seed(1)
with open(out, 'w') as fh:
    for off in random.sample(range(0, N - 100, 100), pages):
        u = f'https://datasets-server.huggingface.co/rows?dataset=Humanbased-AI/MM-Food-100K&config=default&split=train&offset={off}&length=100'
        for k in range(6):
            try:
                rows = json.loads(urllib.request.urlopen(u, timeout=60).read())['rows']
                break
            except Exception:
                time.sleep(5 * (k + 1))
        else:
            continue
        for r in rows:
            fh.write(json.dumps({'row': r['row_idx'], **{k: r['row'][k] for k in ('image_url', 'dish_name', 'food_type', 'food_prob')}}) + '\n')
        time.sleep(0.5)
