"""Ship the dish recognizer to the web app: web/public/models/dishes/.

    python prep/classifier/build_web.py <model_dir> <vision.onnx>

Writes
  vision.onnx       the image tower (copied)
  preprocess.json   resize/crop/normalization for the browser
  dishes.json       one entry per dish: id, name, aliases, taste, USDA nutrients per 100 g, sources
  text_emb.bin      int8 text embeddings, row-major (dishes x dim), with per-row float32 scales
                    appended (dequantized row = int8 * scale; cosine error < 0.001)
"""
import json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from dishes import load_all, with_taste  # noqa: E402
from usda_tastes import load as load_fndds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '../..'))
OUT = os.path.join(ROOT, 'web/public/models/dishes')


def main():
    model_dir, vision = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    foods = load_fndds()
    dishes = with_taste(load_all(), foods)
    T = np.load(os.path.join(model_dir, 'text_emb.npy')).astype(np.float32)
    assert T.shape[0] == len(dishes), 'text_emb.npy is stale: re-run embed_dishes.py'

    scale = np.abs(T).max(1) / 127
    q = np.round(T / scale[:, None]).astype(np.int8)
    worst = ((q * scale[:, None]) * T).sum(1).min()
    with open(os.path.join(OUT, 'text_emb.bin'), 'wb') as fh:
        fh.write(q.tobytes())
        fh.write(scale.astype(np.float32).tobytes())

    rnd = lambda x, n=2: round(float(x), n)
    out = [{
        'id': d['id'], 'name': d['name'], 'aliases': d.get('aliases', []),
        'taste': {k: rnd(v) for k, v in d['taste'].items()},
        'per100g': {'sugar_g': rnd(d['nutrients']['sugar'], 1), 'sodium_mg': round(d['nutrients']['sodium']),
                    'water_g': rnd(d['nutrients']['water'], 1), 'caffeine_mg': round(d['nutrients']['caffeine']),
                    'theobromine_mg': round(d['nutrients']['theobromine']),
                    'bitter_share': rnd(d['nutrients']['bitter_share'])},
        'usda': [[foods[f]['description'], g] for f, g in d['fndds']],
    } for d in dishes]
    pre = json.load(open(os.path.join(model_dir, 'preprocess.json')))
    json.dump({'dim': int(T.shape[1]), 'dishes': out}, open(os.path.join(OUT, 'dishes.json'), 'w'), separators=(',', ':'), ensure_ascii=False)
    json.dump(pre, open(os.path.join(OUT, 'preprocess.json'), 'w'), indent=1)
    shutil.copyfile(vision, os.path.join(OUT, 'vision.onnx'))
    for f in sorted(os.listdir(OUT)):
        print(f'{f:16s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:6.2f} MB')
    print(f'{len(out)} dishes; worst int8 cosine {worst:.5f}')


if __name__ == '__main__':
    main()
