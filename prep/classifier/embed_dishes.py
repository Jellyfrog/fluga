"""Text embeddings of the dish vocabulary for zero-shot matching (build time; the browser ships only the result).

    python prep/classifier/embed_dishes.py <model_dir> [--no-desc] [--no-aliases]

<model_dir> holds preprocess.json from export_clip.py (it names the model). Writes <model_dir>/text_emb.npy
(float32, one L2-normalized row per dish, in load_all() order). Each row averages the prompt ensemble over
the dish name, its aliases and, with its visual description, one descriptive prompt.
"""
import json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from dishes import load_all  # noqa: E402
from zeroshot import load  # noqa: E402

NAME_PROMPTS = ['a photo of {}, a type of food.', 'a photo of {}.', 'a close-up photo of {}, a dish.']


def prompts(d, desc=True, aliases=True):
    names = [d['name'], *(d.get('aliases', []) if aliases else [])]
    out = [p.format(n) for n in names for p in NAME_PROMPTS]
    if desc and d.get('desc'):
        out += [f"a photo of {d['name']}, {d['desc']}."] * len(NAME_PROMPTS)  # weigh it like one name
    return out


@torch.no_grad()
def embed(model, proc, texts):
    t = proc(text=texts, return_tensors='pt', padding=True, truncation=True)
    e = model.get_text_features(**t)
    e = getattr(e, 'pooler_output', e)
    return torch.nn.functional.normalize(e, dim=-1)


def main():
    model_dir = sys.argv[1]
    desc, aliases = '--no-desc' not in sys.argv, '--no-aliases' not in sys.argv
    pre = json.load(open(os.path.join(model_dir, 'preprocess.json')))
    model, proc = load(pre['model'])
    torch.set_num_threads(os.cpu_count())
    per_dish = [prompts(d, desc, aliases) for d in load_all()]
    flat = [t for ps in per_dish for t in ps]
    E = torch.cat([embed(model, proc, flat[i:i + 256]) for i in range(0, len(flat), 256)])
    rows, at = [], 0
    for ps in per_dish:
        rows.append(torch.nn.functional.normalize(E[at:at + len(ps)].mean(0), dim=0).numpy())
        at += len(ps)
    T = np.stack(rows).astype(np.float32)
    np.save(os.path.join(model_dir, 'text_emb.npy'), T)
    print(f'{T.shape[0]} dishes x {T.shape[1]} -> {model_dir}/text_emb.npy (desc={desc}, aliases={aliases})')


if __name__ == '__main__':
    main()
