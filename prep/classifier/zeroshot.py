"""Zero-shot dish recognition with CLIP-family models (transformers, CPU).

    python prep/classifier/zeroshot.py <imgdir> <model> [model ...]

Images are named <idx>__<food101_label>.jpg (see fetchval.py). Prints 101-way top-1/top-5 per model;
image embeddings are cached in <imgdir>/../emb/<model>.npy for later vocabulary experiments.
"""
import glob, os, sys, time
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

PROMPTS = ['a photo of {}, a type of food.', 'a photo of {}.', 'a close-up photo of {}, a dish.']


def load(name):
    model = AutoModel.from_pretrained(name).eval()
    return model, AutoProcessor.from_pretrained(name)


@torch.no_grad()
def embed_text(model, proc, names, prompts=PROMPTS):
    out = []
    for n in names:
        texts = [p.format(n) for p in prompts]
        kw = {'padding': 'max_length', 'max_length': 64} if 'siglip' in model.config.model_type else {'padding': True}
        t = proc(text=texts, return_tensors='pt', truncation=True, **kw)
        e = model.get_text_features(**t)
        e = getattr(e, 'pooler_output', e)
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=0))
    return torch.stack(out).numpy()


@torch.no_grad()
def embed_images(model, proc, files, bs=32):
    out = []
    for i in range(0, len(files), bs):
        ims = [Image.open(f).convert('RGB') for f in files[i:i + bs]]
        e = model.get_image_features(**proc(images=ims, return_tensors='pt'))
        e = getattr(e, 'pooler_output', e)
        out.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(out).numpy()


def main():
    imgdir, models = sys.argv[1], sys.argv[2:]
    files = sorted(glob.glob(os.path.join(imgdir, '*.jpg')))
    gt = [os.path.basename(f).split('__')[1][:-4] for f in files]
    labels = sorted(set(gt))
    names = [l.replace('_', ' ') for l in labels]
    embdir = os.path.join(os.path.dirname(os.path.abspath(imgdir)), 'emb')
    os.makedirs(embdir, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    for m in models:
        t0 = time.time()
        model, proc = load(m)
        T = embed_text(model, proc, names)
        cache = os.path.join(embdir, m.replace('/', '__') + '.npy')
        I = np.load(cache) if os.path.exists(cache) else embed_images(model, proc, files)
        np.save(cache, I)
        S = I @ T.T
        rank = S.argsort(1)[:, ::-1]
        idx = np.array([labels.index(g) for g in gt])
        top1 = np.mean(rank[:, 0] == idx)
        top5 = np.mean([i in r[:5] for i, r in zip(idx, rank)])
        nparams = sum(p.numel() for p in model.vision_model.parameters()) / 1e6
        print(f'{m}: vision {nparams:.1f}M params  top1={top1:.3f} top5={top5:.3f}  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
