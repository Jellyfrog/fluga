"""Open-vocabulary dish recognition, judged on what reaches the fly: the taste vector.

    python prep/classifier/eval_vocab.py <model_dir> <vision.onnx> <set>:<imgdir> [<set>:<imgdir> ...]

Image files are <anything>__<dish id>.jpg (a Food-101 label is a dish id). For each set prints exact
top-1/top-5 over the whole vocabulary and the mean absolute taste error (4 channels, 0..1) of
  top1   the top dish's taste,
  blend  the similarity-weighted taste of the top 5,
and the old MobileNetV2 Food-101 classifier on the same photos (its label's USDA taste), as baseline.
Also prints how often the top dish is right at each confidence threshold, to set LOW_CONFIDENCE.
"""
import glob, json, os, sys
import numpy as np, onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from dishes import load_all, with_taste  # noqa: E402
from usda_tastes import load as load_fndds  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, '../..'))
CH = ('sweet', 'bitter', 'salty', 'watery')
SCALE = 100.0  # CLIP logit scale
TOPK = 5


def image_prep(pre):
    mean, std = np.array(pre['mean'], np.float32), np.array(pre['std'], np.float32)

    def prep(p):
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = pre['size'] / min(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        w, h = im.size
        c = pre['crop'] or pre['size']
        l, t = (w - c) // 2, (h - c) // 2
        return ((np.asarray(im.crop((l, t, l + c, t + c)), np.float32) / 255 - mean) / std).transpose(2, 0, 1)
    return prep


def mobilenet_prep(p):
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    im = Image.open(p).convert('RGB')
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return ((np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - mean) / std).transpose(2, 0, 1)


def run(session, X, name, bs=32):
    return np.concatenate([session.run(None, {name: X[i:i + bs]})[0] for i in range(0, len(X), bs)])


def main():
    model_dir, vision, sets = sys.argv[1], sys.argv[2], sys.argv[3:]
    pre = json.load(open(os.path.join(model_dir, 'preprocess.json')))
    dishes = with_taste(load_all(), load_fndds())
    ids = [d['id'] for d in dishes]
    T = np.load(os.path.join(model_dir, 'text_emb.npy')).astype(np.float32)
    assert T.shape[0] == len(ids), 'text_emb.npy is stale: re-run embed_dishes.py'
    taste = np.array([[d['taste'][c] for c in CH] for d in dishes])
    sess = ort.InferenceSession(vision, providers=['CPUExecutionProvider'])
    mob = ort.InferenceSession(os.path.join(ROOT, 'data/eval/food101_mobilenet/model.onnx'), providers=['CPUExecutionProvider'])
    f101 = json.load(open(os.path.join(ROOT, 'data/eval/food101_mobilenet/labels.json')))
    prep = image_prep(pre)
    conf_all, right_all = [], []
    for spec in sets:
        label, d = spec.split(':', 1)
        files = sorted(glob.glob(os.path.join(d, '*.jpg')))
        gt_ids = [os.path.basename(f).rsplit('__', 1)[1][:-4] for f in files]
        keep = [i for i, g in enumerate(gt_ids) if g in ids]
        files, gt_ids = [files[i] for i in keep], [gt_ids[i] for i in keep]
        gt = np.array([ids.index(g) for g in gt_ids])
        X = np.stack([prep(f) for f in files])
        I = run(sess, X, sess.get_inputs()[0].name)
        S = I @ T.T
        rank = S.argsort(1)[:, ::-1]
        top = rank[:, :TOPK]
        w = np.exp(SCALE * (np.take_along_axis(S, top, 1) - S.max(1, keepdims=True)))
        w /= w.sum(1, keepdims=True)
        blend = (w[:, :, None] * taste[top]).sum(1)
        p_all = np.exp(SCALE * (S - S.max(1, keepdims=True)))
        conf = (p_all / p_all.sum(1, keepdims=True)).max(1)
        right = rank[:, 0] == gt
        conf_all += list(conf); right_all += list(right)
        mae = lambda est: np.abs(est - taste[gt]).mean()
        M = run(mob, np.stack([mobilenet_prep(f) for f in files]), mob.get_inputs()[0].name, bs=1)  # exported with batch 1
        mob_taste = taste[[ids.index(f101[k]) for k in M.argmax(1)]]
        in101 = np.array([g in f101 for g in gt_ids])
        print(f'{label}: n={len(files)} ({in101.sum()} Food-101 dishes)  top1 {right.mean():.3f}  top5 {np.mean([g in r[:5] for g, r in zip(gt, rank)]):.3f}')
        print(f'   taste MAE: top1 {mae(taste[rank[:, 0]]):.3f}  blend {mae(blend):.3f}   old MobileNet {mae(mob_taste):.3f}'
              f'  (on Food-101 dishes only: new {np.abs(blend - taste[gt])[in101].mean():.3f} old {np.abs(mob_taste - taste[gt])[in101].mean():.3f};'
              f' others: new {np.abs(blend - taste[gt])[~in101].mean() if (~in101).any() else float("nan"):.3f}'
              f' old {np.abs(mob_taste - taste[gt])[~in101].mean() if (~in101).any() else float("nan"):.3f})')
        wrong = [(gt_ids[i], ids[rank[i, 0]]) for i in np.where(~right)[0]]
        print('   e.g. wrong:', ', '.join(f'{a}->{b}' for a, b in wrong[:12]))
    conf_all, right_all = np.array(conf_all), np.array(right_all)
    print('confidence threshold: share flagged / top1 accuracy of the rest / share of wrong ones flagged')
    for th in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        f = conf_all < th
        print(f'   {th:.2f}: {f.mean():.2f} / {right_all[~f].mean():.3f} / {f[~right_all].mean():.2f}')


if __name__ == '__main__':
    main()
