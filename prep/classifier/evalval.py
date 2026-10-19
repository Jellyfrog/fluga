"""Top-1/top-5 of ONNX models on labelled images named <idx>__<label>.jpg; also flags systematic label swaps."""
import json, sys, glob, os, collections
import numpy as np, onnxruntime as ort
sys.argv, rest = sys.argv[:1], sys.argv[1:]
labels_path, imgdir, models = rest[0], rest[1], rest[2:]
labels = json.load(open(labels_path))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
def prep(p):
    im = Image.open(p).convert('RGB'); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return ((np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - mean) / std).transpose(2, 0, 1)[None]
files = sorted(glob.glob(os.path.join(imgdir, '*.jpg'))); X = [prep(f) for f in files]
gt = [os.path.basename(f).split('__')[1][:-4] for f in files]
preds = {}
for mp in models:
    s = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
    Z = np.stack([s.run(None, {'input': x})[0][0] for x in X]); preds[mp] = Z
    t5 = Z.argsort(1)[:, ::-1][:, :5]
    top1 = np.mean([labels[t[0]] == g for t, g in zip(t5, gt)]); top5 = np.mean([g in [labels[i] for i in t] for t, g in zip(t5, gt)])
    conf = collections.Counter((g, labels[t[0]]) for t, g in zip(t5, gt) if labels[t[0]] != g)
    print(f"{os.path.basename(mp)}: n={len(gt)} top1={top1:.3f} top5={top5:.3f}; common confusions: {conf.most_common(6)}")
if len(models) == 2:
    a, b = preds.values(); print('top1 agreement', np.mean(a.argmax(1) == b.argmax(1)))
