"""Run a Food-101 ONNX model on test images; torchvision-style Resize(256)+CenterCrop(224)+ImageNet norm."""
import json, sys, glob, os
import numpy as np, onnxruntime as ort
from PIL import Image
model, labels, imgdir = sys.argv[1:4]
labels = json.load(open(labels))
sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
def prep(p):
    im = Image.open(p).convert('RGB'); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR); w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - mean) / std
    return a.transpose(2, 0, 1)[None]
res = {}
for p in sorted(glob.glob(os.path.join(imgdir, '*.jpg'))):
    z = sess.run(None, {sess.get_inputs()[0].name: prep(p)})[0][0]
    e = np.exp(z - z.max()); pr = e / e.sum(); top = pr.argsort()[::-1][:3]
    res[os.path.basename(p)] = z
    print(f"{os.path.basename(p):18s}", ', '.join(f"{labels[i]} {pr[i]:.2f}" for i in top))
if len(sys.argv) > 4: np.savez(sys.argv[4], **res)
