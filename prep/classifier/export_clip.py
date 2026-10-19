"""Export a CLIP-family image tower (transformers) to ONNX: pixels in, L2-normalized embedding out.

    python prep/classifier/export_clip.py <hf_model> <out_dir>

Writes <out_dir>/vision_fp32.onnx, vision_fp16.onnx (fp16 weights and math, fp32 input/output) and
vision_q4.onnx (4-bit MatMulNBits weights, fp32 activations), plus preprocess.json for the browser.
The text tower is not exported: dish text embeddings are computed at build time (embed_dishes.py).
"""
import json, os, sys
import torch
from transformers import AutoModel, AutoProcessor


class ImageTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        e = self.model.get_image_features(pixel_values=pixel_values)
        e = getattr(e, 'pooler_output', e)
        return torch.nn.functional.normalize(e, dim=-1)


def preprocess_config(proc):
    ip = proc.image_processor
    size = ip.size.get('shortest_edge') or ip.size.get('height')
    crop = (ip.crop_size or {}).get('height', size) if getattr(ip, 'do_center_crop', False) else None
    return {'size': size, 'crop': crop, 'mean': list(ip.image_mean), 'std': list(ip.image_std),
            'resize': 'shortest_edge' if 'shortest_edge' in ip.size else 'squash'}


def main():
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    model = AutoModel.from_pretrained(name).eval()
    proc = AutoProcessor.from_pretrained(name)
    pre = preprocess_config(proc)
    side = pre['crop'] or pre['size']
    tower = ImageTower(model)
    x = torch.randn(1, 3, side, side)
    fp32 = os.path.join(out, 'vision_fp32.onnx')
    torch.onnx.export(tower, (x,), fp32, input_names=['pixel_values'], output_names=['embedding'],
                      dynamic_axes={'pixel_values': {0: 'batch'}, 'embedding': {0: 'batch'}}, opset_version=17, dynamo=False)

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m = onnx.load(fp32)
    onnx.save(convert_float_to_float16(m, keep_io_types=True), os.path.join(out, 'vision_fp16.onnx'))

    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(fp32), block_size=32, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(os.path.join(out, 'vision_q4.onnx'), use_external_data_format=False)

    with torch.no_grad():
        ref = tower(x).numpy()
    import onnxruntime as ort
    for f in ('vision_fp32.onnx', 'vision_fp16.onnx', 'vision_q4.onnx'):
        p = os.path.join(out, f)
        e = ort.InferenceSession(p, providers=['CPUExecutionProvider']).run(None, {'pixel_values': x.numpy()})[0]
        print(f'{f}: {os.path.getsize(p) / 1e6:.1f} MB, cosine to torch {float((e * ref).sum()):.4f}')
    json.dump({'model': name, 'dim': int(ref.shape[1]), **pre}, open(os.path.join(out, 'preprocess.json'), 'w'), indent=1)
    print(json.dumps(pre))


if __name__ == '__main__':
    main()
