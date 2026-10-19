"""Export AlexKoff88's torchvision MobileNetV2/Food-101 checkpoint to ONNX (the repo's .onnx has untrained weights)."""
import sys, torch, torchvision
ckpt, out = sys.argv[1:3]
sd = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict']
m = torchvision.models.mobilenet_v2(num_classes=101)
m.load_state_dict({k.removeprefix('module.'): v for k, v in sd.items()})
m.eval()
torch.onnx.export(m, torch.zeros(1, 3, 224, 224), out, input_names=['input'], output_names=['logits'],
                  opset_version=17, do_constant_folding=True, dynamo=False)
