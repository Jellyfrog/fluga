"""Weight-only int8: store Conv/Gemm weights as per-output-channel symmetric int8 + DequantizeLinear.
Activations stay fp32, so every EP runs plain float Conv after a constant DQ."""
import sys, numpy as np, onnx
from onnx import numpy_helper, helper
src, dst = sys.argv[1:3]
m = onnx.load(src); g = m.graph
inits = {t.name: t for t in g.initializer}
new_nodes = []
for n in g.node:
    if n.op_type in ('Conv', 'Gemm') and n.input[1] in inits:
        w = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
        if n.op_type == 'Gemm' and any(a.name == 'transB' and a.i == 1 for a in n.attribute):
            axis = 0
        elif n.op_type == 'Gemm':
            axis = 1
        else:
            axis = 0
        red = tuple(i for i in range(w.ndim) if i != axis)
        scale = np.maximum(np.abs(w).max(axis=red), 1e-12) / 127
        shp = [1] * w.ndim; shp[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shp)), -127, 127).astype(np.int8)
        name = n.input[1]
        g.initializer.remove(inits[name])
        g.initializer.extend([numpy_helper.from_array(q, name + '_q'),
                              numpy_helper.from_array(scale.astype(np.float32), name + '_s'),
                              numpy_helper.from_array(np.zeros_like(scale, np.int8), name + '_z')])
        new_nodes.append(helper.make_node('DequantizeLinear', [name + '_q', name + '_s', name + '_z'], [name], axis=axis))
    new_nodes.append(n)
del g.node[:]; g.node.extend(new_nodes)
onnx.checker.check_model(m)
onnx.save(m, dst)
