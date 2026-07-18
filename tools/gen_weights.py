#!/usr/bin/env python3
"""生成 TinyInfer 示例模型权重 + 示例输入，并用 numpy 计算参考输出用于验证。

用法:
    python3 tools/gen_weights.py
产物:
    models/mlp3.json           (已存在，这里仅生成权重)
    models/weights/layer*_weight.bin / layer*_bias.bin
    data/sample.bin            (随机输入)
    data/ref_output.bin        (numpy 前向计算参考输出)
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型配置（需与 models/mlp3.json 一致）
LAYERS = [
    ("Linear", 128, 64),
    ("ReLU", None, None),
    ("Linear", 64, 32),
    ("ReLU", None, None),
    ("Linear", 32, 10),
    ("Softmax", None, None),
]

WEIGHT_DIR = os.path.join(ROOT, "models", "weights")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(WEIGHT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

rng = np.random.default_rng(42)
np.set_printoptions(precision=6)


def save_bin(path, arr):
    with open(path, "wb") as f:
        f.write(arr.astype(np.float32).tobytes())


# 收集 Linear 层参数并写文件
weights = {}
for layer_idx, (typ, inf, outf) in enumerate(LAYERS):
    if typ == "Linear":
        W = rng.standard_normal((outf, inf)).astype(np.float32) * 0.1
        b = rng.standard_normal(outf).astype(np.float32) * 0.1
        wpath = os.path.join(WEIGHT_DIR, f"layer{layer_idx}_weight.bin")
        bpath = os.path.join(WEIGHT_DIR, f"layer{layer_idx}_bias.bin")
        save_bin(wpath, W)
        save_bin(bpath, b)
        weights[layer_idx] = (W, b)

# 生成随机输入
x = rng.standard_normal((1, 128)).astype(np.float32)
save_bin(os.path.join(DATA_DIR, "sample.bin"), x)
x4 = rng.standard_normal((4, 128)).astype(np.float32)
save_bin(os.path.join(DATA_DIR, "batch4.bin"), x4)

# numpy 参考前向
ref = x
for layer_idx, (typ, inf, outf) in enumerate(LAYERS):
    if typ == "Linear":
        W, b = weights[layer_idx]
        ref = ref @ W.T + b
    elif typ == "ReLU":
        ref = np.maximum(ref, 0.0)
    elif typ == "Softmax":
        ref = np.exp(ref - ref.max(axis=1, keepdims=True))
        ref = ref / ref.sum(axis=1, keepdims=True)

save_bin(os.path.join(DATA_DIR, "ref_output.bin"), ref)
print("generated weights + sample.bin + ref_output.bin")
print("ref output:", ref)
