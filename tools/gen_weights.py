#!/usr/bin/env python3
"""生成 TinyInfer 测试模型的权重、输入与 numpy 参考输出。

用法:
    python3 tools/gen_weights.py            # 生成全部模型与数据
    python3 tools/gen_weights.py mlp3       # 只生成指定模型

产物（以 mlp3 为例）:
    models/mlp3.json                     模型结构描述
    models/weights/mlp3/layer*_*.bin     权重/偏置
    data/mlp3_b{B}.bin / *_ref.bin       batch ∈ {1,4,8} 的输入与参考输出
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 测试模型定义：与计划文档 6.1 节一致
# mlp_large：输入 [B, 1024]，4 层 Linear + 3 层 ReLU + 1 层 Softmax，8 个中间张量
MODELS = {
    "mlp3": [
        ("Linear", 128, 64),
        ("ReLU", None, None),
        ("Linear", 64, 32),
        ("ReLU", None, None),
        ("Linear", 32, 10),
        ("Softmax", None, None),
    ],
    "mlp_large": [
        ("Linear", 1024, 512),
        ("ReLU", None, None),
        ("Linear", 512, 256),
        ("ReLU", None, None),
        ("Linear", 256, 128),
        ("ReLU", None, None),
        ("Linear", 128, 10),
        ("Softmax", None, None),
    ],
}

BATCHES = [1, 4, 8]


def save_bin(path, arr):
    with open(path, "wb") as f:
        f.write(arr.astype(np.float32).tobytes())


def gen_model(name, layers):
    rng = np.random.default_rng(42)
    weight_dir = os.path.join(ROOT, "models", "weights", name)
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 1. 生成权重 + 写 JSON 模型描述
    weights = {}
    weight_files = {}
    json_layers = []
    for layer_idx, (typ, inf, outf) in enumerate(layers):
        if typ == "Linear":
            W = rng.standard_normal((outf, inf)).astype(np.float32) * 0.1
            b = rng.standard_normal(outf).astype(np.float32) * 0.1
            wkey, bkey = f"layer{layer_idx}_weight", f"layer{layer_idx}_bias"
            save_bin(os.path.join(weight_dir, f"{wkey}.bin"), W)
            save_bin(os.path.join(weight_dir, f"{bkey}.bin"), b)
            weights[layer_idx] = (W, b)
            weight_files[wkey] = f"weights/{name}/{wkey}.bin"
            weight_files[bkey] = f"weights/{name}/{bkey}.bin"
            json_layers.append(
                {"type": "Linear", "in_features": inf, "out_features": outf})
        elif typ == "ReLU":
            json_layers.append({"type": "ReLU", "inplace": True})
        elif typ == "Softmax":
            json_layers.append({"type": "Softmax", "dim": 1})

    model_json = {
        "name": name,
        "input_shape": [1, layers[0][1]],
        "layers": json_layers,
        "weights": weight_files,
    }
    with open(os.path.join(ROOT, "models", f"{name}.json"), "w") as f:
        json.dump(model_json, f, indent=2, ensure_ascii=False)

    # 2. numpy 参考前向
    def forward(x):
        ref = x
        for layer_idx, (typ, inf, outf) in enumerate(layers):
            if typ == "Linear":
                W, b = weights[layer_idx]
                ref = ref @ W.T + b
            elif typ == "ReLU":
                ref = np.maximum(ref, 0.0)
            elif typ == "Softmax":
                ref = np.exp(ref - ref.max(axis=1, keepdims=True))
                ref = ref / ref.sum(axis=1, keepdims=True)
        return ref

    # 3. 生成各 batch 的输入与参考输出
    in_f = layers[0][1]
    for B in BATCHES:
        x = rng.standard_normal((B, in_f)).astype(np.float32)
        save_bin(os.path.join(data_dir, f"{name}_b{B}.bin"), x)
        save_bin(os.path.join(data_dir, f"{name}_b{B}_ref.bin"), forward(x))

    print(f"[gen] {name}: weights + batches {BATCHES} done")


def main():
    targets = sys.argv[1:] or list(MODELS.keys())
    for name in targets:
        if name not in MODELS:
            print(f"unknown model: {name}", file=sys.stderr)
            sys.exit(1)
        gen_model(name, MODELS[name])


if __name__ == "__main__":
    main()
