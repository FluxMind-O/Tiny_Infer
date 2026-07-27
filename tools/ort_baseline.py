#!/usr/bin/env python3
"""阶段零：ONNX Runtime 基线（验证假设 H1：通用框架在小模型场景的调度开销）。

用与正式 benchmark 相同的口径测量 ORT 端到端延迟：
预热 100 次 + 测量 1000 次，报告 mean / std / P50 / P99 / min。
模型为 mlp_large（1024->512->256->128->10），权重与 data/ 下的测试数据一致。

用法:
    python3 tools/ort_baseline.py [batch]
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_onnx(path):
    """用与 gen_weights.py 相同的权重构建 mlp_large ONNX 图。"""
    import onnx
    from onnx import helper, TensorProto

    weight_dir = os.path.join(ROOT, "models", "weights", "mlp_large")
    dims = [(1024, 512), (512, 256), (256, 128), (128, 10)]

    nodes, initializers = [], []
    prev = "input"
    for i, (inf, outf) in enumerate(dims):
        W = np.fromfile(os.path.join(weight_dir, f"layer{i*2}_weight.bin"),
                        dtype=np.float32).reshape(outf, inf)
        b = np.fromfile(os.path.join(weight_dir, f"layer{i*2}_bias.bin"),
                        dtype=np.float32)
        initializers.append(helper.make_tensor(f"W{i}", TensorProto.FLOAT,
                                               [outf, inf], W.ravel()))
        initializers.append(helper.make_tensor(f"b{i}", TensorProto.FLOAT,
                                               [outf], b))
        # Gemm: Y = alpha*A*B^T + beta*b（W 按 [out,in] 存储，transB=1）
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"b{i}"],
                                      [f"gemm{i}"], transB=1))
        prev = f"gemm{i}"
        if i < len(dims) - 1:
            nodes.append(helper.make_node("Relu", [prev], [f"relu{i}"]))
            prev = f"relu{i}"
    nodes.append(helper.make_node("Softmax", [prev], ["output"], axis=1))

    graph = helper.make_graph(
        nodes, "mlp_large",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["B", 1024])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["B", 10])],
        initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # 兼容 ORT 1.23
    onnx.checker.check_model(model)
    onnx.save(model, path)


def bench(fn, warmup=100, iters=1000):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    arr = np.array(samples)
    return (arr.mean(), arr.std(), np.percentile(arr, 50),
            np.percentile(arr, 99), arr.min())


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    import onnxruntime as ort

    onnx_path = os.path.join(ROOT, "models", "mlp_large.onnx")
    build_onnx(onnx_path)

    x = np.fromfile(os.path.join(ROOT, "data", f"mlp_large_b{batch}.bin"),
                    dtype=np.float32).reshape(batch, 1024)
    ref = np.fromfile(os.path.join(ROOT, "data", f"mlp_large_b{batch}_ref.bin"),
                      dtype=np.float32).reshape(batch, 10)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    # 数值校验
    out = sess.run(None, {"input": x})[0]
    max_err = np.abs(out - ref).max()
    print(f"[ort] max abs err vs numpy ref: {max_err:.3e}")
    assert max_err < 1e-3, "ORT 数值校验失败"

    mean, std, p50, p99, mn = bench(lambda: sess.run(None, {"input": x}))
    print(f"[ort] mlp_large batch={batch}, warmup 100 + 1000 iters (wall time, CPU):")
    print(f"[ort]   mean={mean:.2f} us  std={std:.2f} us  P50={p50:.2f} us  "
          f"P99={p99:.2f} us  min={mn:.2f} us")


if __name__ == "__main__":
    main()
