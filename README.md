# TinyInfer —— 面向小模型场景的 CUDA 推理引擎

> 支持计算图加载、算子融合、静态显存规划与 CUDA Graph 固化的 GPU 推理引擎，
> 专为小规模 MLP（hidden ≤ 1024，batch ≤ 8，FP32）设计。
> 定位：**教学型推理引擎 + 可控消融实验平台**（定位依据见「项目动机」）。

## 项目动机（数据说话，非既定结论）

开发前用 ONNX Runtime 基线验证了两个前置假设（完整数据见
[docs/motivation_report.md](docs/motivation_report.md)）：

- **H1（通用框架开销 ≥ 2×）：不成立。** ORT 1.23 在纯 CPU（Xeon 4214R）上跑
  mlp_large batch=1 仅需 ~48 μs，不慢于本引擎 GPU 全优化配置（~74 μs）。
- **H2（Launch 开销占比 ≥ 30%）：部分成立但被高估。** 实测每次 kernel launch
  摊销 ~1.4 μs，在 8-kernel 链路中占比约 5~10%。

因此本项目不声称「比通用框架快」，价值在于把推理引擎的四大机制完整实现，
并用**可 30 秒复现的消融数据**讲清楚每个优化的真实收益与边界。

## 四层架构

```
+--------------------------------------------------+
|  Model Loader (JSON + 二进制权重)                  |
|    解析模型描述，构建 DAG，自动融合 Linear+ReLU      |
+--------------------------------------------------+
|  Compute Graph (DAG)                              |
|    Kahn 拓扑排序 + 张量生命周期分析                 |
+--------------------------------------------------+
|  Memory Planner                                   |
|    生命周期区间复用，256B 对齐，一次 cudaMalloc      |
+--------------------------------------------------+
|  Executor（单 Stream）                             |
|    拓扑顺序执行 -> CUDA Graph 捕获与重放            |
+--------------------------------------------------+
```

## 三个核心优化（收益均可复现）

1. **FusedLinearReLU 融合算子**：GEMM epilogue 阶段就地完成 BiasAdd+ReLU，
   全局内存只写一次。实测独立收益 ~4.7 μs（batch=1），其中约 2 μs 与
   CUDA Graph 的 launch 节省重叠（2×2 解耦实验测得，见下文）。
2. **静态显存规划**：按生命周期区间贪心复用中间 buffer，256B 对齐，
   初始化一次 `cudaMalloc`。对延迟贡献≈0（预期内），价值在显存峰值与
   分配次数：mlp_large 中间张量 4 块复用为 3 块 buffer，峰值降 16~20%。
3. **CUDA Graph 固化**：预热后捕获纯计算 kernel 序列，后续一次
   `cudaGraphLaunch` 提交。实测省 ~4.5 μs（launch 开销摊销）。
   H2D/D2H 在图外通过固定地址交互。

## Benchmark（实测数据）

测试环境：RTX 3080 Ti / CUDA 11.8 / 驱动 580.76.05 / Xeon Silver 4214R。
模型：mlp_large（1024→512→256→128→10）。口径：预热 100 + 测量 1000 次，
cudaEvent GPU 端计时（不含 H2D/D2H）。复现：`./build/tinyinfer_test .`

延迟单位 μs（mean；完整 mean±std/P50/P99/min 见测试输出）：

| 配置 | kernels | batch=1 | batch=4 | batch=8 |
|---|---|---|---|---|
| Config 0 无优化 | 8 | 95.8 | 83.6 | 84.3 |
| Config 1 +融合 | 5 | 90.9 | 79.0 | 79.2 |
| Config 2 +显存复用 | 5 | 78.4 | 79.1 | 82.5 |
| Config 3 +CUDA Graph | 5 | 73.9 | 74.6 | 76.0 |
| cuBLAS 基线（参照） | 12 | 53.0 | 53.4 | 70.1 |
| ORT 参照（CPU，wall time） | - | 48.1 | - | 136.7(P50) |

融合 × Graph 2×2 解耦（batch=1，reuse 关闭）：

| fuse \ graph | 关 | 开 |
|---|---|---|
| 关 | 82.9 | 76.8 |
| 开 | 78.3 | 74.0 |

融合单独开省 4.7 μs；已开 Graph 后融合的边际收益收窄为 2.9 μs
（交互项 ~1.8 μs，即融合的 launch 收益被 Graph 覆盖的部分）。

### 诚实归因

- **显存复用对延迟无显著影响**（Config 1→2 差异 < 2×stddev），符合预期——
  它不改动任何 kernel，价值体现在显存类指标，见上。
- **手写 GEMM 落后 cuBLAS**：Config 3（74 μs）vs cuBLAS 基线（53 μs）。
  归因：M ≤ 8 的 skinny 形状下 tiling kernel（TILE=16）并行度不足——
  M=1 单层微基准实测手写 48.3 μs vs cuBLAS 11.8 μs，有效带宽
  ~43 GB/s vs ~178 GB/s（3080 Ti 理论 912 GB/s，达成率 5% vs 20%）。
  skinny GEMM 是带宽瓶颈型负载，但当前实现的瓶颈在并行度；
  split-K / GEMV 专用路径是已识别的改进方向（未实施）。
- 同环境两次重测存在 ~10% 漂移（GPU 时钟策略），消融结论均按
  「增量 ≥ 2×stddev 才视为显著」判读。

## 构建与运行

```bash
python3 tools/gen_weights.py        # 生成 mlp3 / mlp_large 权重与 batch{1,4,8} 数据
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   # 按显卡调整：A100=80 T4=75
cmake --build build -j8
```

```bash
# 推理（开关可任意组合）
./build/tinyinfer --model models/mlp_large.json --input data/mlp_large_b1.bin
./build/tinyinfer --model models/mlp_large.json --input data/mlp_large_b8.bin --batch 8 --bench
./build/tinyinfer --model models/mlp_large.json --input data/mlp_large_b1.bin --no-fuse --no-reuse --no-graph
./build/tinyinfer --model models/mlp_large.json --input data/mlp_large_b1.bin --cublas  # cuBLAS 基线

# 统一测试：数值正确性（全配置 × batch{1,4,8}）+ 显存对比 + 消融基准 + 2×2 解耦
./build/tinyinfer_test .

# 阶段零 ORT 基线（需 pip install onnxruntime onnx）
python3 tools/ort_baseline.py 1
```

## 支持范围与局限性（非目标即设计选择）

- 仅支持线性 MLP（Linear / BiasAdd / ReLU / Softmax），FP32，固定 Shape，
  batch ∈ [1, 8]，单 NVIDIA GPU；
- 不支持 CNN/Transformer、动态 Shape、FP16/INT8、Tensor Core、多卡多流、
  服务化部署；
- 手写 GEMM 为通用 tiling 实现，未做 skinny 形状特化，性能落后于 cuBLAS
  （数据见上），替换 cuBLAS 不是当前形态下的性能卖点；
- 模型格式为自定义 JSON + 二进制权重（`tools/gen_weights.py` 可生成示例）。

## 目录结构

```
include/tinyinfer/   引擎头文件（tensor/op/graph/memory/executor/registry/json_min）
src/                 引擎实现（kernels.cu 为手写 tiling GEMM + 融合/基础算子）
src/main.cpp         CLI（--no-fuse/--no-reuse/--no-graph/--cublas/--bench）
test/test_main.cpp   统一测试入口（数值 / 显存 / 消融基准 / 2×2 解耦）
tools/gen_weights.py 权重与测试数据生成；tools/ort_baseline.py ORT 基线
docs/motivation_report.md  阶段零动机验证报告（H1/H2 结论与归因）
```

## License

MIT，见 [LICENSE](LICENSE)。
