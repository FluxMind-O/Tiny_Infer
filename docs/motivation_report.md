# TinyInfer 动机验证报告（阶段零）

日期：2026-07-27
测试模型：mlp_large（1024→512→256→128→10，4×Linear + 3×ReLU + Softmax）

## 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti（驱动 580.76.05） |
| CUDA | 11.8 |
| CPU | Intel Xeon Silver 4214R @ 2.40GHz |
| ONNX Runtime | 1.23.2（CPUExecutionProvider） |

计时口径：预热 100 次 + 测量 1000 次。引擎侧为 cudaEvent GPU 端纯计算时间（不含 H2D/D2H）；
ORT 侧为 Python wall time（CPU 推理，含其内部全部调度）。两侧口径不同，ORT 数据仅用于
判断「通用框架的调度开销是否在小模型场景占主导」，不构成同条件性能对比。

## 实测数据

### H1：框架开销假设（ORT 端到端 ≥ 2× 手写链路）

| 实现 | batch=1 mean | batch=8 mean |
|---|---|---|
| ONNX Runtime (CPU) | 48.1 μs (P50 48.0, P99 68.2) | 136.7 μs P50 |
| 本引擎 全优化 Config3 (GPU) | 73.9 μs | 76.0 μs |
| 本引擎 cuBLAS 基线 (GPU) | 53.0 μs | 70.1 μs |

**结论：H1 不成立。** ORT 在纯 CPU 上即可达到 48 μs，不慢于本引擎 GPU 全优化配置，
更远未出现预期的 2× 差距。按执行计划 1.1 节的预案，项目定位调整为
**「教学型推理引擎 + 可控消融实验平台」**：价值不在「比通用框架快」，而在于把图优化、
静态显存规划、CUDA Graph 的机制完整实现并用消融数据讲清楚。

### H2：Launch 开销假设（Kernel Launch 占比 ≥ 30%）

本环境未安装 nsys，无法用 timeline 直接统计 ORT 的 Kernel 间隙。改用本引擎的
消融数据间接验证（mlp_large, batch=1，2×2 解耦，reuse 关闭）：

| fuse \ graph | 关 | 开 |
|---|---|---|
| 关 | 82.9 μs | 76.8 μs |
| 开 | 78.3 μs | 74.0 μs |

- 开 Graph（5 个 kernel）省 ~4.3 μs ≈ 每次 launch 摊销 ~1.4 μs（3 次驱动调用压为 1 次，
  加上提交路径缩短），占总延迟 ~5%。
- 融合的独立收益 4.7 μs 中约 2 μs 与 Graph 重叠（交互项），即融合的收益约四成为
  launch 节省、六成为带宽节省。

**结论：H2 部分成立但被高估。** Launch 开销真实存在且可测量（~1.4 μs/次），
但在 8-kernel 链路中占比约 5~10%，未达到假设的 30%。

### 附：skinny GEMM 带宽达成率（M=1, K=1024, N=512，单 Linear 微基准）

每次 GEMM 实际搬运 ≈ 2.10 MB（权重主导）。RTX 3080 Ti 理论带宽 912 GB/s。

| 实现 | mean 延迟 | 有效带宽 | 带宽达成率 |
|---|---|---|---|
| 手写 tiling kernel (TILE=16) | 48.3 μs | ~43 GB/s | ~5% |
| cuBLAS SGEMM | 11.8 μs | ~178 GB/s | ~20% |

M=1 时 tiling kernel 每 block 16 行仅 1 行活跃，并行度严重不足；两者都远未触及
带宽上限（μs 级 kernel 还受 launch/尾延迟影响）。**skinny GEMM 是带宽瓶颈型负载，
但当前实现的主要矛盾是并行度不足**——这正是 split-K / GEMV 专用路径（计划任务 3.3，
弹性项，未实施）要解决的问题。性能落后 cuBLAS 如实记录。

## 叙事决定

1. README 动机一节按「H1 被数据推翻」如实改写，不声称超越通用框架；
2. 项目核心价值表述改为：四大机制（图融合 / 静态显存 / CUDA Graph / 手写 kernel）
   的完整实现与可复现消融；
3. 手写 GEMM 落后 cuBLAS 约 4×（端到端 74 vs 53 μs），在 README 中如实给出并归因。
