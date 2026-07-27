#pragma once
// 执行引擎：拓扑排序 + 单 Stream 调度 + CUDA Graph 固化
#include "common.h"
#include "graph.h"
#include "memory.h"
#include <vector>

namespace tinyinfer {

struct ExecOptions {
    bool fuse_gemm_relu = true;   // 将 Linear+ReLU 融合为 FusedLinearReLU（--no-fuse 关闭）
    bool reuse_memory = true;     // 静态显存规划复用中间 buffer（--no-reuse 关闭）
    bool use_cuda_graph = true;   // 固化推理流程，消除 CPU launch overhead（--no-graph 关闭）
};

// benchmark 统计结果（cudaEvent 记录 GPU 端时间，不含 H2D/D2H 拷贝）
struct BenchStats {
    double mean_us = 0.0;
    double stddev_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    int iters = 0;
};

class Executor {
public:
    Executor(ComputeGraph& graph, const ExecOptions& opt = ExecOptions{});
    ~Executor();

    // 准备：拓扑排序 + 生命周期分析 + 静态显存规划 + 分配 + 预热（Graph 捕获前）
    void prepare();

    // 设置输入 tensor（host -> device）。input 为 row-major [batch, in_features]
    void set_input(const dtype* h_input, int batch);

    // 取输出（device -> host）。返回输出 tensor 的 host 拷贝。
    void get_output(std::vector<dtype>& h_output);

    // 执行一次前向推理（纯计算，不含 H2D/D2H）
    void run();

    // 性能基准：预热 warmup 次后测量 iters 次，cudaEvent 逐次计时，
    // 返回 mean / stddev / P50 / P99 / min（微秒）
    BenchStats benchmark(int warmup = 100, int iters = 1000);

    size_t planned_bytes() const { return planner_.total_bytes(); }

    void set_io(int input_tid, int output_tid) {
        input_tid_ = input_tid;
        output_tid_ = output_tid;
    }

private:
    void run_once_on_stream();     // 按拓扑序在 stream 上执行全部节点
    void capture_cuda_graph();     // 预热完成后捕获整个计算流

    ComputeGraph& graph_;
    ExecOptions opt_;
    std::vector<int> order_;
    MemoryPlanner planner_;

    dtype* dev_buffer_ = nullptr;     // 算子间复用的统一 device buffer
    dtype* dev_input_ = nullptr;      // 输入 buffer（单独分配，不参与复用）
    dtype* dev_output_ = nullptr;     // 输出 buffer（单独分配）
    int input_tid_ = -1;
    int output_tid_ = -1;
    int batch_ = 1;

    cudaStream_t stream_ = nullptr;   // 单 Stream 设计
    bool graph_captured_ = false;
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
};

}  // namespace tinyinfer
