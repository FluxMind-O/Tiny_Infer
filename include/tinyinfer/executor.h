#pragma once
// 执行引擎：拓扑排序 + 异步 Stream 调度 + CUDA Graph 固化
#include "common.h"
#include "graph.h"
#include "memory.h"
#include <vector>

namespace tinynfer {

struct ExecOptions {
    bool use_async_stream = true;   // 保留接口；compute 图在单一 stream 上顺序执行，
                                    // 以保证复用 buffer 的读写依赖正确（避免竞争）
    bool use_cuda_graph = true;     // 固化推理流程，消除 CPU launch overhead
    bool fuse_gemm_relu = true;     // 将 Linear+ReLU 融合为 FusedLinearReLU
    int num_streams = 3;            // 保留接口，当前仅创建 stream[0] 用于 compute
};

class Executor {
public:
    Executor(ComputeGraph& graph, const ExecOptions& opt = ExecOptions{});
    ~Executor();

    // 准备：静态显存规划 + 分配 + （可选）构建 CUDA Graph
    void prepare();

    // 设置输入 tensor（host -> device）。input 为 row-major [batch, in_features]
    void set_input(const dtype* h_input, int batch);

    // 取输出（device -> host）。返回输出 tensor 的 host 拷贝。
    void get_output(std::vector<dtype>& h_output);

    // 执行一次前向推理
    void run();

    // 性能基准：重复 n 次，返回平均延迟（微秒）
    double benchmark(int n);

    size_t planned_bytes() const { return planner_.total_bytes(); }

    void set_io(int input_tid, int output_tid) {
        input_tid_ = input_tid;
        output_tid_ = output_tid;
    }

private:
    void build_graph();          // 执行所有节点（捕获或即时）
    void capture_cuda_graph();   // 第一次推理：cudaStreamBeginCapture
    void launch_cuda_graph();    // 后续推理：cudaGraphLaunch

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

    std::vector<cudaStream_t> streams_;
    bool graph_captured_ = false;
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
};

}  // namespace tinynfer
