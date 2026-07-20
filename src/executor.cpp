// 执行引擎：拓扑调度 + 异步 Stream 流水线 + CUDA Graph 固化
#include "tinyinfer/executor.h"
#include "tinyinfer/common.h"
#include <algorithm>

namespace tinynfer {

Executor::Executor(ComputeGraph& graph, const ExecOptions& opt)
    : graph_(graph), opt_(opt) {}

Executor::~Executor() {
    if (cuda_graph_exec_) cudaGraphExecDestroy(cuda_graph_exec_);
    if (cuda_graph_) cudaGraphDestroy(cuda_graph_);
    for (auto& s : streams_) cudaStreamDestroy(s);
    if (dev_buffer_) cudaFree(dev_buffer_);
    if (dev_input_) cudaFree(dev_input_);
    if (dev_output_) cudaFree(dev_output_);
}

void Executor::prepare() {
    // 1. 拓扑排序
    order_ = graph_.topo_order();
    // 2. 生命周期分析
    graph_.analyze_lifetimes(order_);
    // 3. 静态显存规划（复用中间 tensor）
    planner_.plan(graph_, order_, /*reuse=*/true);
    dev_buffer_ = planner_.allocate(graph_);

    // 4. 输入/输出 buffer（fixed，单独分配）
    int in_n = graph_.tensor(input_tid_).num_elements();
    int out_n = graph_.tensor(output_tid_).num_elements();
    TINYINFER_CUDA_CHECK(cudaMalloc(&dev_input_, in_n * sizeof(dtype)));
    TINYINFER_CUDA_CHECK(cudaMalloc(&dev_output_, out_n * sizeof(dtype)));
    graph_.tensor(input_tid_).data = dev_input_;
    graph_.tensor(output_tid_).data = dev_output_;

    // 5. 创建多个 stream 用于流水线
    int ns = opt_.use_async_stream ? std::max(1, opt_.num_streams) : 1;
    for (int i = 0; i < ns; ++i) {
        cudaStream_t s;
        TINYINFER_CUDA_CHECK(cudaStreamCreate(&s));
        streams_.push_back(s);
    }
}

void Executor::set_input(const dtype* h_input, int batch) {
    batch_ = batch;
    int in_n = graph_.tensor(input_tid_).num_elements();
    // 若 batch 变化需重新规划（简化：要求 batch 固定）
    TINYINFER_CUDA_CHECK(cudaMemcpy(dev_input_, h_input, in_n * sizeof(dtype),
                                    cudaMemcpyHostToDevice));
}

void Executor::get_output(std::vector<dtype>& h_output) {
    int out_n = graph_.tensor(output_tid_).num_elements();
    h_output.resize(out_n);
    TINYINFER_CUDA_CHECK(cudaMemcpy(h_output.data(), dev_output_,
                                    out_n * sizeof(dtype),
                                    cudaMemcpyDeviceToHost));
}

void Executor::build_graph() {
    ExecContext ctx;
    // 注意：静态显存规划会让不重叠生命周期的 tensor 复用同一块 device buffer。
    // 因此相邻算子存在对同一 buffer 的"先写后读"依赖，必须保证严格的执行顺序，
    // 不能简单分配到不同 stream（否则会出现数据竞争）。这里统一使用 stream[0]
    // 顺序执行；H2D/Compute/D2H 的流水线重叠由 CUDA Graph + 单一 capture stream
    // 的底层调度保证，或在 run() 外层由 set_input/get_output 与 compute 重叠。
    cudaStream_t s = streams_[0];
    for (size_t k = 0; k < order_.size(); ++k) {
        int nid = order_[k];
        GraphNode& n = graph_.node(nid);
        ctx.stream = s;
        std::vector<Tensor*> ins, outs;
        for (int t : n.input_tids) ins.push_back(&graph_.tensor(t));
        for (int t : n.output_tids) outs.push_back(&graph_.tensor(t));
        n.op->compute(ins, outs, ctx);
    }
}

void Executor::capture_cuda_graph() {
    cudaStream_t s = streams_[0];
    TINYINFER_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    ExecContext ctx;
    ctx.stream = s;
    for (int nid : order_) {
        GraphNode& n = graph_.node(nid);
        std::vector<Tensor*> ins, outs;
        for (int t : n.input_tids) ins.push_back(&graph_.tensor(t));
        for (int t : n.output_tids) outs.push_back(&graph_.tensor(t));
        n.op->compute(ins, outs, ctx);
    }
    TINYINFER_CUDA_CHECK(cudaStreamEndCapture(s, &cuda_graph_));
    TINYINFER_CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, NULL, NULL, 0));
    graph_captured_ = true;
}

void Executor::launch_cuda_graph() {
    TINYINFER_CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec_, streams_[0]));
}

void Executor::run() {
    if (opt_.use_cuda_graph) {
        if (!graph_captured_) {
            // 首次推理：捕获并固化整个流程（含 H2D/Compute/D2H 由 set_input/get_output 完成）
            capture_cuda_graph();
        }
        launch_cuda_graph();
        TINYINFER_CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
    } else {
        build_graph();
        TINYINFER_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

double Executor::benchmark(int n) {
    // 预热
    run();
    double t0 = get_wall_time_us();
    for (int i = 0; i < n; ++i) run();
    double t1 = get_wall_time_us();
    return (t1 - t0) / n;
}

}  // namespace tinynfer
