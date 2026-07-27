// 执行引擎：拓扑调度 + 单 Stream 顺序执行 + CUDA Graph 固化
#include "tinyinfer/executor.h"
#include "tinyinfer/common.h"
#include <algorithm>
#include <cmath>

namespace tinyinfer {

Executor::Executor(ComputeGraph& graph, const ExecOptions& opt)
    : graph_(graph), opt_(opt) {}

Executor::~Executor() {
    if (cuda_graph_exec_) cudaGraphExecDestroy(cuda_graph_exec_);
    if (cuda_graph_) cudaGraphDestroy(cuda_graph_);
    if (stream_) cudaStreamDestroy(stream_);
    if (dev_buffer_) cudaFree(dev_buffer_);
    if (dev_input_) cudaFree(dev_input_);
    if (dev_output_) cudaFree(dev_output_);
}

void Executor::prepare() {
    // 1. 拓扑排序
    order_ = graph_.topo_order();
    // 2. 生命周期分析
    graph_.analyze_lifetimes(order_);
    // 3. 静态显存规划（reuse_memory 开关控制是否复用中间 tensor）
    planner_.plan(graph_, order_, /*reuse=*/opt_.reuse_memory);
    dev_buffer_ = planner_.allocate(graph_);

    // 4. 输入/输出 buffer（fixed，单独分配）
    int in_n = graph_.tensor(input_tid_).num_elements();
    int out_n = graph_.tensor(output_tid_).num_elements();
    TINYINFER_CUDA_CHECK(cudaMalloc(&dev_input_, in_n * sizeof(dtype)));
    TINYINFER_CUDA_CHECK(cudaMalloc(&dev_output_, out_n * sizeof(dtype)));
    graph_.tensor(input_tid_).data = dev_input_;
    graph_.tensor(output_tid_).data = dev_output_;

    // 5. 单 Stream：小模型场景下传输时间远小于计算时间，
    //    多 Stream 流水重叠收益趋近于零，反而增加复杂度。
    TINYINFER_CUDA_CHECK(cudaStreamCreate(&stream_));

    // 6. 若启用 CUDA Graph：先做一次完整预热推理再捕获。
    //    预热让 cuBLAS 等库完成内部 lazy 初始化（含 workspace 分配），
    //    避免捕获期间触发内部 cudaMalloc 导致捕获失败。
    if (opt_.use_cuda_graph) {
        run_once_on_stream();
        TINYINFER_CUDA_CHECK(cudaStreamSynchronize(stream_));
        capture_cuda_graph();
    }
}

void Executor::set_input(const dtype* h_input, int batch) {
    batch_ = batch;
    int in_n = graph_.tensor(input_tid_).num_elements();
    // 固定 Shape：batch 在模型加载期确定，运行期不变
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

void Executor::run_once_on_stream() {
    ExecContext ctx;
    ctx.stream = stream_;
    for (int nid : order_) {
        GraphNode& n = graph_.node(nid);
        std::vector<Tensor*> ins, outs;
        for (int t : n.input_tids) ins.push_back(&graph_.tensor(t));
        for (int t : n.output_tids) outs.push_back(&graph_.tensor(t));
        n.op->compute(ins, outs, ctx);
    }
}

void Executor::capture_cuda_graph() {
    // 仅捕获纯计算部分的 Kernel Launch；H2D/D2H 拷贝在 Graph 外部，
    // 通过固定显存地址（dev_input_ / dev_output_）与 Graph 交互。
    TINYINFER_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    run_once_on_stream();
    TINYINFER_CUDA_CHECK(cudaStreamEndCapture(stream_, &cuda_graph_));
    TINYINFER_CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, NULL, NULL, 0));
    graph_captured_ = true;
}

void Executor::run() {
    if (opt_.use_cuda_graph) {
        TINYINFER_CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec_, stream_));
    } else {
        run_once_on_stream();
    }
    TINYINFER_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

BenchStats Executor::benchmark(int warmup, int iters) {
    // 预热（不在计时范围内）
    for (int i = 0; i < warmup; ++i) run();

    cudaEvent_t start, stop;
    TINYINFER_CUDA_CHECK(cudaEventCreate(&start));
    TINYINFER_CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples(iters);
    for (int i = 0; i < iters; ++i) {
        // cudaEvent 记录 GPU 端时间；H2D/D2H 在 set_input/get_output 中，不计入
        TINYINFER_CUDA_CHECK(cudaEventRecord(start, stream_));
        if (opt_.use_cuda_graph) {
            TINYINFER_CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec_, stream_));
        } else {
            run_once_on_stream();
        }
        TINYINFER_CUDA_CHECK(cudaEventRecord(stop, stream_));
        TINYINFER_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        TINYINFER_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples[i] = ms * 1000.0f;  // -> 微秒
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::sort(samples.begin(), samples.end());
    BenchStats s;
    s.iters = iters;
    s.min_us = samples.front();
    s.p50_us = samples[(size_t)(iters * 0.50)];
    s.p99_us = samples[std::min((size_t)(iters * 0.99), (size_t)(iters - 1))];
    double sum = 0.0;
    for (float v : samples) sum += v;
    s.mean_us = sum / iters;
    double sq = 0.0;
    for (float v : samples) sq += ((double)v - s.mean_us) * ((double)v - s.mean_us);
    s.stddev_us = std::sqrt(sq / iters);
    return s;
}

}  // namespace tinyinfer
