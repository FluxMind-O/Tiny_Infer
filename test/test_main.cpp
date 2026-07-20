// TinyInfer 测试：验证前向推理数值正确性 + 显存规划 + 性能对比
#include "tinyinfer/common.h"
#include "tinyinfer/model.h"
#include "tinyinfer/executor.h"
#include "tinyinfer/graph.h"
#include "tinyinfer/memory.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

using namespace tinynfer;

static std::vector<dtype> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("test: cannot open " + path);
    f.seekg(0, std::ios::end);
    size_t sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<dtype> v(sz / sizeof(dtype));
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)sz);
    return v;
}

int main(int argc, char** argv) {
    std::string root = (argc > 1) ? argv[1] : ".";
    std::string model_json = root + "/models/mlp3.json";
    std::string sample = root + "/data/sample.bin";
    std::string ref = root + "/data/ref_output.bin";

    int failures = 0;
    try {
        Model model = load_model_json(model_json);
        ComputeGraph graph;
        int in_tid, out_tid;
        build_graph(model, graph, in_tid, out_tid, /*fuse=*/true);
        std::cout << "[test] tensors=" << graph.num_tensors()
                  << " nodes=" << graph.num_nodes() << "\n";

        // ---- 静态显存规划对比 ----
        auto order = graph.topo_order();
        graph.analyze_lifetimes(order);
        MemoryPlanner planner;
        size_t reuse_bytes = planner.plan(graph, order, /*reuse=*/true);
        size_t indep_bytes = planner.plan(graph, order, /*reuse=*/false);
        std::cout << "[test] intermediate buffer: reuse=" << (reuse_bytes / 1024)
                  << " KB, independent=" << (indep_bytes / 1024) << " KB\n";
        if (indep_bytes > 0 && (double)reuse_bytes > (double)indep_bytes * 0.95) {
            std::cerr << "[FAIL] memory reuse too weak\n";
            failures++;
        }

        // ---- 前向推理（融合 + CUDA Graph）----
        ExecOptions opt;
        opt.use_async_stream = true;
        opt.use_cuda_graph = true;
        opt.fuse_gemm_relu = true;
        Executor ex(graph, opt);
        ex.set_io(in_tid, out_tid);
        ex.prepare();

        auto input = load_bin(sample);
        ex.set_input(input.data(), 1);
        ex.run();
        std::vector<dtype> out;
        ex.get_output(out);

        auto ref_out = load_bin(ref);
        if (out.size() != ref_out.size()) {
            std::cerr << "[FAIL] output size mismatch\n";
            failures++;
        } else {
            double max_err = 0.0;
            for (size_t i = 0; i < out.size(); ++i)
                max_err = std::max(max_err, (double)std::fabs(out[i] - ref_out[i]));
            std::cout << "[test] max abs error vs numpy ref: " << max_err << "\n";
            if (max_err > 1e-3) {
                std::cerr << "[FAIL] numerical mismatch (max_err=" << max_err << ")\n";
                failures++;
            } else {
                std::cout << "[PASS] fused+cuda_graph matches reference\n";
            }
        }

        // ---- 性能基准：融合 vs 无融合 ----
        ExecOptions opt2 = opt;
        opt2.fuse_gemm_relu = false;
        opt2.use_cuda_graph = false;
        ComputeGraph graph2;
        int in2, out2;
        build_graph(model, graph2, in2, out2, false);
        Executor ex2(graph2, opt2);
        ex2.set_io(in2, out2);
        ex2.prepare();
        auto input2 = load_bin(sample);
        ex2.set_input(input2.data(), 1);
        double lat_fused = ex.benchmark(50);
        double lat_nofuse = ex2.benchmark(50);
        std::cout << "[bench] fused+cuda_graph: " << lat_fused << " us\n";
        std::cout << "[bench] unoptimized:      " << lat_nofuse << " us\n";

        free_model_weights(model);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (failures == 0) {
        std::cout << "\nALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << "\n" << failures << " TEST(S) FAILED\n";
    return 1;
}
