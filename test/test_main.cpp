// TinyInfer 统一测试入口（阶段四 任务4.1）
// 三类测试：
//   1. 数值正确性：全部消融配置 + cuBLAS 基线 vs numpy 参考（max abs err < 1e-3）
//      测试矩阵覆盖 batch ∈ {1, 4, 8}
//   2. 显存对比：静态规划（复用）vs 独立分配的显存峰值 / buffer 数量
//   3. 性能基准：四组消融配置 + 「融合×Graph」2×2 解耦 + cuBLAS 基线
//      统计口径：预热 100 次 + 测量 1000 次，cudaEvent GPU 端计时（不含 H2D/D2H）
#include "tinyinfer/common.h"
#include "tinyinfer/model.h"
#include "tinyinfer/executor.h"
#include "tinyinfer/graph.h"
#include "tinyinfer/memory.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace tinyinfer;

static int g_failures = 0;

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

static void check(bool ok, const std::string& name) {
    if (ok) {
        std::cout << "  [PASS] " << name << "\n";
    } else {
        std::cerr << "  [FAIL] " << name << "\n";
        g_failures++;
    }
}

// 一组配置的推理上下文
struct RunCtx {
    Model model;
    ComputeGraph graph;
    int in_tid = -1, out_tid = -1;
    std::unique_ptr<Executor> ex;

    ~RunCtx() {
        ex.reset();  // 先析构 Executor（释放统一 buffer），再释放权重
        free_model_weights(model);
    }
};

// 构建一组配置并验证数值正确性，返回 max abs error
static double run_and_verify(const std::string& root, const std::string& model_name,
                             int batch, const ExecOptions& opt, bool use_cublas) {
    RunCtx ctx;
    ctx.model = load_model_json(root + "/models/" + model_name + ".json");
    ctx.model.input_shape[0] = batch;
    build_graph(ctx.model, ctx.graph, ctx.in_tid, ctx.out_tid,
                opt.fuse_gemm_relu, use_cublas);
    ctx.ex.reset(new Executor(ctx.graph, opt));
    ctx.ex->set_io(ctx.in_tid, ctx.out_tid);
    ctx.ex->prepare();

    auto input = load_bin(root + "/data/" + model_name + "_b" +
                          std::to_string(batch) + ".bin");
    ctx.ex->set_input(input.data(), batch);
    ctx.ex->run();
    std::vector<dtype> out;
    ctx.ex->get_output(out);

    auto ref = load_bin(root + "/data/" + model_name + "_b" +
                        std::to_string(batch) + "_ref.bin");
    if (out.size() != ref.size()) return 1e9;
    double max_err = 0.0;
    for (size_t i = 0; i < out.size(); ++i)
        max_err = std::max(max_err, (double)std::fabs(out[i] - ref[i]));
    return max_err;
}

// 性能基准：为指定配置构建独立实例并测量
static BenchStats bench_config(const std::string& root, const std::string& model_name,
                               int batch, const ExecOptions& opt, bool use_cublas) {
    RunCtx ctx;
    ctx.model = load_model_json(root + "/models/" + model_name + ".json");
    ctx.model.input_shape[0] = batch;
    build_graph(ctx.model, ctx.graph, ctx.in_tid, ctx.out_tid,
                opt.fuse_gemm_relu, use_cublas);
    ctx.ex.reset(new Executor(ctx.graph, opt));
    ctx.ex->set_io(ctx.in_tid, ctx.out_tid);
    ctx.ex->prepare();
    auto input = load_bin(root + "/data/" + model_name + "_b" +
                          std::to_string(batch) + ".bin");
    ctx.ex->set_input(input.data(), batch);
    return ctx.ex->benchmark(/*warmup=*/100, /*iters=*/1000);
}

// 统计一组配置下推理链的 kernel launch 次数（理论值：每个节点 1 次，
// cuBLAS Linear 额外带 1 次 BiasAdd）
static int count_kernels(const std::string& root, const std::string& model_name,
                         const ExecOptions& opt, bool use_cublas) {
    Model model = load_model_json(root + "/models/" + model_name + ".json");
    ComputeGraph graph;
    int in_tid, out_tid;
    build_graph(model, graph, in_tid, out_tid, opt.fuse_gemm_relu, use_cublas);
    int kernels = 0;
    for (int i = 0; i < graph.num_nodes(); ++i) {
        kernels += 1;
        if (use_cublas && graph.node(i).op_type == "CublasLinear") kernels += 1;
    }
    return kernels;
}

int main(int argc, char** argv) {
    std::string root = (argc > 1) ? argv[1] : ".";
    const std::string model_name = "mlp_large";  // 计划 6.1 节基准模型
    const std::vector<int> batches = {1, 4, 8};  // 计划验收测试矩阵

    try {
        // ============ 1. 数值正确性（全部配置 × batch 矩阵） ============
        std::cout << "== 1. Numerical correctness (vs numpy ref, threshold 1e-3) ==\n";
        struct Cfg { const char* name; ExecOptions opt; bool cublas; };
        ExecOptions c0{false, false, false}, c1{true, false, false},
                    c2{true, true, false}, c3{true, true, true};
        std::vector<Cfg> cfgs = {
            {"Config0(no opt)", c0, false},
            {"Config1(fuse)", c1, false},
            {"Config2(fuse+reuse)", c2, false},
            {"Config3(fuse+reuse+graph)", c3, false},
            {"cuBLAS baseline", ExecOptions{false, true, false}, true},
        };
        for (auto& c : cfgs) {
            for (int b : batches) {
                double err = run_and_verify(root, model_name, b, c.opt, c.cublas);
                std::cout << "  " << c.name << " batch=" << b
                          << "  max_err=" << err << "\n";
                check(err < 1e-3, std::string(c.name) + " batch=" + std::to_string(b));
            }
        }

        // ============ 2. 显存规划对比 ============
        std::cout << "\n== 2. Memory planning (reuse vs independent) ==\n";
        for (int b : batches) {
            Model model = load_model_json(root + "/models/" + model_name + ".json");
            model.input_shape[0] = b;
            ComputeGraph graph;
            int in_tid, out_tid;
            build_graph(model, graph, in_tid, out_tid, true);
            auto order = graph.topo_order();
            graph.analyze_lifetimes(order);

            MemoryPlanner planner;
            size_t reuse_bytes = planner.plan(graph, order, true);
            std::set<int> reuse_offsets;
            int n_tensors = 0;
            for (int i = 0; i < graph.num_tensors(); ++i)
                if (!graph.tensor(i).fixed && graph.tensor(i).size > 0) {
                    reuse_offsets.insert(graph.tensor(i).offset);
                    n_tensors++;
                }
            size_t indep_bytes = planner.plan(graph, order, false);

            double ratio = indep_bytes ? 100.0 * reuse_bytes / indep_bytes : 0.0;
            printf("  batch=%d: intermediate tensors=%d, reused buffers=%zu, "
                   "peak %.1f KB -> %.1f KB (%.1f%%)\n",
                   b, n_tensors, reuse_offsets.size(),
                   indep_bytes / 1024.0, reuse_bytes / 1024.0, ratio);
            check(reuse_bytes < indep_bytes,
                  "memory reuse reduces peak, batch=" + std::to_string(b));
        }

        // ============ 3. 性能基准（消融 + 2x2 解耦 + cuBLAS 基线） ============
        std::cout << "\n== 3. Benchmark (warmup 100 + 1000 iters, cudaEvent, "
                     "H2D/D2H excluded) ==\n";
        for (int b : batches) {
            printf("\n  -- batch=%d --\n", b);
            printf("  %-26s %8s %8s %8s %8s %8s %8s\n",
                   "config", "kernels", "mean", "std", "P50", "P99", "min");
            double prev = 0.0;
            bool first = true;
            for (auto& c : cfgs) {
                BenchStats s = bench_config(root, model_name, b, c.opt, c.cublas);
                int nk = count_kernels(root, model_name, c.opt, c.cublas);
                printf("  %-26s %8d %8.2f %8.2f %8.2f %8.2f %8.2f",
                       c.name, nk, s.mean_us, s.stddev_us, s.p50_us,
                       s.p99_us, s.min_us);
                if (!first) {
                    double delta = s.mean_us - prev;
                    // 显著性判据：增量 < 2*stddev 标注「差异不显著」
                    if (std::fabs(delta) < 2.0 * s.stddev_us)
                        printf("  (vs prev: %+.2f us, 差异不显著)", delta);
                    else
                        printf("  (vs prev: %+.2f us)", delta);
                }
                printf("\n");
                prev = s.mean_us;
                first = false;
            }

            // 融合 × Graph 2×2 解耦实验（reuse 固定关闭，隔离交互项）
            ExecOptions ff{false, false, false}, tf{true, false, false},
                          ft{false, false, true}, tt{true, false, true};
            double t_ff = bench_config(root, model_name, b, ff, false).mean_us;
            double t_tf = bench_config(root, model_name, b, tf, false).mean_us;
            double t_ft = bench_config(root, model_name, b, ft, false).mean_us;
            double t_tt = bench_config(root, model_name, b, tt, false).mean_us;
            printf("  [2x2 decouple] fuse\\graph: off/off=%.2f on/off=%.2f "
                   "off/on=%.2f on/on=%.2f us\n", t_ff, t_tf, t_ft, t_tt);
            printf("    fuse alone saves %.2f us; fuse marginal on graph saves "
                   "%.2f us (交互项 %.2f us)\n",
                   t_ff - t_tf, t_ft - t_tt,
                   (t_ff - t_tf) - (t_ft - t_tt));
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (g_failures == 0) {
        std::cout << "\nALL TESTS PASSED\n";
        return 0;
    }
    std::cerr << "\n" << g_failures << " TEST(S) FAILED\n";
    return 1;
}
