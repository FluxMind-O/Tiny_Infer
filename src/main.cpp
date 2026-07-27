// TinyInfer CLI 入口
// 用法:
//   ./build/tinyinfer --model models/mlp_large.json --input data/mlp_large_b1.bin --batch 1
//   ./build/tinyinfer --model models/mlp3.json --input data/sample.bin --bench
//   ./build/tinyinfer --model models/mlp3.json --input input.txt --no-fuse --no-graph
// 开关:
//   --no-fuse   关闭 Linear+ReLU 融合
//   --no-reuse  关闭静态显存复用（每个中间 tensor 独立分配）
//   --no-graph  关闭 CUDA Graph 固化
//   --cublas    Linear 使用 cuBLAS 基线实现（外部参照，自动关闭融合）
#include "tinyinfer/common.h"
#include "tinyinfer/model.h"
#include "tinyinfer/executor.h"
#include "tinyinfer/graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tinyinfer;

static std::vector<dtype> read_input(const std::string& path, int expected) {
    std::vector<dtype> data;
    if (path.size() > 4 && path.substr(path.size() - 4) == ".bin") {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "cannot open input " << path << "\n"; exit(1); }
        f.seekg(0, std::ios::end);
        size_t sz = (size_t)f.tellg();
        f.seekg(0, std::ios::beg);
        size_t n = sz / sizeof(dtype);
        data.resize(n);
        f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)sz);
    } else {
        std::ifstream f(path);
        if (!f) { std::cerr << "cannot open input " << path << "\n"; exit(1); }
        std::string line;
        while (std::getline(f, line)) {
            for (char& c : line)
                if (c == ',') c = ' ';
            std::stringstream ss(line);
            double value = 0.0;
            while (ss >> value) data.push_back((dtype)value);
        }
    }
    if (expected > 0 && (int)data.size() != expected) {
        std::cerr << "input size mismatch: got " << data.size()
                  << " expected " << expected << "\n";
        exit(1);
    }
    return data;
}

static void write_output_bin(const std::string& path, const std::vector<dtype>& out) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()),
            (std::streamsize)(out.size() * sizeof(dtype)));
}

static void print_usage() {
    std::cerr <<
        "Usage: tinyinfer --model <json> --input <bin|txt> [options]\n"
        "  --output <bin>   将输出写为二进制文件\n"
        "  --batch N        覆盖输入 batch（默认取模型 input_shape[0]）\n"
        "  --bench          性能基准（预热 100 次 + 测量 1000 次，cudaEvent 计时）\n"
        "  --no-fuse        关闭 Linear+ReLU 融合\n"
        "  --no-reuse       关闭静态显存复用\n"
        "  --no-graph       关闭 CUDA Graph\n"
        "  --cublas         使用 cuBLAS 基线 GEMM（自动关闭融合）\n";
}

int main(int argc, char** argv) {
    std::string model_path, input_path, output_path;
    int batch = 0;  // 0 = 取模型默认
    bool bench = false;
    ExecOptions opt;
    bool use_cublas = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--batch" && i + 1 < argc) batch = std::stoi(argv[++i]);
        else if (a == "--bench") bench = true;
        else if (a == "--no-fuse") opt.fuse_gemm_relu = false;
        else if (a == "--no-reuse") opt.reuse_memory = false;
        else if (a == "--no-graph") opt.use_cuda_graph = false;
        else if (a == "--cublas") use_cublas = true;
        else if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else { std::cerr << "unknown arg: " << a << "\n"; print_usage(); return 1; }
    }
    if (model_path.empty() || input_path.empty()) {
        print_usage();
        return 1;
    }

    try {
        Model model = load_model_json(model_path);
        if (model.input_shape.empty()) {
            std::cerr << "model input_shape is empty\n";
            return 1;
        }
        if (batch > 0) model.input_shape[0] = batch;
        TINYINFER_LOG("model: %s (batch=%d)", model.name.c_str(), model.input_shape[0]);

        ComputeGraph graph;
        int in_tid = -1, out_tid = -1;
        build_graph(model, graph, in_tid, out_tid, opt.fuse_gemm_relu, use_cublas);

        Executor ex(graph, opt);
        ex.set_io(in_tid, out_tid);
        ex.prepare();

        int in_n = graph.tensor(in_tid).num_elements();
        auto input = read_input(input_path, in_n);
        ex.set_input(input.data(), model.input_shape[0]);

        TINYINFER_LOG("config: fuse=%d reuse=%d graph=%d cublas=%d",
                      (int)opt.fuse_gemm_relu && !use_cublas,
                      (int)opt.reuse_memory, (int)opt.use_cuda_graph,
                      (int)use_cublas);
        TINYINFER_LOG("planned intermediate buffer: %.2f KB",
                      ex.planned_bytes() / 1024.0);

        if (bench) {
            BenchStats s = ex.benchmark(/*warmup=*/100, /*iters=*/1000);
            printf("[bench] GPU compute latency over %d iters (warmup 100):\n", s.iters);
            printf("[bench]   mean=%.2f us  std=%.2f us  P50=%.2f us  P99=%.2f us  min=%.2f us\n",
                   s.mean_us, s.stddev_us, s.p50_us, s.p99_us, s.min_us);
        }

        ex.run();
        std::vector<dtype> output;
        ex.get_output(output);
        print_tensor(output.data(), (int)output.size(), "output", 10);

        if (!output_path.empty()) write_output_bin(output_path, output);

        free_model_weights(model);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
