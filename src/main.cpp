// TinyInfer CLI 入口
// 用法:
//   ./build/tinyinfer --model models/mlp3.json --input data/sample.bin
//   ./build/tinyinfer --model models/mlp3.json --input input.txt
//   ./build/tinyinfer --model models/mlp3.json --input data/batch4.bin --batch 4
//   ./build/tinyinfer --model models/mlp3.json --input input.txt --bench 100
#include "tinyinfer/common.h"
#include "tinyinfer/model.h"
#include "tinyinfer/executor.h"
#include "tinyinfer/graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace tinynfer;

static std::vector<dtype> read_input(const std::string& path, int expected) {
    std::vector<dtype> data;
    // 尝试按二进制读（.bin）
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
        // 文本：逗号/空格分隔
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

int main(int argc, char** argv) {
    std::string model_path, input_path, output_path;
    int batch = 1, bench = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (a == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--batch" && i + 1 < argc) batch = std::stoi(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) bench = std::stoi(argv[++i]);
    }
    if (model_path.empty() || input_path.empty()) {
        std::cerr << "Usage: tinyinfer --model <json> --input <bin|txt>"
                     " [--output <bin>] [--batch N] [--bench K]\n";
        return 1;
    }

    try {
        if (batch <= 0) {
            std::cerr << "--batch must be positive\n";
            return 1;
        }

        Model model = load_model_json(model_path);
        if (model.input_shape.empty()) {
            std::cerr << "model input_shape is empty\n";
            return 1;
        }
        model.input_shape[0] = batch;
        std::cout << "[TinyInfer] model: " << model.name << "\n";

        ComputeGraph graph;
        int in_tid = -1, out_tid = -1;
        build_graph(model, graph, in_tid, out_tid, /*fuse=*/true);

        ExecOptions opt;
        opt.use_async_stream = true;
        opt.use_cuda_graph = true;
        opt.fuse_gemm_relu = true;

        Executor ex(graph, opt);
        ex.set_io(in_tid, out_tid);
        ex.prepare();

        int in_n = graph.tensor(in_tid).num_elements();
        auto input = read_input(input_path, in_n);
        ex.set_input(input.data(), batch);

        if (bench > 0) {
            double lat = ex.benchmark(bench);
            std::cout << "[TinyInfer] avg latency over " << bench
                      << " runs: " << lat << " us\n";
        }

        ex.run();

        std::vector<dtype> output;
        ex.get_output(output);

        // 打印结果（前若干）
        print_tensor(output.data(), (int)output.size(), "output", 10);

        if (!output_path.empty()) write_output_bin(output_path, output);
        std::cout << "[TinyInfer] planned intermediate buffer: "
                  << (ex.planned_bytes() / (1024 * 1024)) << " MB\n";

        free_model_weights(model);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
