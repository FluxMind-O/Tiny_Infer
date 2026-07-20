// 模型加载实现
#include "tinyinfer/model.h"
#include "tinyinfer/json_min.h"
#include "tinyinfer/registry.h"
#include "tinyinfer/ops.h"
#include "tinyinfer/common.h"
#include <fstream>
#include <sstream>
#include <cstring>

namespace tinynfer {

namespace json {
ValuePtr parse_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Model: cannot open JSON '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();
    Parser p(content);
    return p.parse();
}
}  // namespace json

static std::vector<int> parse_int_array(const json::ValuePtr& v) {
    std::vector<int> out;
    for (auto& e : v->arr) out.push_back(e->as_int());
    return out;
}

static bool is_absolute_path(const std::string& path) {
    if (path.empty()) return false;
    if (path[0] == '/' || path[0] == '\\') return true;
    return path.size() >= 3 &&
           path[1] == ':' &&
           (path[2] == '/' || path[2] == '\\');
}

static std::string dirname_of(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static std::string join_path(const std::string& base, const std::string& child) {
    if (base.empty() || base == ".") return child;
    char last = base.back();
    if (last == '/' || last == '\\') return base + child;
    return base + "/" + child;
}

static dtype* load_weight_bin(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Model: cannot open weight file '" + path + "'");
    dtype* d = nullptr;
    TINYINFER_CUDA_CHECK(cudaMalloc(&d, n * sizeof(dtype)));
    std::vector<dtype> h(n);
    f.read(reinterpret_cast<char*>(h.data()), (std::streamsize)(n * sizeof(dtype)));
    if ((size_t)f.gcount() != n * sizeof(dtype))
        throw std::runtime_error("Model: weight file size mismatch '" + path + "'");
    TINYINFER_CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(dtype),
                                    cudaMemcpyHostToDevice));
    return d;
}

Model load_model_json(const std::string& json_path) {
    auto root = json::parse_file(json_path);
    std::string model_dir = dirname_of(json_path);
    Model m;
    m.name = root->at("name")->as_string();
    m.input_shape = parse_int_array(root->at("input_shape"));

    for (auto& l : root->at("layers")->arr) {
        LayerCfg cfg;
        cfg.type = l->at("type")->as_string();
        for (auto& kv : l->obj) {
            if (kv.first == "type") continue;
            if (kv.second->type == json::Value::String)
                cfg.params[kv.first] = kv.second->as_string();
            else if (kv.second->type == json::Value::Number)
                cfg.params[kv.first] = std::to_string(kv.second->as_int());
            else if (kv.second->type == json::Value::Bool)
                cfg.params[kv.first] = kv.second->as_bool() ? "true" : "false";
        }
        m.layers.push_back(std::move(cfg));
    }

    if (root->has("weights")) {
        for (auto& kv : root->at("weights")->obj) {
            std::string weight_path = kv.second->as_string();
            if (!is_absolute_path(weight_path))
                weight_path = join_path(model_dir, weight_path);
            m.weight_files[kv.first] = weight_path;
        }
    }
    return m;
}

void load_weights_to_device(Model& model) {
    // 在 build_graph 中按 layer 尺寸直接加载，这里为空（保留接口）。
    (void)model;
}

void build_graph(Model& model, ComputeGraph& graph,
                 int& input_tid, int& output_tid, bool fuse_gemm_relu) {
    input_tid = graph.add_tensor(model.input_shape);
    graph.tensor(input_tid).fixed = true;

    auto alloc_by_layer = [&](size_t n, const std::string& key) -> const dtype* {
        auto it = model.weights.find(key);
        if (it != model.weights.end())
            return it->second;
        std::string filepath;
        auto wf_it = model.weight_files.find(key);
        if (wf_it != model.weight_files.end())
            filepath = wf_it->second;
        else
            filepath = key;
        dtype* ptr = load_weight_bin(filepath, n);
        model.weights[key] = ptr;
        return ptr;
    };

    int prev_tid = input_tid;
    for (size_t layer_idx = 0; layer_idx < model.layers.size(); ++layer_idx) {
        const auto& layer = model.layers[layer_idx];
        const std::string& type = layer.type;
        if (type == "Linear") {
            int in_f = std::stoi(layer.params.at("in_features"));
            int out_f = std::stoi(layer.params.at("out_features"));
            std::string wkey = "layer" + std::to_string(layer_idx) + "_weight";
            std::string bkey = "layer" + std::to_string(layer_idx) + "_bias";
            if (!model.weight_files.count(wkey))
                throw std::runtime_error("Model: missing weight entry '" + wkey + "'");
            const dtype* W = alloc_by_layer((size_t)in_f * out_f, wkey);
            const dtype* B = model.weight_files.count(bkey)
                ? alloc_by_layer((size_t)out_f, bkey)
                : nullptr;

            int out_tid = graph.add_tensor({model.input_shape[0], out_f});
            bool can_fuse = fuse_gemm_relu && (layer_idx + 1 < model.layers.size()) &&
                            model.layers[layer_idx + 1].type == "ReLU";

            if (can_fuse) {
                std::unordered_map<std::string, std::string> cfg = {
                    {"in_features", std::to_string(in_f)},
                    {"out_features", std::to_string(out_f)},
                    {"weight", wkey},
                };
                if (B) cfg["bias"] = bkey;
                std::unordered_map<std::string, const dtype*> wmap;
                wmap[wkey] = W;
                if (B) wmap[bkey] = B;
                auto op = OpRegistry::instance().create("FusedLinearReLU", cfg, wmap);
                graph.add_node(std::move(op), "FusedLinearReLU", {prev_tid}, {out_tid});
                prev_tid = out_tid;
                ++layer_idx;
                continue;
            } else {
                std::unordered_map<std::string, std::string> cfg = {
                    {"in_features", std::to_string(in_f)},
                    {"out_features", std::to_string(out_f)},
                    {"weight", wkey},
                };
                if (B) cfg["bias"] = bkey;
                std::unordered_map<std::string, const dtype*> wmap;
                wmap[wkey] = W;
                if (B) wmap[bkey] = B;
                auto op = OpRegistry::instance().create("Linear", cfg, wmap);
                graph.add_node(std::move(op), "Linear", {prev_tid}, {out_tid});
            }
            prev_tid = out_tid;
        } else if (type == "ReLU") {
            int out_tid = graph.add_tensor(graph.tensor(prev_tid).shape);
            std::unordered_map<std::string, const dtype*> empty_wmap;
            auto op = OpRegistry::instance().create("ReLU", {}, empty_wmap);
            graph.add_node(std::move(op), "ReLU", {prev_tid}, {out_tid});
            prev_tid = out_tid;
        } else if (type == "Softmax") {
            int out_tid = graph.add_tensor(graph.tensor(prev_tid).shape);
            std::unordered_map<std::string, std::string> cfg;
            if (layer.params.count("dim")) cfg["dim"] = layer.params.at("dim");
            std::unordered_map<std::string, const dtype*> empty_wmap;
            auto op = OpRegistry::instance().create("Softmax", cfg, empty_wmap);
            graph.add_node(std::move(op), "Softmax", {prev_tid}, {out_tid});
            prev_tid = out_tid;
        } else {
            throw std::runtime_error("Model: unsupported layer type '" + type + "'");
        }
    }
    output_tid = prev_tid;
    graph.tensor(output_tid).fixed = true;
}

void free_model_weights(Model& model) {
    for (auto& kv : model.weights) {
        if (kv.second) {
            cudaFree((void*)kv.second);
            kv.second = nullptr;
        }
    }
    model.weights.clear();
}

} // namespace tinynfer
