#pragma once
// 模型加载：解析 JSON 模型描述 + 加载权重二进制文件
#include "common.h"
#include "graph.h"
#include <string>
#include <unordered_map>

namespace tinyinfer {

// 一层（layer）的 JSON 配置
struct LayerCfg {
    std::string type;                 // Linear / ReLU / Softmax ...
    std::unordered_map<std::string, std::string> params;
};

// 加载后的模型：计算图 + 权重（device 指针）
struct Model {
    std::string name;
    std::vector<int> input_shape;
    std::vector<LayerCfg> layers;
    // 权重名 -> device 指针（在 build_graph 时已 cudaMalloc + 拷贝）
    std::unordered_map<std::string, dtype*> weights;
    std::unordered_map<std::string, std::string> weight_files;  // 权重名 -> 权重文件路径
};

// 解析 JSON 文件，构建 Model 描述（不分配 device 显存）
Model load_model_json(const std::string& json_path);

// 将 Model 中的权重文件加载到 device 显存
void load_weights_to_device(Model& model);

// 根据 Model 构建计算图（含算子实例、边、融合优化）
// 返回 graph 与输入/输出 tensor id 信息
// fuse_gemm_relu：自动识别 Linear+ReLU 模式替换为融合算子
// use_cublas：Linear 使用 cuBLAS 基线实现（外部参照，此时融合自动关闭）
// 注意：分配的权重 device 指针将存入 model.weights，需调用 free_model_weights() 释放
void build_graph(Model& model, ComputeGraph& graph,
                 int& input_tid, int& output_tid,
                 bool fuse_gemm_relu = true,
                 bool use_cublas = false);

// 释放 build_graph 中分配的权重 device 显存
void free_model_weights(Model& model);

}  // namespace tinyinfer
