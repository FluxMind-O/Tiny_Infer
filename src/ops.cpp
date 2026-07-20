// 具体算子的 compute 实现 + 算子注册（工厂模式）
#include "tinyinfer/ops.h"
#include "tinyinfer/common.h"
#include "tinyinfer/registry.h"
#include "tinyinfer/kernels_launcher.h"
#include <stdexcept>

namespace tinynfer {

void LinearOp::compute(const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs,
                       ExecContext& ctx) {
    int M = inputs[0]->shape[0];
    int K = in_f_;
    int N = out_f_;
    kernels::launch_gemm(inputs[0]->data, W_, b_, outputs[0]->data,
                         M, K, N, /*apply_relu=*/false, ctx.stream);
}

void BiasAddOp::compute(const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs,
                        ExecContext& ctx) {
    int M = inputs[0]->shape[0];
    int N = features_;
    kernels::launch_bias_add(inputs[0]->data, b_, outputs[0]->data, M, N, ctx.stream);
}

void ReLUOp::compute(const std::vector<Tensor*>& inputs,
                     const std::vector<Tensor*>& outputs,
                     ExecContext& ctx) {
    int n = inputs[0]->num_elements();
    kernels::launch_relu(inputs[0]->data, outputs[0]->data, n, ctx.stream);
}

void SoftmaxOp::compute(const std::vector<Tensor*>& inputs,
                        const std::vector<Tensor*>& outputs,
                        ExecContext& ctx) {
    int M = inputs[0]->shape[0];
    int N = inputs[0]->num_elements() / M;
    kernels::launch_softmax(inputs[0]->data, outputs[0]->data, M, N, ctx.stream);
}

void FusedLinearReLUOp::compute(const std::vector<Tensor*>& inputs,
                                const std::vector<Tensor*>& outputs,
                                ExecContext& ctx) {
    int M = inputs[0]->shape[0];
    int K = in_f_;
    int N = out_f_;
    // 关键：GEMM + ReLU 合并为一次 kernel、一次 Global Memory 写回
    kernels::launch_fused_gemm_relu(inputs[0]->data, W_, b_, outputs[0]->data,
                                    M, K, N, ctx.stream);
}

// ---------------- 算子注册（工厂模式） ----------------
// 注册表负责按 JSON "type" 创建算子实例，权重已以文件名 -> device 指针形式提供。
static const dtype* wptr(const std::unordered_map<std::string, const dtype*>& w,
                         const std::string& key) {
    auto it = w.find(key);
    return it == w.end() ? nullptr : it->second;
}

TINYINFER_REGISTER_OP(Linear, [](const std::unordered_map<std::string, std::string>& cfg,
                                 const std::unordered_map<std::string, const dtype*>& w) {
    int in_f = std::stoi(cfg.at("in_features"));
    int out_f = std::stoi(cfg.at("out_features"));
    const dtype* W = wptr(w, cfg.at("weight"));
    const dtype* B = cfg.count("bias") ? wptr(w, cfg.at("bias")) : nullptr;
    return std::unique_ptr<Op>(new LinearOp(in_f, out_f, W, B));
});

TINYINFER_REGISTER_OP(ReLU, [](const std::unordered_map<std::string, std::string>&,
                               const std::unordered_map<std::string, const dtype*>&) {
    return std::unique_ptr<Op>(new ReLUOp());
});

TINYINFER_REGISTER_OP(Softmax, [](const std::unordered_map<std::string, std::string>& cfg,
                                  const std::unordered_map<std::string, const dtype*>&) {
    int dim = cfg.count("dim") ? std::stoi(cfg.at("dim")) : 1;
    return std::unique_ptr<Op>(new SoftmaxOp(dim));
});

TINYINFER_REGISTER_OP(FusedLinearReLU,
    [](const std::unordered_map<std::string, std::string>& cfg,
       const std::unordered_map<std::string, const dtype*>& w) {
        int in_f = std::stoi(cfg.at("in_features"));
        int out_f = std::stoi(cfg.at("out_features"));
        const dtype* W = wptr(w, cfg.at("weight"));
        const dtype* B = cfg.count("bias") ? wptr(w, cfg.at("bias")) : nullptr;
        return std::unique_ptr<Op>(new FusedLinearReLUOp(in_f, out_f, W, B));
    });

}  // namespace tinynfer
