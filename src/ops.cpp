// 具体算子的 compute 实现 + 算子注册（工厂模式）
#include "tinyinfer/ops.h"
#include "tinyinfer/common.h"
#include "tinyinfer/registry.h"
#include "tinyinfer/kernels_launcher.h"
#include <cublas_v2.h>
#include <stdexcept>

namespace tinyinfer {

// cuBLAS 句柄单例（RAII 获取即初始化；进程级泄漏以避免退出时的析构顺序问题）。
// 预分配 workspace：避免 CUDA Graph 捕获期间 cuBLAS 触发内部 cudaMalloc。
static cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS: cublasCreate failed");
        void* workspace = nullptr;
        size_t ws_size = 4 * 1024 * 1024;  // 4MB 预分配工作区
        TINYINFER_CUDA_CHECK(cudaMalloc(&workspace, ws_size));
        if (cublasSetWorkspace(h, workspace, ws_size) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS: cublasSetWorkspace failed");
        return h;
    }();
    return handle;
}

void CublasLinearOp::compute(const std::vector<Tensor*>& inputs,
                             const std::vector<Tensor*>& outputs,
                             ExecContext& ctx) {
    int M = inputs[0]->shape[0];
    int K = in_f_;
    int N = out_f_;
    cublasHandle_t h = cublas_handle();
    cublasSetStream(h, ctx.stream);
    const float alpha = 1.0f, beta = 0.0f;
    // 行主序 C[M,N] = X[M,K] @ W[N,K]^T 的列主序等价形式：
    // C^T = W @ X^T。W 行主序 [N,K] 等价于列主序 W^T [K,N]，
    // 故对 W 取 OP_T 还原为数学矩阵 [N,K]；X 行主序 [M,K] 等价于
    // 列主序 X^T [K,M]，直接 OP_N。按 (m=N, n=M, k=K) 调用。
    cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                                    N, M, K, &alpha,
                                    W_, K, inputs[0]->data, K,
                                    &beta, outputs[0]->data, N);
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS: cublasSgemm failed");
    // 偏置单独一个 kernel（cuBLAS 基线不做 epilogue 融合）
    if (b_) kernels::launch_bias_add(outputs[0]->data, b_, outputs[0]->data,
                                     M, N, ctx.stream);
}


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

TINYINFER_REGISTER_OP(CublasLinear,
    [](const std::unordered_map<std::string, std::string>& cfg,
       const std::unordered_map<std::string, const dtype*>& w) {
        int in_f = std::stoi(cfg.at("in_features"));
        int out_f = std::stoi(cfg.at("out_features"));
        const dtype* W = wptr(w, cfg.at("weight"));
        const dtype* B = cfg.count("bias") ? wptr(w, cfg.at("bias")) : nullptr;
        return std::unique_ptr<Op>(new CublasLinearOp(in_f, out_f, W, B));
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

}  // namespace tinyinfer
