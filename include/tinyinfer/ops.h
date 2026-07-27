#pragma once
// 具体算子定义：Linear(GEMM) / BiasAdd / ReLU / Softmax / FusedGemmReLU
#include "op.h"
#include <vector>

namespace tinyinfer {

// ---------- Linear (GEMM): Y = X @ W^T + b ----------
// 权重 W 形状 [out_features, in_features]，按行主序存储。
// 支持可选偏置 b [out_features]。
// 当融合模式开启时，ReLU 在 epilogue 阶段就地完成（见 FusedLinearReLU）。
class LinearOp : public Op {
public:
    LinearOp(int in_features, int out_features,
             const dtype* W, const dtype* b)
        : in_f_(in_features), out_f_(out_features), W_(W), b_(b) {}

    const char* type() const override { return "Linear"; }
    void compute(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs,
                 ExecContext& ctx) override;

private:
    int in_f_, out_f_;
    const dtype* W_;   // device 指针（权重，规划在常量区/独立 buffer）
    const dtype* b_;   // device 指针（偏置，可为 nullptr）
};

// ---------- CublasLinear: cuBLAS SGEMM 基线 ----------
// 外部参照实现（任务 2.4）：库函数手动编排，GEMM 由 cuBLAS 完成，
// 偏置由独立 BiasAdd kernel 完成，不参与融合。用作性能对比的公平基线。
class CublasLinearOp : public Op {
public:
    CublasLinearOp(int in_features, int out_features,
                   const dtype* W, const dtype* b)
        : in_f_(in_features), out_f_(out_features), W_(W), b_(b) {}
    const char* type() const override { return "CublasLinear"; }
    void compute(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs,
                 ExecContext& ctx) override;
private:
    int in_f_, out_f_;
    const dtype* W_;
    const dtype* b_;
};

// ---------- BiasAdd: Y = X + b （按特征维广播） ----------
class BiasAddOp : public Op {
public:
    BiasAddOp(int features, const dtype* b) : features_(features), b_(b) {}
    const char* type() const override { return "BiasAdd"; }
    void compute(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs,
                 ExecContext& ctx) override;
private:
    int features_;
    const dtype* b_;
};

// ---------- ReLU ----------
class ReLUOp : public Op {
public:
    ReLUOp() = default;
    const char* type() const override { return "ReLU"; }
    bool fusable() const override { return true; }  // 可被 Linear 融合
    void compute(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs,
                 ExecContext& ctx) override;
};

// ---------- Softmax（按 dim=1 行归一化） ----------
class SoftmaxOp : public Op {
public:
    explicit SoftmaxOp(int dim = 1) : dim_(dim) {}
    const char* type() const override { return "Softmax"; }
    void compute(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs,
                 ExecContext& ctx) override;
private:
    int dim_;
};

// ---------- FusedGemmReLU: Y = max(0, X @ W^T + b) ----------
// 融合算子：在 GEMM 的 epilogue 阶段，C 矩阵仍在寄存器/Shared Memory 中时
// 就地做 ReLU，只写回一次 Global Memory，减少一次读写与一次 kernel launch。
class FusedLinearReLUOp : public Op {
public:
    FusedLinearReLUOp(int in_features, int out_features,
                      const dtype* W, const dtype* b)
        : in_f_(in_features), out_f_(out_features), W_(W), b_(b) {}
    const char* type() const override { return "FusedLinearReLU"; }
    void compute(const std::vector<Tensor*>& inputs,
                 const std::vector<Tensor*>& outputs,
                 ExecContext& ctx) override;
private:
    int in_f_, out_f_;
    const dtype* W_;
    const dtype* b_;
};

}  // namespace tinyinfer
