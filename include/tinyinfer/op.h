#pragma once
// 算子抽象：所有计算图中的节点（算子实例）都继承 Op
#include "common.h"
#include "tensor.h"
#include <string>

namespace tinyinfer {

// 前向执行上下文：传入 stream、输入/输出 tensor
struct ExecContext {
    cudaStream_t stream;
};

// 算子基类。每个具体算子实现 compute()。
// inputs / outputs 是计算图在拓扑执行时传入的实际 Tensor 视图。
class Op {
public:
    virtual ~Op() = default;
    virtual const char* type() const = 0;

    // 该算子是否支持被合并进 fusion（如 ReLU 可被 GEMM+ReLU 融合）
    virtual bool fusable() const { return false; }

    // 核心计算。子类实现。
    virtual void compute(const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs,
                         ExecContext& ctx) = 0;

    // 返回该算子需要的额外（非输入/输出）device workspace 字节数，默认 0
    virtual size_t workspace_bytes() const { return 0; }
};

}  // namespace tinyinfer
