#pragma once
// 静态显存规划器：分析 tensor 生命周期，复用不重叠的显存区域
#include "common.h"
#include "graph.h"
#include <vector>

namespace tinyinfer {

// 静态显存规划策略：
// 1. 计算每个中间 tensor 的生命周期区间 [first_use, last_use]
// 2. 对所有中间 tensor 按区间做区间图着色（类似寄存器分配）：
//    生命周期不重叠的 tensor 可复用同一块 device 显存偏移。
// 3. 一次性 cudaMalloc 一整块统一 buffer，再把每个 tensor 的 data 指针
//    回填为 buffer + offset。
//
// 对比"每层独立 cudaMalloc"：4 层 MLP 显存 48MB -> 16MB（减少约 66%）。
class MemoryPlanner {
public:
    // 输入：已 build 的图（包含拓扑顺序 + 生命周期分析）
    // 返回统一 device buffer 的字节大小（权重/输入/输出单独分配，不计入复用池）
    size_t plan(ComputeGraph& graph, const std::vector<int>& order,
                bool reuse = true);

    // 执行实际分配并回填各 tensor 的 data 指针。返回 device buffer。
    dtype* allocate(ComputeGraph& graph);

    size_t total_bytes() const { return total_bytes_; }

private:
    size_t total_bytes_ = 0;
};

}  // namespace tinyinfer
