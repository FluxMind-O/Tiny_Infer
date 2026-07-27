#pragma once
// 计算图中的张量（tensor）抽象
#include "common.h"
#include <vector>

namespace tinyinfer {

// Tensor 描述：形状 + device 指针（由 MemoryPlanner 在静态规划阶段分配）
struct Tensor {
    std::vector<int> shape;   // 例如 [batch, features]
    int size = 0;             // 元素总数 = prod(shape)
    int offset = 0;           // 在统一 device buffer 中的字节偏移（静态规划）
    int first_use = -1;       // 生命周期：首次被使用的拓扑位置
    int last_use = -1;        // 生命周期：最后被使用的拓扑位置
    bool fixed = false;       // 是否为固定 buffer（输入/输出/权重），不参与复用
    dtype* data = nullptr;    // 实际 device 指针（规划后回填）

    int num_elements() const {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }
};

}  // namespace tinyinfer
