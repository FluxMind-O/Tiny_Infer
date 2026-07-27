// 静态显存规划器：生命周期区间复用（区间图着色 / 内存池复用）
#include "tinyinfer/memory.h"
#include "tinyinfer/common.h"
#include <algorithm>
#include <vector>

namespace tinyinfer {

// 256 字节对齐（CUDA 全局内存访问的最优对齐粒度）
static inline size_t align_up(size_t bytes, size_t align = 256) {
    return (bytes + align - 1) / align * align;
}

size_t MemoryPlanner::plan(ComputeGraph& graph, const std::vector<int>& order,
                           bool reuse) {
    (void)order;
    // 收集所有"可复用"的中间 tensor（非 fixed：输入/输出/权重不参与）
    std::vector<int> live;
    for (int i = 0; i < graph.num_tensors(); ++i) {
        if (!graph.tensor(i).fixed && graph.tensor(i).size > 0)
            live.push_back(i);
    }

    if (!reuse) {
        // 传统方式：每个中间 tensor 独立分配，互不重叠（无复用）
        size_t total = 0;
        for (int t : live) {
            graph.tensor(t).offset = (int)total;
            total += align_up(graph.tensor(t).size * sizeof(dtype));
        }
        total_bytes_ = total;
        return total;
    }

    // 区间图着色 / 内存池复用：
    // 维护一个 free_list（已释放、可复用的偏移区间），按偏移升序。
    // 处理按 first_use 排序的每个 tensor，复用第一个能容纳其大小的偏移；
    // 没有可复用区间则向上扩展统一 buffer。
    struct Seg { int tid; int first_use, last_use; size_t size; };
    std::vector<Seg> segs;
    for (int t : live) {
        segs.push_back({t, graph.tensor(t).first_use,
                        graph.tensor(t).last_use,
                        align_up((size_t)graph.tensor(t).size * sizeof(dtype))});
    }
    std::sort(segs.begin(), segs.end(),
              [](const Seg& a, const Seg& b) { return a.first_use < b.first_use; });

    // free_list: 可复用偏移 (offset, size)
    std::vector<std::pair<size_t, size_t>> free_list;
    // running: 活动区间 (last_use, offset, size)
    std::vector<std::tuple<int, size_t, size_t>> running;
    size_t total = 0;

    for (auto& s : segs) {
        // 归还所有生命周期已结束（last_use < 当前 first_use）的区间
        for (auto it = running.begin(); it != running.end();) {
            if (std::get<0>(*it) < s.first_use) {
                free_list.push_back({std::get<1>(*it), std::get<2>(*it)});
                it = running.erase(it);
            } else {
                ++it;
            }
        }
        // 尝试复用第一个能容纳的 free slot（偏移最小优先）
        bool placed = false;
        for (auto it = free_list.begin(); it != free_list.end(); ++it) {
            if (it->second >= s.size) {
                graph.tensor(s.tid).offset = (int)it->first;
                running.push_back({s.last_use, it->first, s.size});
                // 如果 slot 比需要的大，将剩余空间放回 free_list
                size_t extra_offset = 0, extra_size = 0;
                bool has_extra = false;
                if (it->second > s.size) {
                    extra_offset = it->first + s.size;
                    extra_size = it->second - s.size;
                    has_extra = true;
                }
                free_list.erase(it);
                if (has_extra) {
                    free_list.push_back({extra_offset, extra_size});
                }
                placed = true;
                break;
            }
        }
        if (!placed) {
            graph.tensor(s.tid).offset = (int)total;
            running.push_back({s.last_use, total, s.size});
            total += s.size;
        }
    }

    total_bytes_ = total;
    return total;
}

dtype* MemoryPlanner::allocate(ComputeGraph& graph) {
    if (total_bytes_ == 0) return nullptr;
    dtype* buf = nullptr;
    TINYINFER_CUDA_CHECK(cudaMalloc(&buf, total_bytes_));
    TINYINFER_CUDA_CHECK(cudaMemset(buf, 0, total_bytes_));
    for (int i = 0; i < graph.num_tensors(); ++i) {
        if (!graph.tensor(i).fixed && graph.tensor(i).size > 0)
            graph.tensor(i).data = buf + (graph.tensor(i).offset / sizeof(dtype));
    }
    return buf;
}

}  // namespace tinyinfer
