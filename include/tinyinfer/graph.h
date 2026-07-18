#pragma once
// 计算图（DAG）：节点 = 算子实例，边 = 依赖关系（tensor）
#include "common.h"
#include "op.h"
#include "tensor.h"
#include <vector>
#include <unordered_map>

namespace tinynfer {

// 计算图中的单个节点：持有一个算子实例 + 其输入/输出 tensor id
struct GraphNode {
    int id = -1;
    std::string op_type;          // 如 "Linear", "ReLU", "Softmax"
    std::unique_ptr<Op> op;        // 实际算子
    std::vector<int> input_tids;   // 输入 tensor id
    std::vector<int> output_tids;  // 输出 tensor id
    int first_use = -1;            // 拓扑序中首次被使用（生命周期分析用）
    int last_use = -1;             // 拓扑序中最后被使用
};

// 计算图 DAG：管理所有 tensor 与节点，并提供拓扑排序
class ComputeGraph {
public:
    // 创建 tensor，返回其 id
    int add_tensor(const std::vector<int>& shape);

    // 注册节点（算子实例已构建好）
    int add_node(std::unique_ptr<Op> op, const std::string& op_type,
                 const std::vector<int>& input_tids,
                 const std::vector<int>& output_tids);

    Tensor& tensor(int tid) { return tensors_[tid]; }
    const Tensor& tensor(int tid) const { return tensors_[tid]; }
    GraphNode& node(int nid) { return nodes_[nid]; }
    const GraphNode& node(int nid) const { return nodes_[nid]; }

    int num_tensors() const { return (int)tensors_.size(); }
    int num_nodes() const { return (int)nodes_.size(); }

    // 拓扑排序（Kahn 算法），返回执行顺序的节点 id 列表
    std::vector<int> topo_order() const;

    // 计算每个 tensor 的 first_use / last_use（生命周期区间），用于静态显存规划
    void analyze_lifetimes(const std::vector<int>& order);

private:
    std::vector<Tensor> tensors_;
    std::vector<GraphNode> nodes_;
};

}  // namespace tinynfer
