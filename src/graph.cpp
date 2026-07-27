// 计算图实现：tensor/节点管理、拓扑排序、生命周期分析
#include "tinyinfer/graph.h"
#include "tinyinfer/common.h"
#include <queue>
#include <algorithm>

namespace tinyinfer {

int ComputeGraph::add_tensor(const std::vector<int>& shape) {
    Tensor t;
    t.shape = shape;
    t.size = t.num_elements();
    tensors_.push_back(t);
    return (int)tensors_.size() - 1;
}

int ComputeGraph::add_node(std::unique_ptr<Op> op, const std::string& op_type,
                           const std::vector<int>& input_tids,
                           const std::vector<int>& output_tids) {
    GraphNode n;
    n.id = (int)nodes_.size();
    n.op_type = op_type;
    n.op = std::move(op);
    n.input_tids = input_tids;
    n.output_tids = output_tids;
    nodes_.push_back(std::move(n));
    return (int)nodes_.size() - 1;
}

std::vector<int> ComputeGraph::topo_order() const {
    // Kahn 算法：统计每个节点的入度（输入 tensor 被多少个节点消费）
    std::vector<int> indeg(nodes_.size(), 0);
    // 输入 tensor 的"生产者" -> 决定边。简化模型：节点按顺序执行，
    // 边由 tensor 依赖表达；这里入度 = 该节点是否存在输入 tensor 由尚未就绪的节点产生。
    // 由于我们的图是线性链，直接按下标顺序即为可行拓扑序；
    // 通用实现：统计每个 tensor 的 producer。
    std::vector<int> producer(num_tensors(), -1);
    for (const auto& n : nodes_)
        for (int t : n.output_tids) producer[t] = n.id;

    for (const auto& n : nodes_)
        for (int t : n.input_tids)
            if (producer[t] >= 0) indeg[n.id]++;  // 来自其他节点的依赖

    std::queue<int> q;
    for (int i = 0; i < (int)nodes_.size(); ++i)
        if (indeg[i] == 0) q.push(i);

    std::vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int t : nodes_[u].output_tids) {
            // 找到消费 tensor t 的节点，入度-1
            for (int v = 0; v < (int)nodes_.size(); ++v) {
                if (v == u) continue;
                bool consumes = false;
                for (int it : nodes_[v].input_tids)
                    if (it == t) { consumes = true; break; }
                if (consumes) {
                    if (--indeg[v] == 0) q.push(v);
                }
            }
        }
    }
    if ((int)order.size() != (int)nodes_.size())
        throw std::runtime_error("ComputeGraph: cycle detected in DAG");
    return order;
}

void ComputeGraph::analyze_lifetimes(const std::vector<int>& order) {
    for (int i = 0; i < num_tensors(); ++i) {
        tensor(i).first_use = -1;
        tensor(i).last_use = -1;
    }
    for (int pos = 0; pos < (int)order.size(); ++pos) {
        int nid = order[pos];
        const auto& n = node(nid);
        auto touch = [&](int t) {
            if (tensor(t).first_use == -1) tensor(t).first_use = pos;
            tensor(t).last_use = pos;
        };
        for (int t : n.input_tids) touch(t);
        for (int t : n.output_tids) touch(t);
    }
}

}  // namespace tinyinfer
