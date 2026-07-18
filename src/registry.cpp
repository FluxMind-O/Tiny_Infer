// 算子注册表实现（工厂模式）
#include "tinyinfer/registry.h"
#include "tinyinfer/ops.h"

namespace tinynfer {

OpRegistry& OpRegistry::instance() {
    static OpRegistry reg;
    return reg;
}

void OpRegistry::register_op(const std::string& type, Creator c) {
    creators_[type] = std::move(c);
}

std::unique_ptr<Op> OpRegistry::create(
    const std::string& type,
    const std::unordered_map<std::string, std::string>& cfg,
    const std::unordered_map<std::string, const dtype*>& weights) const {
    auto it = creators_.find(type);
    if (it == creators_.end())
        throw std::runtime_error("OpRegistry: unknown op type '" + type + "'");
    return it->second(cfg, weights);
}

bool OpRegistry::has(const std::string& type) const {
    return creators_.find(type) != creators_.end();
}

}  // namespace tinynfer
