#pragma once
// 算子注册表：工厂模式。按 JSON 中的 "type" 创建对应算子实例。
#include "common.h"
#include "op.h"
#include <string>
#include <functional>
#include <unordered_map>
#include <memory>

namespace tinyinfer {

// 算子创建器：根据 layer 配置 + 权重指针，构造 Op 实例。
// 参数约定：in/out features 用于 Linear；weights 为已加载到 device 的指针。
class OpRegistry {
public:
    using Creator = std::function<std::unique_ptr<Op>(
        const std::unordered_map<std::string, std::string>& cfg,
        const std::unordered_map<std::string, const dtype*>& weights)>;

    static OpRegistry& instance();

    void register_op(const std::string& type, Creator c);
    std::unique_ptr<Op> create(const std::string& type,
        const std::unordered_map<std::string, std::string>& cfg,
        const std::unordered_map<std::string, const dtype*>& weights) const;

    bool has(const std::string& type) const;

private:
    std::unordered_map<std::string, Creator> creators_;
};

// 注册辅助宏（在 .cu/.cpp 顶层使用）
#define TINYINFER_REGISTER_OP(Type, CtorFn)                                  \
    static struct _Reg_##Type {                                             \
        _Reg_##Type() {                                                     \
            OpRegistry::instance().register_op(#Type, CtorFn);             \
        }                                                                   \
    } _reg_##Type;

}  // namespace tinyinfer
