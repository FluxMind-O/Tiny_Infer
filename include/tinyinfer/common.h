#pragma once
// TinyInfer 公共类型与基础设施
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>

namespace tinyinfer {

// 统一浮点精度（FP32，与 PDF 中 Benchmark 一致）
using dtype = float;

// CUDA 调用错误检查
#define TINYINFER_CUDA_CHECK(call)                                              \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            throw std::runtime_error(                                         \
                std::string("[CUDA ERROR] ") + cudaGetErrorString(_e) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                      \
    } while (0)

// 日志宏（阶段四：统一日志系统）
#define TINYINFER_LOG(fmt, ...) \
    fprintf(stdout, "[TinyInfer] " fmt "\n", ##__VA_ARGS__)
#define TINYINFER_WARN(fmt, ...) \
    fprintf(stderr, "[TinyInfer][WARN] " fmt "\n", ##__VA_ARGS__)

// 简单计时（微秒）
double get_wall_time_us();

// 在 host 上打印向量（调试用）
void print_tensor(const dtype* h_data, int n, const std::string& name,
                  int max_print = 8);

}  // namespace tinyinfer
