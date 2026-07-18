#pragma once
// kernels.cu 中启动函数的声明（供 ops.cpp / executor 调用）
#include "common.h"
#include <cuda_runtime.h>

namespace tinynfer {
namespace kernels {

void launch_gemm(const dtype* A, const dtype* W, const dtype* B, dtype* C,
                 int M, int K, int N, bool apply_relu, cudaStream_t s);
void launch_fused_gemm_relu(const dtype* A, const dtype* W, const dtype* B,
                            dtype* C, int M, int K, int N, cudaStream_t s);
void launch_bias_add(const dtype* X, const dtype* B, dtype* Y, int M, int N,
                     cudaStream_t s);
void launch_relu(const dtype* X, dtype* Y, int n, cudaStream_t s);
void launch_softmax(const dtype* X, dtype* Y, int M, int N, cudaStream_t s);

}  // namespace kernels
}  // namespace tinynfer
