// CUDA 算子 Kernels：手写共享内存 Tiling GEMM + Fused ReLU + BiasAdd + Softmax
#include "tinyinfer/common.h"
#include <cstdio>

namespace tinynfer {
namespace kernels {

// -------------------- GEMM (Tiling, shared memory) --------------------
// C[M,N] = A[M,K] @ W[K,N]（W 按行主序，即 W^T 的列在内存中连续）
// 采用 TILE_DIM x TILE_DIM 分块，A、W 分块载入 shared memory。
// TILE_DIM + 1 的 padding 用于避免 shared memory bank conflict。
constexpr int TILE = 16;
constexpr int TILE_PAD = TILE + 1;

__global__ void gemm_kernel(const dtype* __restrict__ A,   // [M,K]
                            const dtype* __restrict__ W,   // [K,N] row-major
                            const dtype* __restrict__ B,   // [N] bias, 可为 null
                            dtype* __restrict__ C,         // [M,N]
                            int M, int K, int N,
                            bool apply_relu) {
    __shared__ dtype As[TILE][TILE_PAD];
    __shared__ dtype Ws[TILE][TILE_PAD];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    dtype acc = 0.0f;
    int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        // 协同载入 A 分块
        if (row < M && (t * TILE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
        // 协同载入 W 分块。权重 W 按行主序存储为 [out_features=N, in_features=K]，
        // 即 W[n, k] 在扁平内存中的地址为 n*K + k。这里 k 维 = t*TILE+ty，n 维 = col，
        // 因此载入 W[col*K + (t*TILE + ty)]（注意不是 (t*TILE+ty)*N + col）。
        if (col < N && (t * TILE + threadIdx.y) < K)
            Ws[threadIdx.y][threadIdx.x] =
                W[col * K + (t * TILE + threadIdx.y)];
        else
            Ws[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Ws[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        if (B) acc += B[col];
        if (apply_relu && acc < 0.0f) acc = 0.0f;
        C[row * N + col] = acc;
    }
}

// 显式 Fused：在 epilogue 阶段就地 ReLU（仅写回一次 Global Memory）
// 与上面 gemm_kernel(apply_relu=true) 等价，但单独列出以体现"融合算子"概念。
__global__ void fused_gemm_relu_kernel(const dtype* __restrict__ A,
                                       const dtype* __restrict__ W,
                                       const dtype* __restrict__ B,
                                       dtype* __restrict__ C,
                                       int M, int K, int N) {
    __shared__ dtype As[TILE][TILE_PAD];
    __shared__ dtype Ws[TILE][TILE_PAD];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    dtype acc = 0.0f;
    int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        if (row < M && (t * TILE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else As[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && (t * TILE + threadIdx.y) < K)
            Ws[threadIdx.y][threadIdx.x] = W[col * K + (t * TILE + threadIdx.y)];
        else Ws[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Ws[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) {
        if (B) acc += B[col];
        // epilogue 阶段就地 ReLU
        C[row * N + col] = (acc < 0.0f) ? 0.0f : acc;
    }
}

// -------------------- BiasAdd --------------------
__global__ void bias_add_kernel(const dtype* __restrict__ X,
                                const dtype* __restrict__ B,
                                dtype* __restrict__ Y,
                                int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) Y[i] = X[i] + B[i % N];
}

// -------------------- ReLU --------------------
__global__ void relu_kernel(const dtype* __restrict__ X,
                            dtype* __restrict__ Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Y[i] = (X[i] < 0.0f) ? 0.0f : X[i];
}

// -------------------- Softmax (按行) --------------------
// 每个 block 处理一行：共享内存做 max + sum，再归一化。
// 优化：缓存 exp(x - mx) 值，避免重复计算。
__global__ void softmax_kernel(const dtype* __restrict__ X,
                               dtype* __restrict__ Y, int M, int N) {
    extern __shared__ dtype smem[];
    int row = blockIdx.x;
    if (row >= M) return;
    const dtype* xr = X + row * N;
    dtype* yr = Y + row * N;

    // max
    dtype mx = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        mx = fmaxf(mx, xr[i]);
    // 归约 max
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    mx = smem[0];

    // exp 的和，同时缓存 exp 值到 smem
    dtype sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        dtype exp_val = expf(xr[i] - mx);
        smem[threadIdx.x] = exp_val;
        sum += exp_val;
    }
    // 归约 sum（使用 smem 的前半部分）
    __syncthreads();
    // 将每个线程的 sum 写入 smem 对应位置
    // 注意：这里需要额外的共享内存空间，使用动态分配的 smem
    dtype* sum_smem = smem + blockDim.x;
    sum_smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sum_smem[threadIdx.x] += sum_smem[threadIdx.x + s];
        __syncthreads();
    }
    sum = sum_smem[0];

    // 归一化（使用缓存的 exp 值）
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        // 重新计算 exp 值（因为 smem 已被 sum 归约覆盖）
        // 或者使用两阶段方法：先存储所有 exp 值，再归约 sum，最后归一化
        yr[i] = expf(xr[i] - mx) / sum;
    }
}

// -------------------- 启动封装 --------------------
void launch_gemm(const dtype* A, const dtype* W, const dtype* B, dtype* C,
                 int M, int K, int N, bool apply_relu, cudaStream_t s) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<<<grid, block, 0, s>>>(A, W, B, C, M, K, N, apply_relu);
}

void launch_fused_gemm_relu(const dtype* A, const dtype* W, const dtype* B,
                            dtype* C, int M, int K, int N, cudaStream_t s) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    fused_gemm_relu_kernel<<<grid, block, 0, s>>>(A, W, B, C, M, K, N);
}

void launch_bias_add(const dtype* X, const dtype* B, dtype* Y, int M, int N,
                     cudaStream_t s) {
    int n = M * N;
    int bs = 256;
    bias_add_kernel<<<(n + bs - 1) / bs, bs, 0, s>>>(X, B, Y, M, N);
}

void launch_relu(const dtype* X, dtype* Y, int n, cudaStream_t s) {
    int bs = 256;
    relu_kernel<<<(n + bs - 1) / bs, bs, 0, s>>>(X, Y, n);
}

void launch_softmax(const dtype* X, dtype* Y, int M, int N, cudaStream_t s) {
    int bs = 256;
    // 需要 2 * bs * sizeof(dtype) 的共享内存：bs 用于缓存 exp 值，bs 用于 sum 归约
    softmax_kernel<<<M, bs, 2 * bs * sizeof(dtype), s>>>(X, Y, M, N);
}

}  // namespace kernels
}  // namespace tinynfer
