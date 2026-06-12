#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

constexpr int kDefaultTileSize = 16;

enum KernelId {
  kCpuBaseline = 0,
  kGpuNaive = 1,
  kGpuTiled = 2,
  kGpuVectorized = 3,
  kGpuCoarsened = 4,
  kCublasReference = 5,
};

const char* kernel_name(int kernel);

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K);
bool verify_result(const float* expected, const float* actual, int M, int N,
                   float atol = 1.0e-3f);

cudaError_t launch_naive(const float* A, const float* B, float* C, int M, int N,
                         int K, cudaStream_t stream = nullptr);
cudaError_t launch_tiled(const float* A, const float* B, float* C, int M, int N,
                         int K, cudaStream_t stream = nullptr);
cudaError_t launch_vectorized(const float* A, const float* B, float* C, int M,
                              int N, int K, cudaStream_t stream = nullptr);
cudaError_t launch_coarsened(const float* A, const float* B, float* C, int M,
                             int N, int K, cudaStream_t stream = nullptr);
cublasStatus_t launch_cublas(cublasHandle_t handle, const float* A,
                             const float* B, float* C, int M, int N, int K);

