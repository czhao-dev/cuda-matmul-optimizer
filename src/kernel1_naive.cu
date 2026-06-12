#include "kernels.cuh"

namespace {

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

}  // namespace

cudaError_t launch_naive(const float* A, const float* B, float* C, int M, int N,
                         int K, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  matmul_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

