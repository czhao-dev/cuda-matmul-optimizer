#include "kernels.cuh"

cublasStatus_t launch_cublas(cublasHandle_t handle, const float* A,
                             const float* B, float* C, int M, int N, int K) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Row-major C = A(MxK) * B(KxN) is equivalent to column-major
  // C^T = B^T(NxK) * A^T(KxM) using the same memory buffers.
  return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                     A, K, &beta, C, N);
}

