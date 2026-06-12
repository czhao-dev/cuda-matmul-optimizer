#include "kernels.cuh"

#include <algorithm>
#include <cmath>

const char* kernel_name(int kernel) {
  switch (kernel) {
    case kCpuBaseline:
      return "cpu_baseline";
    case kGpuNaive:
      return "gpu_naive";
    case kGpuTiled:
      return "gpu_tiled";
    case kGpuVectorized:
      return "gpu_vectorized";
    case kGpuCoarsened:
      return "gpu_coarsened";
    case kCublasReference:
      return "cublas_reference";
    default:
      return "unknown";
  }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
  std::fill(C, C + static_cast<long long>(M) * N, 0.0f);

  for (int row = 0; row < M; ++row) {
    for (int k = 0; k < K; ++k) {
      const float a = A[row * K + k];
      for (int col = 0; col < N; ++col) {
        C[row * N + col] += a * B[k * N + col];
      }
    }
  }
}

bool verify_result(const float* expected, const float* actual, int M, int N,
                   float atol) {
  const long long elements = static_cast<long long>(M) * N;
  for (long long i = 0; i < elements; ++i) {
    if (std::fabs(expected[i] - actual[i]) > atol) {
      return false;
    }
  }
  return true;
}

