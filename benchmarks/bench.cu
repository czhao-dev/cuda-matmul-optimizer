#include "bench.cuh"

#include "kernels.cuh"

#include <chrono>
#include <stdexcept>
#include <string>

namespace {

void throw_if_cuda_error(cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(error));
  }
}

void throw_if_cublas_error(cublasStatus_t status, const char* context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void launch_by_id(int kernel, const float* d_A, const float* d_B, float* d_C,
                  int M, int N, int K, cublasHandle_t cublas_handle) {
  switch (kernel) {
    case kGpuNaive:
      throw_if_cuda_error(launch_naive(d_A, d_B, d_C, M, N, K),
                          "launch_naive");
      break;
    case kGpuTiled:
      throw_if_cuda_error(launch_tiled(d_A, d_B, d_C, M, N, K),
                          "launch_tiled");
      break;
    case kGpuVectorized:
      throw_if_cuda_error(launch_vectorized(d_A, d_B, d_C, M, N, K),
                          "launch_vectorized");
      break;
    case kGpuCoarsened:
      throw_if_cuda_error(launch_coarsened(d_A, d_B, d_C, M, N, K),
                          "launch_coarsened");
      break;
    case kCublasReference:
      throw_if_cublas_error(launch_cublas(cublas_handle, d_A, d_B, d_C, M, N, K),
                            "launch_cublas");
      break;
    default:
      throw std::runtime_error("Unsupported GPU kernel id: " +
                               std::to_string(kernel));
  }
}

}  // namespace

double time_cpu_baseline_ms(const float* A, const float* B, float* C, int M,
                            int N, int K, int runs) {
  const int measured_runs = runs > 0 ? runs : 1;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < measured_runs; ++i) {
    matmul_cpu(A, B, C, M, N, K);
  }
  const auto stop = std::chrono::steady_clock::now();
  const std::chrono::duration<double, std::milli> elapsed = stop - start;
  return elapsed.count() / measured_runs;
}

float time_kernel_gpu_ms(int kernel, const float* d_A, const float* d_B,
                         float* d_C, int M, int N, int K,
                         cublasHandle_t cublas_handle, int warmup_runs,
                         int timed_runs) {
  const int warmups = warmup_runs >= 0 ? warmup_runs : 0;
  const int measured_runs = timed_runs > 0 ? timed_runs : 1;

  for (int i = 0; i < warmups; ++i) {
    launch_by_id(kernel, d_A, d_B, d_C, M, N, K, cublas_handle);
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "warmup synchronize");

  cudaEvent_t start;
  cudaEvent_t stop;
  throw_if_cuda_error(cudaEventCreate(&start), "cudaEventCreate start");
  throw_if_cuda_error(cudaEventCreate(&stop), "cudaEventCreate stop");

  throw_if_cuda_error(cudaEventRecord(start), "cudaEventRecord start");
  for (int i = 0; i < measured_runs; ++i) {
    launch_by_id(kernel, d_A, d_B, d_C, M, N, K, cublas_handle);
  }
  throw_if_cuda_error(cudaEventRecord(stop), "cudaEventRecord stop");
  throw_if_cuda_error(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float elapsed_ms = 0.0f;
  throw_if_cuda_error(cudaEventElapsedTime(&elapsed_ms, start, stop),
                      "cudaEventElapsedTime");
  throw_if_cuda_error(cudaEventDestroy(start), "cudaEventDestroy start");
  throw_if_cuda_error(cudaEventDestroy(stop), "cudaEventDestroy stop");

  return elapsed_ms / measured_runs;
}

