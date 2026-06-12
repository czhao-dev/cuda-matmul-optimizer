#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(error));
  }
}

void cublas_check(cublasStatus_t status, const char* context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::vector<float> make_matrix(int rows, int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(static_cast<long long>(rows) * cols);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

void launch_kernel(int kernel, const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K, cublasHandle_t handle) {
  switch (kernel) {
    case kGpuNaive:
      cuda_check(launch_naive(d_A, d_B, d_C, M, N, K), "launch_naive");
      break;
    case kGpuTiled:
      cuda_check(launch_tiled(d_A, d_B, d_C, M, N, K), "launch_tiled");
      break;
    case kGpuVectorized:
      cuda_check(launch_vectorized(d_A, d_B, d_C, M, N, K),
                 "launch_vectorized");
      break;
    case kGpuCoarsened:
      cuda_check(launch_coarsened(d_A, d_B, d_C, M, N, K),
                 "launch_coarsened");
      break;
    case kCublasReference:
      cublas_check(launch_cublas(handle, d_A, d_B, d_C, M, N, K),
                   "launch_cublas");
      break;
    default:
      throw std::runtime_error("Unsupported kernel");
  }
}

bool run_case(int M, int N, int K, cublasHandle_t handle) {
  std::vector<float> A = make_matrix(M, K, 123);
  std::vector<float> B = make_matrix(K, N, 456);
  std::vector<float> ref(static_cast<long long>(M) * N);
  std::vector<float> got(static_cast<long long>(M) * N);

  matmul_cpu(A.data(), B.data(), ref.data(), M, N, K);

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  const size_t bytes_A = A.size() * sizeof(float);
  const size_t bytes_B = B.size() * sizeof(float);
  const size_t bytes_C = got.size() * sizeof(float);

  cuda_check(cudaMalloc(&d_A, bytes_A), "cudaMalloc A");
  cuda_check(cudaMalloc(&d_B, bytes_B), "cudaMalloc B");
  cuda_check(cudaMalloc(&d_C, bytes_C), "cudaMalloc C");
  cuda_check(cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice),
             "copy A");
  cuda_check(cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice),
             "copy B");

  bool ok = true;
  for (int kernel = kGpuNaive; kernel <= kCublasReference; ++kernel) {
    cuda_check(cudaMemset(d_C, 0, bytes_C), "clear C");
    launch_kernel(kernel, d_A, d_B, d_C, M, N, K, handle);
    cuda_check(cudaDeviceSynchronize(), "kernel synchronize");
    cuda_check(cudaMemcpy(got.data(), d_C, bytes_C, cudaMemcpyDeviceToHost),
               "copy C");

    const bool kernel_ok = verify_result(ref.data(), got.data(), M, N, 1.0e-2f);
    std::cout << "  " << kernel_name(kernel) << ": "
              << (kernel_ok ? "PASS" : "FAIL") << "\n";
    ok = ok && kernel_ok;
  }

  cuda_check(cudaFree(d_A), "cudaFree A");
  cuda_check(cudaFree(d_B), "cudaFree B");
  cuda_check(cudaFree(d_C), "cudaFree C");
  return ok;
}

}  // namespace

int main() {
  try {
    cublasHandle_t handle = nullptr;
    cublas_check(cublasCreate(&handle), "cublasCreate");

    const int cases[][3] = {
        {1, 1, 1}, {16, 16, 16}, {31, 29, 17},
        {64, 64, 64}, {100, 96, 80}, {127, 65, 33},
    };

    bool ok = true;
    for (const auto& test_case : cases) {
      std::cout << "Case M=" << test_case[0] << " N=" << test_case[1]
                << " K=" << test_case[2] << "\n";
      ok = run_case(test_case[0], test_case[1], test_case[2], handle) && ok;
    }

    cublas_check(cublasDestroy(handle), "cublasDestroy");
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}

