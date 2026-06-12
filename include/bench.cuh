#pragma once

#include <cublas_v2.h>

double time_cpu_baseline_ms(const float* A, const float* B, float* C, int M,
                            int N, int K, int runs);
float time_kernel_gpu_ms(int kernel, const float* d_A, const float* d_B,
                         float* d_C, int M, int N, int K,
                         cublasHandle_t cublas_handle, int warmup_runs,
                         int timed_runs);

