#include "bench.cuh"
#include "kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int kernel = kGpuTiled;
  int M = 1024;
  int N = 1024;
  int K = 1024;
  int warmup_runs = 10;
  int timed_runs = 100;
  bool run_test = false;
  bool run_bench = false;
  bool verify = true;
  std::string output = "benchmarks/results.csv";
};

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

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --kernel <0-5>       Kernel id: 0 CPU, 1 naive, 2 tiled, 3 vectorized,\n"
      << "                       4 coarsened, 5 cuBLAS (default: 2)\n"
      << "  --size <n>           Square matrix size M=N=K (default: 1024)\n"
      << "  --m <n> --n <n> --k <n>\n"
      << "                       Rectangular dimensions for C(MxN)=A(MxK)*B(KxN)\n"
      << "  --warmup <n>         GPU warmup launches (default: 10)\n"
      << "  --runs <n>           Timed runs to average (default: 100)\n"
      << "  --no-verify          Skip CPU reference verification\n"
      << "  --test               Run correctness checks across kernels and sizes\n"
      << "  --bench              Run benchmark suite and write CSV\n"
      << "  --output <path>      Benchmark CSV path (default: benchmarks/results.csv)\n"
      << "  --help               Show this help\n";
}

int parse_int_arg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("Missing value for ") + argv[i]);
  }
  return std::stoi(argv[++i]);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--kernel") {
      options.kernel = parse_int_arg(argc, argv, i);
    } else if (arg == "--size") {
      options.M = options.N = options.K = parse_int_arg(argc, argv, i);
    } else if (arg == "--m") {
      options.M = parse_int_arg(argc, argv, i);
    } else if (arg == "--n") {
      options.N = parse_int_arg(argc, argv, i);
    } else if (arg == "--k") {
      options.K = parse_int_arg(argc, argv, i);
    } else if (arg == "--warmup") {
      options.warmup_runs = parse_int_arg(argc, argv, i);
    } else if (arg == "--runs") {
      options.timed_runs = parse_int_arg(argc, argv, i);
    } else if (arg == "--no-verify") {
      options.verify = false;
    } else if (arg == "--test") {
      options.run_test = true;
    } else if (arg == "--bench") {
      options.run_bench = true;
    } else if (arg == "--output") {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for --output");
      }
      options.output = argv[++i];
    } else if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  return options;
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

double gflops(int M, int N, int K, double ms) {
  const double ops = 2.0 * static_cast<double>(M) * N * K;
  return ops / (ms * 1.0e6);
}

void run_gpu_once(const Options& options, const std::vector<float>& A,
                  const std::vector<float>& B, std::vector<float>& C,
                  cublasHandle_t handle) {
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  const size_t bytes_A = A.size() * sizeof(float);
  const size_t bytes_B = B.size() * sizeof(float);
  const size_t bytes_C = C.size() * sizeof(float);

  cuda_check(cudaMalloc(&d_A, bytes_A), "cudaMalloc A");
  cuda_check(cudaMalloc(&d_B, bytes_B), "cudaMalloc B");
  cuda_check(cudaMalloc(&d_C, bytes_C), "cudaMalloc C");
  cuda_check(cudaMemcpy(d_A, A.data(), bytes_A, cudaMemcpyHostToDevice),
             "copy A to device");
  cuda_check(cudaMemcpy(d_B, B.data(), bytes_B, cudaMemcpyHostToDevice),
             "copy B to device");

  const float ms = time_kernel_gpu_ms(options.kernel, d_A, d_B, d_C, options.M,
                                      options.N, options.K, handle,
                                      options.warmup_runs, options.timed_runs);
  cuda_check(cudaMemcpy(C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost),
             "copy C to host");

  cuda_check(cudaFree(d_A), "cudaFree A");
  cuda_check(cudaFree(d_B), "cudaFree B");
  cuda_check(cudaFree(d_C), "cudaFree C");

  std::cout << kernel_name(options.kernel) << ": " << std::fixed
            << std::setprecision(4) << ms << " ms, "
            << std::setprecision(2) << gflops(options.M, options.N, options.K, ms)
            << " GFLOP/s\n";
}

bool run_correctness_case(int M, int N, int K, cublasHandle_t handle) {
  std::vector<float> A = make_matrix(M, K, 123);
  std::vector<float> B = make_matrix(K, N, 456);
  std::vector<float> ref(static_cast<long long>(M) * N);
  std::vector<float> got(static_cast<long long>(M) * N);

  matmul_cpu(A.data(), B.data(), ref.data(), M, N, K);

  bool ok = true;
  for (int kernel = kGpuNaive; kernel <= kCublasReference; ++kernel) {
    Options options;
    options.kernel = kernel;
    options.M = M;
    options.N = N;
    options.K = K;
    options.warmup_runs = 1;
    options.timed_runs = 1;
    run_gpu_once(options, A, B, got, handle);
    const bool kernel_ok = verify_result(ref.data(), got.data(), M, N, 1.0e-2f);
    std::cout << "  " << kernel_name(kernel) << " correctness: "
              << (kernel_ok ? "PASS" : "FAIL") << "\n";
    ok = ok && kernel_ok;
  }
  return ok;
}

int run_tests() {
  cublasHandle_t handle = nullptr;
  cublas_check(cublasCreate(&handle), "cublasCreate");

  const int cases[][3] = {
      {1, 1, 1}, {16, 16, 16}, {31, 29, 17},
      {64, 64, 64}, {100, 96, 80}, {127, 65, 33},
  };

  bool ok = true;
  for (const auto& test_case : cases) {
    std::cout << "Testing M=" << test_case[0] << " N=" << test_case[1]
              << " K=" << test_case[2] << "\n";
    ok = run_correctness_case(test_case[0], test_case[1], test_case[2],
                              handle) && ok;
  }

  cublas_check(cublasDestroy(handle), "cublasDestroy");
  return ok ? 0 : 1;
}

void run_single(const Options& options) {
  if (options.kernel < kCpuBaseline || options.kernel > kCublasReference) {
    throw std::runtime_error("Kernel must be between 0 and 5");
  }

  std::vector<float> A = make_matrix(options.M, options.K, 123);
  std::vector<float> B = make_matrix(options.K, options.N, 456);
  std::vector<float> C(static_cast<long long>(options.M) * options.N);
  std::vector<float> ref(static_cast<long long>(options.M) * options.N);

  if (options.kernel == kCpuBaseline) {
    const double ms = time_cpu_baseline_ms(A.data(), B.data(), C.data(),
                                           options.M, options.N, options.K,
                                           std::max(1, options.timed_runs));
    std::cout << kernel_name(options.kernel) << ": " << std::fixed
              << std::setprecision(4) << ms << " ms, "
              << std::setprecision(2)
              << gflops(options.M, options.N, options.K, ms) << " GFLOP/s\n";
    return;
  }

  cublasHandle_t handle = nullptr;
  cublas_check(cublasCreate(&handle), "cublasCreate");
  run_gpu_once(options, A, B, C, handle);
  cublas_check(cublasDestroy(handle), "cublasDestroy");

  if (options.verify) {
    matmul_cpu(A.data(), B.data(), ref.data(), options.M, options.N, options.K);
    const bool ok = verify_result(ref.data(), C.data(), options.M, options.N,
                                  1.0e-2f);
    std::cout << "verification: " << (ok ? "PASS" : "FAIL") << "\n";
    if (!ok) {
      throw std::runtime_error("Verification failed");
    }
  }
}

void run_benchmarks(const Options& options) {
  std::ofstream csv(options.output);
  if (!csv) {
    throw std::runtime_error("Could not open benchmark output: " +
                             options.output);
  }
  csv << "kernel,size_m,size_n,size_k,time_ms,speedup_vs_cpu,gflops,verified\n";

  cublasHandle_t handle = nullptr;
  cublas_check(cublasCreate(&handle), "cublasCreate");

  const int sizes[] = {256, 1024, 4096};
  for (int size : sizes) {
    std::vector<float> A = make_matrix(size, size, 123);
    std::vector<float> B = make_matrix(size, size, 456);
    std::vector<float> ref(static_cast<long long>(size) * size);
    std::vector<float> C(static_cast<long long>(size) * size);

    std::cout << "Benchmarking size " << size << "x" << size << "\n";
    const double cpu_ms =
        time_cpu_baseline_ms(A.data(), B.data(), ref.data(), size, size, size, 1);
    csv << kernel_name(kCpuBaseline) << "," << size << "," << size << ","
        << size << "," << cpu_ms << ",1," << gflops(size, size, size, cpu_ms)
        << ",true\n";

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    const size_t bytes = static_cast<size_t>(size) * size * sizeof(float);
    cuda_check(cudaMalloc(&d_A, bytes), "cudaMalloc A");
    cuda_check(cudaMalloc(&d_B, bytes), "cudaMalloc B");
    cuda_check(cudaMalloc(&d_C, bytes), "cudaMalloc C");
    cuda_check(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice),
               "copy A to device");
    cuda_check(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice),
               "copy B to device");

    for (int kernel = kGpuNaive; kernel <= kCublasReference; ++kernel) {
      cuda_check(cudaMemset(d_C, 0, bytes), "clear device C");
      const float ms = time_kernel_gpu_ms(kernel, d_A, d_B, d_C, size, size,
                                          size, handle, options.warmup_runs,
                                          options.timed_runs);
      cuda_check(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost),
                 "copy C to host");
      const bool verified =
          verify_result(ref.data(), C.data(), size, size, 1.0e-2f);
      csv << kernel_name(kernel) << "," << size << "," << size << "," << size
          << "," << ms << "," << cpu_ms / ms << ","
          << gflops(size, size, size, ms) << ","
          << (verified ? "true" : "false") << "\n";
      std::cout << "  " << kernel_name(kernel) << ": " << ms << " ms, "
                << (verified ? "PASS" : "FAIL") << "\n";
    }

    cuda_check(cudaFree(d_A), "cudaFree A");
    cuda_check(cudaFree(d_B), "cudaFree B");
    cuda_check(cudaFree(d_C), "cudaFree C");
  }

  cublas_check(cublasDestroy(handle), "cublasDestroy");
  std::cout << "Wrote " << options.output << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.run_test) {
      return run_tests();
    }
    if (options.run_bench) {
      run_benchmarks(options);
      return 0;
    }
    run_single(options);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
