// extern "C" entry points for the Rust FFI layer (src/ffi.rs). Each function
// is a one-line delegate to the existing C++ launch_* functions declared in
// kernels.cuh — no grid/block math or kernel launch syntax lives here, only
// the name-mangling boundary between C++ and Rust.
#include "kernels.cuh"

extern "C" int cuda_matmul_launch_naive(const float* a, const float* b, float* c,
                                        int m, int n, int k) {
  return static_cast<int>(launch_naive(a, b, c, m, n, k));
}

extern "C" int cuda_matmul_launch_tiled(const float* a, const float* b, float* c,
                                        int m, int n, int k) {
  return static_cast<int>(launch_tiled(a, b, c, m, n, k));
}

extern "C" int cuda_matmul_launch_vectorized(const float* a, const float* b,
                                             float* c, int m, int n, int k) {
  return static_cast<int>(launch_vectorized(a, b, c, m, n, k));
}

extern "C" int cuda_matmul_launch_coarsened(const float* a, const float* b,
                                            float* c, int m, int n, int k) {
  return static_cast<int>(launch_coarsened(a, b, c, m, n, k));
}
