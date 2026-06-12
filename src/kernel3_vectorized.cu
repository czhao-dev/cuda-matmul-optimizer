#include "kernels.cuh"

namespace {

constexpr int TILE = kDefaultTileSize;
constexpr int VEC = 4;

__device__ int valid_vector_width(int remaining) {
  if (remaining <= 0) {
    return 0;
  }
  return remaining < VEC ? remaining : VEC;
}

__device__ void load4_or_scalar(const float* src, float* dst, int valid_count,
                                bool aligned) {
  if (valid_count == VEC && aligned) {
    const float4 values = *reinterpret_cast<const float4*>(src);
    dst[0] = values.x;
    dst[1] = values.y;
    dst[2] = values.z;
    dst[3] = values.w;
    return;
  }

  #pragma unroll
  for (int i = 0; i < VEC; ++i) {
    dst[i] = (i < valid_count) ? src[i] : 0.0f;
  }
}

__global__ void matmul_vectorized_kernel(const float* A, const float* B,
                                         float* C, int M, int N, int K) {
  __shared__ float tileA[TILE][TILE];
  __shared__ float tileB[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int tile_start = 0; tile_start < K; tile_start += TILE) {
    if (threadIdx.x < TILE / VEC) {
      #pragma unroll
      for (int i = 0; i < VEC; ++i) {
        tileA[threadIdx.y][threadIdx.x * VEC + i] = 0.0f;
        tileB[threadIdx.y][threadIdx.x * VEC + i] = 0.0f;
      }

      const int a_col = tile_start + threadIdx.x * VEC;
      const int a_valid = valid_vector_width(K - a_col);
      if (row < M && a_valid > 0) {
        const bool a_aligned = ((row * K + a_col) % VEC) == 0;
        load4_or_scalar(&A[row * K + a_col],
                        &tileA[threadIdx.y][threadIdx.x * VEC], a_valid,
                        a_aligned);
      }

      const int b_col = blockIdx.x * TILE + threadIdx.x * VEC;
      const int b_row = tile_start + threadIdx.y;
      const int b_valid = valid_vector_width(N - b_col);
      if (b_row < K && b_valid > 0) {
        const bool b_aligned = ((b_row * N + b_col) % VEC) == 0;
        load4_or_scalar(&B[b_row * N + b_col],
                        &tileB[threadIdx.y][threadIdx.x * VEC], b_valid,
                        b_aligned);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

}  // namespace

cudaError_t launch_vectorized(const float* A, const float* B, float* C, int M,
                              int N, int K, cudaStream_t stream) {
  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_vectorized_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}
