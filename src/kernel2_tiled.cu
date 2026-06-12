#include "kernels.cuh"

namespace {

constexpr int TILE = kDefaultTileSize;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
  __shared__ float tileA[TILE][TILE];
  __shared__ float tileB[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int tile_start = 0; tile_start < K; tile_start += TILE) {
    const int a_col = tile_start + threadIdx.x;
    const int b_row = tile_start + threadIdx.y;

    tileA[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    tileB[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
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

cudaError_t launch_tiled(const float* A, const float* B, float* C, int M, int N,
                         int K, cudaStream_t stream) {
  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

