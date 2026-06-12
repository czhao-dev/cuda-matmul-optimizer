#include "kernels.cuh"

namespace {

constexpr int TILE = kDefaultTileSize;
constexpr int COARSEN = 2;

__global__ void matmul_coarsened_kernel(const float* A, const float* B,
                                        float* C, int M, int N, int K) {
  __shared__ float tileA[TILE * COARSEN][TILE];
  __shared__ float tileB[TILE][TILE * COARSEN];

  const int row0 = blockIdx.y * TILE * COARSEN + threadIdx.y * COARSEN;
  const int row1 = row0 + 1;
  const int col0 = blockIdx.x * TILE * COARSEN + threadIdx.x * COARSEN;
  const int col1 = col0 + 1;

  float sum00 = 0.0f;
  float sum01 = 0.0f;
  float sum10 = 0.0f;
  float sum11 = 0.0f;

  for (int tile_start = 0; tile_start < K; tile_start += TILE) {
    const int a_col = tile_start + threadIdx.x;
    const int b_row = tile_start + threadIdx.y;

    tileA[threadIdx.y * COARSEN][threadIdx.x] =
        (row0 < M && a_col < K) ? A[row0 * K + a_col] : 0.0f;
    tileA[threadIdx.y * COARSEN + 1][threadIdx.x] =
        (row1 < M && a_col < K) ? A[row1 * K + a_col] : 0.0f;

    tileB[threadIdx.y][threadIdx.x * COARSEN] =
        (b_row < K && col0 < N) ? B[b_row * N + col0] : 0.0f;
    tileB[threadIdx.y][threadIdx.x * COARSEN + 1] =
        (b_row < K && col1 < N) ? B[b_row * N + col1] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      const float a0 = tileA[threadIdx.y * COARSEN][k];
      const float a1 = tileA[threadIdx.y * COARSEN + 1][k];
      const float b0 = tileB[k][threadIdx.x * COARSEN];
      const float b1 = tileB[k][threadIdx.x * COARSEN + 1];
      sum00 += a0 * b0;
      sum01 += a0 * b1;
      sum10 += a1 * b0;
      sum11 += a1 * b1;
    }
    __syncthreads();
  }

  if (row0 < M && col0 < N) C[row0 * N + col0] = sum00;
  if (row0 < M && col1 < N) C[row0 * N + col1] = sum01;
  if (row1 < M && col0 < N) C[row1 * N + col0] = sum10;
  if (row1 < M && col1 < N) C[row1 * N + col1] = sum11;
}

}  // namespace

cudaError_t launch_coarsened(const float* A, const float* B, float* C, int M,
                             int N, int K, cudaStream_t stream) {
  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE * COARSEN - 1) / (TILE * COARSEN),
                  (M + TILE * COARSEN - 1) / (TILE * COARSEN));
  matmul_coarsened_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  return cudaGetLastError();
}

