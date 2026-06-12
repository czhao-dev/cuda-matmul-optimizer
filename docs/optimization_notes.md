# Optimization Notes

Raw numbers are in `benchmarks/results.csv`. This document interprets them.

Hardware: NVIDIA Tesla T4 (sm_75), CUDA 13.0, Driver 580.159.04, AWS g4dn.xlarge.

## Kernel 0: CPU Baseline

- Purpose: single-threaded i-k-j loop (row-major friendly) as the floor for all speedup comparisons.
- Timing notes: 256: 2.98 ms (11.2 GFLOP/s) · 1024: 211.6 ms (10.2 GFLOP/s) · 4096: 24475 ms (5.6 GFLOP/s)
- Bottleneck: single core, no SIMD; throughput drops at 4096 as the working set spills out of cache.

## Kernel 1: GPU Naive

- Purpose: one CUDA thread per output element, reading A and B directly from global memory.
- Timing notes: 256: 0.060 ms (562 GFLOP/s, 50x) · 1024: 4.19 ms (512 GFLOP/s, 50x) · 4096: 305 ms (451 GFLOP/s, 80x)
- Bottleneck: memory-bandwidth bound — each thread re-reads a full row of A and column of B from global memory, with massive redundancy across threads.
- Nsight Compute observations: `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` = 4.29 GB at 1024x1024.

## Kernel 2: Shared Memory Tiling

- Purpose: cooperative 16x16 tile loads into shared memory, reused TILE_SIZE times per element.
- Timing notes: 256: 0.050 ms (670 GFLOP/s, 60x) · 1024: 3.14 ms (683 GFLOP/s, 67x) · 4096: 235 ms (585 GFLOP/s, 104x)
- Bottleneck addressed: the redundant global loads from Kernel 1.
- Tile size experiment: not yet run — `TILE_SIZE` is a compile-time constant (`kDefaultTileSize` in `kernels.cuh`); sweeping 8/16/32 requires rebuilding per value.
- Nsight Compute observations: global load bytes dropped from 4.29 GB (naive) to 534 MB (tiled) at 1024x1024 — an **8.0x reduction**, close to the theoretical TILE_SIZE=16 bound. DRAM bytes read = 202 MB, 8.9% of peak DRAM throughput.

## Kernel 3: Vectorized Loads

- Purpose: `float4` tile loads to widen each load transaction to 128 bits.
- Timing notes: 256: 0.067 ms (503 GFLOP/s, 45x) · 1024: 4.14 ms (519 GFLOP/s, 51x) · 4096: 300 ms (458 GFLOP/s, 82x) — **slightly slower than tiled at every size measured.**
- Bottleneck addressed: intended to reduce load instruction count, but did not pay off here.
- Nsight Compute observations: DRAM bytes read = 207 MB (vs 202 MB for tiled), throughput 6.8% of peak (vs 8.9% for tiled) — essentially flat to slightly worse.
- Interpretation: the per-thread bounds/alignment checks (`valid_vector_width`, `load4_or_scalar`) needed to support arbitrary M/N/K add branch and predicate overhead that outweighs the float4 benefit at these sizes, where 1024 and 4096 are already multiples of TILE and VEC. A specialization that skips the runtime checks when K and N are known multiples of 4 would likely recover the expected gain.

## Kernel 4: Thread Coarsening

- Purpose: each thread computes a 2x2 output block, amortizing index computation and `__syncthreads()` overhead across more arithmetic.
- Timing notes: 256: 0.041 ms (816 GFLOP/s, 73x) · 1024: 2.11 ms (1018 GFLOP/s, 100x) · 4096: 144 ms (953 GFLOP/s, 170x) — the best of the four custom kernels, ~1.6-1.7x faster than tiled.
- Bottleneck addressed: thread scheduling and synchronization overhead relative to useful arithmetic.
- Coarsening factor experiment: not yet run — `COARSEN` is hardcoded to 2x2 in `kernel4_coarsened.cu`; 4x4 and 2x4 would each need a separate build.

## cuBLAS Reference

- Timing notes: 256: 0.019 ms (1801 GFLOP/s) · 1024: 0.445 ms (4829 GFLOP/s) · 4096: 31.3 ms (4385 GFLOP/s)
- Gap vs custom kernels: at 4096, the coarsened kernel (953 GFLOP/s) reaches ~22% of cuBLAS (4385 GFLOP/s).
- Interpretation: cuBLAS uses register-level blocking (larger per-thread output tiles held in registers), warp-specialized/double-buffered loads, and tile sizes tuned per architecture. Register blocking is the natural next step to close part of this gap.
