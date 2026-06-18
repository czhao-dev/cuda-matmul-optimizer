# CUDA Matrix Multiplication Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.x%2B-76B900.svg)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)

> A step-by-step optimization of matrix multiplication on the GPU — from a
> naive CPU baseline to a high-performance tiled CUDA kernel — with measured
> speedups at each stage and clear explanations of every technique used.

---

## Contents

- [Overview](#overview)
- [The Five Kernels](#the-five-kernels)
- [Benchmark Results](#benchmark-results)
- [Key Concepts Explained](#key-concepts-explained)
- [Repo Structure](#repo-structure)
- [Build & Run](#build--run)
- [Future Work](#future-work)
- [Further Reading](#further-reading)
- [License](#license)

---

## Overview

Matrix multiplication (GEMM — General Matrix Multiply) is the single most
important operation in deep learning. Every linear layer, every attention
mechanism, every convolution reduces to a matrix multiply at the hardware
level. This project starts with the simplest possible implementation and
optimizes it in five progressive steps, measuring the speedup at each stage
and explaining exactly why each change improves performance.

The goal is not just to produce fast code, but to understand *why* GPU matrix
multiplication is hard to optimize and *how* each technique addresses a
specific hardware bottleneck — global memory latency, memory coalescing,
arithmetic intensity, and scheduling overhead.

---

## The Five Kernels

### Kernel 0 — CPU Baseline
A straightforward triple nested loop on the CPU with no parallelism. This is
the benchmark floor — every subsequent measurement is reported as a speedup
ratio relative to this baseline.

---

### Kernel 1 — GPU Naive
One CUDA thread per output element. Each thread walks its row of A and its
column of B independently, loading every value directly from global memory.

**Why it is slow:** global memory on the GPU has high latency (~400–800 cycles)
and limited bandwidth. Adjacent threads reading the same row of A trigger
redundant global memory loads — the same value is fetched K times across K
different threads. This kernel is memory-bandwidth bound, not compute bound.

---

### Kernel 2 — Shared Memory Tiling
This is the core optimization. Instead of reading directly from global memory
on every multiply-add, threads cooperate to load a tile of A and a tile of B
into shared memory — a small, fast on-chip scratchpad (~100× lower latency
than global memory) — and then compute from there.

```
Global memory          Shared memory (on-chip)
┌─────────────┐        ┌──────────┐
│  Matrix A   │──────▶ │  Tile A  │ ← all threads in block load together
└─────────────┘        └──────────┘
                            │
┌─────────────┐        ┌──────────┐  multiply-accumulate
│  Matrix B   │──────▶ │  Tile B  │ ← from shared memory
└─────────────┘        └──────────┘
```

Threads in a block collectively load a TILE_SIZE × TILE_SIZE sub-matrix of A
and B into shared memory, synchronize with `__syncthreads()`, compute partial
dot products from shared memory, then advance to the next tile.

**Why it is fast:** each value loaded from global memory is now reused
TILE_SIZE times within shared memory instead of once. Global memory traffic
drops by a factor of TILE_SIZE. For TILE_SIZE = 16, that is a 16× reduction
in global memory accesses — the dominant cost in the naive kernel.

---

### Kernel 3 — Vectorized Memory Loads
Global memory loads are most efficient when each thread reads 128 bits (16
bytes) per transaction rather than 32 bits (4 bytes). CUDA's `float4` type
loads four floats in a single memory instruction, quadrupling the effective
memory bandwidth per transaction.

The tile loading step in Kernel 2 is modified to use `float4` loads,
so each thread fetches four elements per global memory instruction instead
of one. This improves memory throughput without changing the tiling logic.

**Why it helps:** GPU memory controllers coalesce loads from adjacent threads
into single wide transactions. `float4` makes this coalescing more explicit
and ensures the hardware's full 128-bit bus width is used on every load.

---

### Kernel 4 — Thread Coarsening
In Kernels 1–3, each thread computes exactly one output element. Thread
coarsening assigns each thread a 2×2 block of output elements instead. The
thread loads the same tile data but accumulates four independent partial sums,
reducing the total number of threads launched and amortizing the overhead of
thread scheduling, index computation, and shared memory synchronization across
more useful arithmetic.

**Why it helps:** launching fewer threads means less scheduler overhead and
more register reuse. The additional arithmetic per thread increases the
arithmetic intensity (ratio of compute to memory operations), making the
kernel more compute-bound and less memory-bound — the regime where GPUs
perform best.

---

## Benchmark Results

Each kernel is measured across three matrix sizes. Timing uses CUDA events
for precise GPU-side measurement, averaged over 100 runs after 10 warm-up
iterations.

| Kernel | 256×256 (ms) | 1024×1024 (ms) | 4096×4096 (ms) | GFLOP/s (4096) | Speedup vs CPU (4096) |
|---|---|---|---|---|---|
| 0 — CPU baseline | 2.9845 | 211.568 | 24475.4 | 5.6 | 1× |
| 1 — GPU naive | 0.0597 | 4.1909 | 305.069 | 450.5 | 80.2× |
| 2 — Shared memory tiling | 0.0501 | 3.1427 | 234.935 | 585.0 | 104.2× |
| 3 — Vectorized loads | 0.0667 | 4.1402 | 299.859 | 458.3 | 81.6× |
| 4 — Thread coarsening | 0.0411 | 2.1095 | 144.259 | 952.7 | 169.7× |
| cuBLAS reference | 0.0186 | 0.4447 | 31.3451 | 4384.7 | 780.8× |

> Results logged to `benchmarks/results.csv`.
> Hardware: NVIDIA Tesla T4 (sm_75), CUDA 13.0, Driver 580.159.04 (AWS g4dn.xlarge).
> cuBLAS is included as an upper-bound reference — not a target to beat.

For Nsight Compute profiler evidence and a kernel-by-kernel discussion of
these numbers (including why vectorized loads underperformed tiling at these
sizes), see [docs/optimization_notes.md](docs/optimization_notes.md).

---

## Key Concepts Explained

**Global memory vs shared memory**
Global memory is the GPU's main DRAM — large (gigabytes) but slow (~400 cycle
latency). Shared memory is a small on-chip scratchpad (~48 KB per block) that
is ~100× faster. The central strategy of GPU optimization is to load data from
global memory into shared memory once and reuse it as many times as possible.

**Memory coalescing**
When 32 threads in a warp access consecutive memory addresses simultaneously,
the GPU hardware combines these into a single wide memory transaction. When
threads access non-consecutive addresses, each access becomes a separate
transaction, wasting bandwidth. Tiling and `float4` loads are both strategies
for ensuring coalesced access patterns.

**Arithmetic intensity**
The ratio of floating-point operations to bytes of memory traffic. GPUs are
most efficient when arithmetic intensity is high — lots of compute per byte
loaded. Naive matmul has low arithmetic intensity (every value is loaded once
and used once). Tiling raises arithmetic intensity by reusing loaded values
TILE_SIZE times from shared memory.

**`__syncthreads()`**
A barrier synchronization instruction that pauses all threads in a block until
every thread has reached that point. Required after loading a tile into shared
memory to ensure all threads see the complete tile before any thread begins
reading from it. A missing `__syncthreads()` is one of the most common sources
of incorrect results in tiled kernels.

**Warp**
The fundamental unit of GPU execution — 32 threads that execute in lockstep.
Thread coarsening reduces the number of active warps, increasing the work
done per warp and improving the ratio of useful arithmetic to scheduling
overhead.

---

## Repo Structure

```
CUDA-Matrix-Multiplication-Optimizer/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── kernels.cuh            ← kernel declarations and shared types
│   └── bench.cuh               ← timing harness declarations
├── src/
│   ├── kernel0_cpu.cpp         ← CPU baseline
│   ├── kernel1_naive.cu        ← GPU naive
│   ├── kernel2_tiled.cu        ← shared memory tiling
│   ├── kernel3_vectorized.cu   ← float4 vectorized loads
│   ├── kernel4_coarsened.cu    ← thread coarsening
│   ├── kernel5_cublas.cu       ← cuBLAS reference
│   └── main.cu                 ← CLI: run, verify, benchmark
├── benchmarks/
│   ├── bench.cu                ← CUDA event timing harness
│   └── results.csv             ← benchmark output
├── tests/
│   └── correctness_test.cu     ← compare all kernels against CPU ground truth
├── docs/
│   ├── setup.md                ← environment and build setup
│   └── optimization_notes.md   ← per-kernel findings and profiler evidence
└── scripts/
    └── check_cuda_env.sh        ← environment sanity check
```

---

## Build & Run

Requirements: an NVIDIA GPU, CUDA Toolkit 12.x+ with `nvcc`, CMake 3.20+, and
a C++17-capable host compiler. Optional: Nsight Compute (`ncu`) for profiling.

```bash
# Sanity-check the environment (nvcc, cmake, GPU)
./scripts/check_cuda_env.sh

# Configure and build. Set CMAKE_CUDA_ARCHITECTURES for your GPU:
#   75 = Turing / RTX 20-series / T4
#   86 = Ampere / RTX 30-series
#   89 = Ada / RTX 40-series
#   90 = Hopper / H100
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j

# Run correctness tests (compare all kernels to CPU ground truth)
ctest --test-dir build --output-on-failure

# Run a specific kernel (0-5) on a 1024x1024 matrix
./build/matmul --kernel 2 --size 1024

# Run the full benchmark suite across all kernels and matrix sizes
./build/matmul --bench --output benchmarks/results.csv

# Profile Kernel 2 with Nsight Compute
ncu --set full ./build/matmul --kernel 2 --size 1024
```

For large benchmark sizes, CPU reference timing and verification can take a
while — for quick iteration on one GPU kernel, use a smaller size first
(e.g. `./build/matmul --kernel 2 --size 256`). See
[docs/setup.md](docs/setup.md) for more environment detail.

---

## Future Work

- **Register blocking** — compute a larger output tile per thread held
  entirely in registers, further reducing shared-memory traffic. This is the
  single biggest remaining gap to cuBLAS (the coarsened kernel reaches ~22%
  of cuBLAS throughput at 4096×4096).
- **Tile size sweep** — measure TILE_SIZE = 8, 16, 32 to quantify the
  shared-memory bank width / register pressure tradeoff directly.
- **Larger coarsening factors** — try 4×4 and 2×4 output tiles per thread and
  characterize the point where register spilling erases the gains.
- **Specialized vectorized loads** — add a fast path for the common case where
  K and N are multiples of 4, removing the per-thread bounds checks that
  currently make Kernel 3 slightly slower than Kernel 2.

---

## Further Reading

- [CUDA C++ Programming Guide — Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [How to Optimize a CUDA Matmul Kernel — Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Programming Massively Parallel Processors — Kirk & Hwu](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) —
  Chapter 4 covers tiled matrix multiplication in depth

---

## License

This project is licensed under the [MIT License](LICENSE).
