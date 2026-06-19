# CUDA Matrix Multiplication Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.x%2B-76B900.svg)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![Rust](https://img.shields.io/badge/Rust-stable-CE422B.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)

> A step-by-step optimization of matrix multiplication on the GPU — from a
> naive CPU baseline to a high-performance tiled CUDA kernel — plus a safe
> Rust wrapper that applies Rust's ownership model to GPU memory management,
> with measured speedups and zero-overhead verification at every stage.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The Five CUDA Kernels](#the-five-cuda-kernels)
- [CUDA Benchmark Results](#cuda-benchmark-results)
- [Safe Rust Wrapper](#safe-rust-wrapper)
- [Key Concepts Explained](#key-concepts-explained)
- [Repo Structure](#repo-structure)
- [Build & Run](#build--run)
- [Future Work](#future-work)
- [Why Wrap CUDA in Rust?](#why-wrap-cuda-in-rust)
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

The repo has two parts:

1. **The CUDA kernels** (`src/`, `include/`) — naive → shared-memory tiled →
   vectorized → thread-coarsened, each one fixing a specific bottleneck in
   the previous version, benchmarked against a CPU baseline and cuBLAS.
2. **A safe Rust wrapper** (`rust/`) — a `cuda-matmul` crate that wraps those
   same kernels (unchanged) in an ownership-based API, so that allocating
   GPU memory, launching a kernel, and freeing the buffer require zero
   `unsafe` code from the caller and zero runtime overhead versus calling
   the kernels directly from C++.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│   Safe Rust API (rust/, public)                        │
│   CudaBuffer<T>, MatMulKernel, CudaError               │
│   No unsafe code required from callers                 │
├────────────────────────────────────────────────────────┤
│   Unsafe FFI Bindings (rust/src/ffi.rs, internal)      │
│   extern "C" declarations, raw pointer passing         │
│   Rust unsafe{} blocks, manually upheld invariants     │
├────────────────────────────────────────────────────────┤
│   CUDA C++ Kernels (src/, include/)                    │
│   kernel1_naive.cu, kernel2_tiled.cu, ...              │
│   Compiled by nvcc, linked as a static library         │
└────────────────────────────────────────────────────────┘
```

The CUDA kernels are the same compiled code whether they're invoked from the
C++ `matmul` binary or from Rust — the Rust crate adds a compile-time safety
layer on top, not a reimplementation underneath.

---

## The Five CUDA Kernels

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

## CUDA Benchmark Results

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

## Safe Rust Wrapper

The `rust/` directory contains `cuda-matmul`, a Rust crate that wraps the four
GPU kernels above in a safe API. The kernels themselves are unchanged; the
crate builds the abstraction layer that sits between them and the caller,
guaranteeing at compile time that GPU buffers cannot be used after they are
freed, cannot be written by two owners simultaneously, and are never leaked.

**Why this matters.** Raw CUDA C++ gives you full control but no safety
guarantees: you can `cudaFree` a pointer you still hold a reference to, forget
to synchronize before reading results back to the host, or pass a device
pointer to a host-only function. These are classes of bugs Rust's type system
eliminates by construction — the same architecture used by Hugging Face's
[candle](https://github.com/huggingface/candle) ML framework and the
[cudarc](https://github.com/coreylowman/cudarc) crate: unsafe, hardware-facing
code wrapped in a safe host-language abstraction so application code never has
to touch a raw pointer.

### Core safety guarantees

| Bug class | C++ (raw CUDA) | This Rust wrapper |
|---|---|---|
| Use after `cudaFree` | Possible — no compiler check | Impossible — `Drop` frees, compiler rejects any further use |
| Double free of device buffer | Possible — call `cudaFree` twice | Impossible — ownership ensures `Drop` runs exactly once |
| Data race: two threads writing | Possible | Impossible — `&mut CudaBuffer<T>` requires exclusive ownership |
| Read before H→D copy finishes | Possible — silent wrong result | Caught — `copy_from_host` returns `Result`, must be checked |
| Leak on early return | Common — must remember every `cudaFree` path | Impossible — `Drop` runs on any exit path, including `?` propagation |
| Wrong element type (float/int) | Silent — pointer cast compiles | Caught at compile time — `CudaBuffer<f32>` vs `CudaBuffer<i32>` |

### Public API

No `unsafe` keyword anywhere in application code (`rust/examples/basic.rs`):

```rust
use cuda_matmul::{CudaBuffer, KernelVariant, MatMulKernel};

fn main() -> Result<(), cuda_matmul::CudaError> {
    let m = 1024usize;
    let n = 1024usize;
    let k = 1024usize;

    // Fill host matrices.
    let a_host: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
    let mut c_host = vec![0.0f32; m * n];

    // Allocate GPU buffers — freed automatically when they go out of scope.
    let mut a = CudaBuffer::<f32>::alloc(m * k)?;
    let mut b = CudaBuffer::<f32>::alloc(k * n)?;
    let mut c = CudaBuffer::<f32>::alloc(m * n)?;

    // Copy host -> device.
    a.copy_from_host(&a_host)?;
    b.copy_from_host(&b_host)?;

    // Launch kernel — caller chooses the variant, no raw pointers required.
    MatMulKernel::launch(&a, &b, &mut c, m, n, k, KernelVariant::Tiled)?;

    // Copy device -> host.
    c.copy_to_host(&mut c_host)?;

    println!("C[0][0] = {:.4}", c_host[0]);

    Ok(())
    // a, b, c go out of scope here — cudaFree called automatically.
}
```

Contrast this with the equivalent raw C++ call, which requires explicit
`cudaMalloc`, `cudaMemcpy`, manual kernel launch syntax, and three separate
`cudaFree` calls on every return path.

### Design notes

**`CudaBuffer<T>` — ownership applied to GPU memory.** `T` is bounded by
`Copy`, not left fully generic: `copy_from_host`/`copy_to_host` move bytes via
a raw `cudaMemcpy`, and a type with a custom `Drop` impl copied that way would
have its destructor invariants silently violated. `Copy` and `Drop` are
mutually exclusive in Rust, so this bound rules out that bug class at the
type level — `PhantomData<T>` keeps the struct generic over the element type
without storing one, which is what correct drop-check and variance reasoning
requires.

```rust
// rust/src/buffer.rs
pub struct CudaBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        // Safety: `self.ptr` was allocated by `cudaMalloc` in `alloc` and
        // is freed exactly once because Rust's ownership model guarantees
        // `drop` runs at most once per value.
        unsafe { ffi::cudaFree(self.ptr.cast()); }
    }
}
```

**The `unsafe`/safe boundary.** Every interaction with the CUDA C API lives
inside `buffer.rs`/`kernel.rs` (which hide the pointer behind a safe API) or
`ffi.rs` (which declares the raw `extern "C"` bindings). The rest of the
crate is entirely safe Rust:

```rust
// rust/src/ffi.rs — the one place unsafe FFI declarations live
extern "C" {
    pub(crate) fn cuda_matmul_launch_naive(
        a: *const c_float, b: *const c_float, c: *mut c_float,
        m: c_int, n: c_int, k: c_int,
    ) -> c_int;
    // ...tiled, vectorized, coarsened — same shape
}
```

```rust
// rust/src/kernel.rs — safe wrapper around ffi.rs
pub fn launch(
    a: &CudaBuffer<f32>, b: &CudaBuffer<f32>, c: &mut CudaBuffer<f32>,
    m: usize, n: usize, k: usize, variant: KernelVariant,
) -> Result<(), CudaError> {
    check_dims(a, b, c, m, n, k)?;          // catch undersized buffers before FFI
    let code = unsafe {
        // Safety: a/b/c are valid device allocations, lengths checked
        // above, c is exclusively borrowed (&mut) so no Rust-level race.
        match variant {
            KernelVariant::Naive => ffi::cuda_matmul_launch_naive(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i32, n as i32, k as i32),
            // ...tiled, vectorized, coarsened
        }
    };
    check(code, CudaError::LaunchFailed)?;
    check(unsafe { ffi::cudaDeviceSynchronize() }, CudaError::SyncFailed)
}
```

`KernelVariant` is a plain unit-only enum (`Naive`/`Tiled`/`Vectorized`/
`Coarsened`) rather than carrying config like `Tiled { tile_size }` — the
underlying kernels hardcode tile size, vectorization width, and coarsening
factor as `constexpr`, so there's nothing for a payload field to configure at
runtime. `cuda_bridge.cu` thin-wraps the existing C++ `launch_naive`/
`launch_tiled`/`launch_vectorized`/`launch_coarsened` functions from
`kernels.cuh` rather than reimplementing grid/block setup — the same launch
math runs either way.

**`build.rs` — compiling CUDA from Cargo.** Cargo runs `build.rs` before
compiling the crate, which is how a Rust build can invoke `nvcc` and link the
result:

```rust
// rust/build.rs (abbreviated)
let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
// compiles ../src/kernel{1,2,3,4}_*.cu + cuda/cuda_bridge.cu in place —
// no copies — into OUT_DIR, with `-arch=<cuda_arch>` and the same
// `--use_fast_math` flag the top-level CMake build uses.
println!("cargo:rustc-link-lib=static=cuda_kernels");
println!("cargo:rustc-link-lib=dylib=cudart");
println!("cargo:rerun-if-env-changed=CUDA_ARCH"); // override per-GPU, see Build & Run
```

**Error handling with `?` propagation.** Every fallible call returns
`Result<T, CudaError>`, carrying the raw `cudaError_t` code rather than
re-encoding the full enum (which would need to track every CUDA Toolkit
version):

```rust
// rust/src/error.rs
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("cudaMalloc failed (cudaError_t={0})")]
    AllocationFailed(i32),
    #[error("cudaMemcpy failed (cudaError_t={0})")]
    MemcpyFailed(i32),
    #[error("kernel launch failed (cudaError_t={0})")]
    LaunchFailed(i32),
    #[error("cudaDeviceSynchronize failed (cudaError_t={0})")]
    SyncFailed(i32),
    #[error("buffer length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}
```

A `cudaFree` leak on an early error return is impossible — `Drop` runs
regardless of which `?` caused the function to exit.

### Verification & benchmarks

Built and verified end-to-end on an AWS `g4dn.xlarge` (Tesla T4, compute
capability 7.5, `CUDA_ARCH=sm_75`), CUDA 12.9:

| Check | Result |
|---|---|
| `cargo build` / `cargo build --release` | Clean, zero warnings |
| `cargo test` (6 dimension cases × 4 kernel variants vs. a Rust-native CPU reference) | 8/8 passed |
| `compute-sanitizer --tool memcheck --leak-check full` (alloc/drop stress test) | 0 errors, 0 bytes leaked |
| `cargo clippy --all-targets -- -D warnings` | Clean |

Throughput across matrix sizes (`cargo run --release --example benchmark`),
2 warmup + 10 timed launches:

| Kernel | 256³ (ms) | 256³ (GFLOP/s) | 1024³ (ms) | 1024³ (GFLOP/s) | 1000³ (ms) | 1000³ (GFLOP/s) |
|---|---|---|---|---|---|---|
| Naive | 0.0648 | 517.4 | 3.664 | 586.1 | 3.048 | 656.2 |
| Tiled | 0.0454 | 738.6 | 2.734 | 785.5 | 2.471 | 809.5 |
| Vectorized | 0.0575 | 584.0 | 3.503 | 613.1 | 3.170 | 630.8 |
| Coarsened | 0.0292 | 1148.3 | 1.339 | 1603.3 | 1.240 | 1613.0 |

Wrapper overhead vs. calling the C++ kernels directly, same hardware,
1024×1024×1024:

| Kernel | C++ direct (ms) | Rust wrapper (ms) | Overhead |
|---|---|---|---|
| Naive | 3.598 | 3.661 | +1.7% |
| Tiled | 2.730 | 2.731 | ~0% |
| Vectorized | 3.759 | 3.495 | −7.0% |
| Coarsened | 1.922 | 1.336 | −30.5% |

> Naive/Tiled/Vectorized land within run-to-run noise of each other,
> consistent with the wrapper adding no real per-launch cost — it's a
> pointer pass-through plus a `match` and a `cudaDeviceSynchronize` call.
> Coarsened's gap is the *opposite* sign you'd expect from "the wrapper is
> slower," and is a measurement-methodology artifact: the C++ harness
> (`benchmarks/bench.cu`) times a batch of back-to-back launches with one
> `cudaEvent` timestamp at the end, while the Rust harness
> (`examples/benchmark.rs`) times each launch via host-side
> `std::time::Instant`, synchronizing after every single call — a
> deliberate safety property of `MatMulKernel::launch`. For a 3–4ms kernel
> that per-call overhead is negligible; for Coarsened's sub-2ms execution
> it's large enough, on a shared cloud GPU, to be dominated by scheduling
> noise rather than by anything either implementation does differently.

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
├── scripts/
│   └── check_cuda_env.sh       ← environment sanity check
└── rust/                       ← safe Rust wrapper (cuda-matmul crate)
    ├── Cargo.toml
    ├── build.rs                ← compiles the kernels above + cuda_bridge.cu
    ├── cuda/
    │   └── cuda_bridge.cu      ← extern "C" entry points for Rust FFI
    ├── src/
    │   ├── lib.rs              ← public API re-exports
    │   ├── buffer.rs           ← CudaBuffer<T>: alloc, drop, copy
    │   ├── kernel.rs           ← MatMulKernel::launch, KernelVariant enum
    │   ├── error.rs            ← CudaError enum
    │   └── ffi.rs              ← extern "C" declarations (unsafe, private)
    ├── examples/
    │   ├── basic.rs            ← the demo from the Public API section above
    │   └── benchmark.rs        ← times all four variants, prints a table
    └── tests/
        ├── correctness.rs      ← output matches a Rust CPU reference to 1e-2
        └── drop_test.rs        ← verify no leak/double-free under compute-sanitizer
```

---

## Build & Run

Requirements: an NVIDIA GPU, CUDA Toolkit 12.x+ with `nvcc`, CMake 3.20+, and
a C++17-capable host compiler. Optional: Nsight Compute (`ncu`) for profiling,
Rust (stable, via [rustup](https://rustup.rs)) for the wrapper crate.

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

### Rust wrapper

```bash
cd rust

# CUDA_ARCH must match your GPU's compute capability (see the table above,
# e.g. sm_75 for a T4); build.rs defaults to sm_86 if unset.
CUDA_ARCH=sm_75 cargo build --release

# Run the public-API demo
CUDA_ARCH=sm_75 cargo run --release --example basic

# Run correctness + drop tests
CUDA_ARCH=sm_75 cargo test

# Run the throughput benchmark
CUDA_ARCH=sm_75 cargo run --release --example benchmark

# Check for GPU memory leaks/errors (cuda-memcheck is deprecated; use compute-sanitizer)
compute-sanitizer --tool memcheck --leak-check full cargo test

# Lint
cargo clippy --all-targets -- -D warnings
```

`build.rs` compiles `../src/kernel{1,2,3,4}_*.cu` and `cuda/cuda_bridge.cu`
directly — no files are copied into `rust/`, so the Rust crate always builds
against whatever kernel code currently lives in `src/`.

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
- **Rust: stream-based async launches** — expose `cudaStream_t` so multiple
  `MatMulKernel::launch` calls can overlap instead of always synchronizing
  the default stream.
- **Rust: apples-to-apples benchmarking** — make `examples/benchmark.rs` use
  CUDA-event timing (matching `benchmarks/bench.cu`'s methodology) instead of
  host-side `Instant`, to remove the measurement-methodology caveat on the
  Coarsened row above.

---

## Why Wrap CUDA in Rust?

The GPU kernels are unchanged — they still run at full CUDA speed. The Rust
layer adds compile-time guarantees: GPU buffers can't be used after they're
freed, can't be leaked on early return, and can't be written by two owners at
once. None of that is checkable at runtime; it's enforced by Rust's ownership
system at compile time, for zero added cost.

This is also the same reason Hugging Face wrote
[Candle](https://github.com/huggingface/candle) and
[cudarc](https://github.com/coreylowman/cudarc) in Rust rather than C++: the
GPU computation stays in CUDA, but the host-side orchestration — allocating
buffers, scheduling work, managing lifetimes across transfers — is exactly
the kind of systems code where Rust's ownership model eliminates whole bug
classes. Production ML inference stacks have this architecture; `rust/` is a
small, complete demonstration of it.

---

## Further Reading

- [CUDA C++ Programming Guide — Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
- [How to Optimize a CUDA Matmul Kernel — Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Programming Massively Parallel Processors — Kirk & Hwu](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) —
  Chapter 4 covers tiled matrix multiplication in depth
- [The Rustonomicon — Unsafe Rust](https://doc.rust-lang.org/nomicon/) —
  the definitive guide to writing correct `unsafe` Rust; the sections on FFI
  and ownership semantics are directly relevant to `rust/`
- [cudarc crate](https://github.com/coreylowman/cudarc) — the production Rust
  CUDA abstraction library; compare its `CudaDevice`/`CudaSlice` API to this
  crate's `CudaBuffer`/`MatMulKernel` to see how the same concepts scale up
- [Hugging Face Candle](https://github.com/huggingface/candle) — a full ML
  framework built on this exact architecture: Rust host code, CUDA kernels
  for the hot paths, a safe typed API for callers
- [Rust FFI Omnibus](https://jakegoulding.com/rust-ffi-omnibus/) — worked
  examples of every common FFI pattern between Rust and C
- [build.rs documentation](https://doc.rust-lang.org/cargo/reference/build-scripts.html) —
  the full reference for `build.rs` capabilities used in `rust/build.rs`

---

## License

This project is licensed under the [MIT License](LICENSE).
