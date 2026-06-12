# Setup

This project needs an NVIDIA GPU and the CUDA Toolkit. It targets Linux,
WSL2, or a CUDA-enabled remote machine (cloud GPU instances work well).

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x with `nvcc`
- CMake 3.20 or newer
- A C++17-capable host compiler
- Optional: Nsight Compute, available as `ncu`

Check the environment:

```bash
./scripts/check_cuda_env.sh
```

## Build

Pick the CUDA architecture for your GPU:

| GPU family | CMake value |
|---|---:|
| Turing / RTX 20-series | 75 |
| Ampere / RTX 30-series | 86 |
| Ada / RTX 40-series | 89 |
| Hopper / H100 | 90 |

Then build:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

## Validate

```bash
ctest --test-dir build --output-on-failure
./build/matmul --test
```

## Run

```bash
./build/matmul --kernel 2 --size 1024
./build/matmul --bench --output benchmarks/results.csv
```

For large benchmark sizes, CPU reference timing and verification can take a
while. For quick iteration on one GPU kernel, use smaller sizes first:

```bash
./build/matmul --kernel 2 --size 256
```

