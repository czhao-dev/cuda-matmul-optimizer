#!/usr/bin/env bash
set -euo pipefail

echo "CUDA matrix multiplication environment check"
echo

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc: $(nvcc --version | tail -n 1)"
else
  echo "nvcc: not found"
  echo "Install the CUDA Toolkit on a CUDA-capable Linux/WSL/remote machine."
fi

if command -v cmake >/dev/null 2>&1; then
  echo "cmake: $(cmake --version | head -n 1)"
else
  echo "cmake: not found"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
else
  echo "nvidia-smi: not found"
fi

echo
echo "Build example:"
echo "  cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86"
echo "  cmake --build build -j"

