//! Safe Rust wrapper around the CUDA matrix-multiplication kernels from the
//! [CUDA-Matrix-Multiplication-Optimizer] project: naive, shared-memory
//! tiled, vectorized, and coarsened. GPU buffers are owned by [`CudaBuffer`]
//! and freed automatically via `Drop`; kernels are dispatched through
//! [`MatMulKernel::launch`]. No `unsafe` is required from callers — every
//! raw pointer and `extern "C"` declaration is contained in the private
//! `ffi` module.
//!
//! [CUDA-Matrix-Multiplication-Optimizer]: https://github.com/czhao-dev/CUDA-Matrix-Multiplication-Optimizer

mod buffer;
mod error;
mod ffi;
mod kernel;

pub use buffer::CudaBuffer;
pub use error::CudaError;
pub use kernel::{KernelVariant, MatMulKernel};
