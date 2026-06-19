use crate::buffer::CudaBuffer;
use crate::error::{check, CudaError};
use crate::ffi;

/// Which compiled CUDA kernel to launch.
///
/// Unlike a generic-config sketch (`Tiled { tile_size: usize }`,
/// `Coarsened { factor: usize }`), this is a plain unit-only enum: the tile
/// size (16), vectorization width (4), and coarsening factor (2) are
/// `constexpr` inside the underlying `.cu` files, not parameters accepted
/// by `launch_tiled`/`launch_vectorized`/`launch_coarsened` in `kernels.cuh`.
/// There is nothing for a payload field to configure at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    Naive,
    Tiled,
    Vectorized,
    Coarsened,
}

/// Dispatches one of the four CUDA matrix-multiplication kernels.
pub struct MatMulKernel;

impl MatMulKernel {
    /// Computes `c = a * b` for row-major matrices `a` (M×K), `b` (K×N), and
    /// `c` (M×N), using the given kernel `variant`. Synchronizes the device
    /// before returning, so `c.copy_to_host(..)` is safe to call
    /// immediately afterward.
    #[must_use = "ignoring this discards a potential CUDA launch/sync error"]
    pub fn launch(
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        c: &mut CudaBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        variant: KernelVariant,
    ) -> Result<(), CudaError> {
        check_dims(a, b, c, m, n, k)?;

        let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);
        let code = unsafe {
            // Safety: `a`, `b`, `c` are valid device allocations (the
            // `CudaBuffer` invariant); their lengths were just checked
            // against m*k, k*n, m*n above, so the kernel cannot read or
            // write out of bounds; `c` is exclusively borrowed (`&mut`),
            // so no other Rust reference can race the kernel's writes to it.
            match variant {
                KernelVariant::Naive => {
                    ffi::cuda_matmul_launch_naive(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m_i, n_i, k_i)
                }
                KernelVariant::Tiled => {
                    ffi::cuda_matmul_launch_tiled(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m_i, n_i, k_i)
                }
                KernelVariant::Vectorized => ffi::cuda_matmul_launch_vectorized(
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    m_i,
                    n_i,
                    k_i,
                ),
                KernelVariant::Coarsened => ffi::cuda_matmul_launch_coarsened(
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    m_i,
                    n_i,
                    k_i,
                ),
            }
        };
        check(code, CudaError::LaunchFailed)?;

        // Safety: no arguments; synchronizes the default stream that every
        // launch above used.
        let sync_code = unsafe { ffi::cudaDeviceSynchronize() };
        check(sync_code, CudaError::SyncFailed)
    }
}

/// Cross-validates caller-given dimensions against the caller-given
/// buffers before any FFI call. The CUDA side trusts `m`/`n`/`k` blindly —
/// an undersized buffer here would otherwise cause a silent out-of-bounds
/// device read or write.
fn check_dims(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    c: &CudaBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    if a.len() != m * k {
        return Err(CudaError::LengthMismatch {
            expected: m * k,
            actual: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(CudaError::LengthMismatch {
            expected: k * n,
            actual: b.len(),
        });
    }
    if c.len() != m * n {
        return Err(CudaError::LengthMismatch {
            expected: m * n,
            actual: c.len(),
        });
    }
    Ok(())
}
