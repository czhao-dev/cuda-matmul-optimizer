/// Errors returned by this crate. The CUDA-runtime-backed variants carry the
/// raw `cudaError_t` code (an `i32`, with `0` meaning `cudaSuccess`) rather
/// than re-encoding every possible CUDA error as a Rust enum — the integer
/// is enough to diagnose a failure, and duplicating the full `cudaError_t`
/// enum would need to be kept in sync with the CUDA Toolkit version.
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

/// Maps a raw `cudaError_t` code into a `Result`, using `ctor` to build the
/// appropriate [`CudaError`] variant on failure. `code == 0` (`cudaSuccess`)
/// is the only success value.
pub(crate) fn check(code: i32, ctor: impl FnOnce(i32) -> CudaError) -> Result<(), CudaError> {
    if code == 0 {
        Ok(())
    } else {
        Err(ctor(code))
    }
}
