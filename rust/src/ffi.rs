//! Raw `extern "C"` bindings — the one place `unsafe` FFI declarations
//! live. Never exposed outside the crate; `buffer.rs` and `kernel.rs` are
//! the only callers.
use std::os::raw::{c_float, c_int, c_void};

/// `cudaMemcpyKind` values from `cuda_runtime_api.h`, limited to the two
/// directions this crate actually uses.
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

extern "C" {
    pub(crate) fn cuda_matmul_launch_naive(
        a: *const c_float,
        b: *const c_float,
        c: *mut c_float,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    pub(crate) fn cuda_matmul_launch_tiled(
        a: *const c_float,
        b: *const c_float,
        c: *mut c_float,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    pub(crate) fn cuda_matmul_launch_vectorized(
        a: *const c_float,
        b: *const c_float,
        c: *mut c_float,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    pub(crate) fn cuda_matmul_launch_coarsened(
        a: *const c_float,
        b: *const c_float,
        c: *mut c_float,
        m: c_int,
        n: c_int,
        k: c_int,
    ) -> c_int;

    // CUDA runtime function names are not snake_case; allow that rather
    // than renaming away from the names documented in the CUDA Toolkit.
    #[allow(non_snake_case)]
    pub(crate) fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int;
    #[allow(non_snake_case)]
    pub(crate) fn cudaFree(dev_ptr: *mut c_void) -> c_int;
    #[allow(non_snake_case)]
    pub(crate) fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    #[allow(non_snake_case)]
    pub(crate) fn cudaDeviceSynchronize() -> c_int;
}
