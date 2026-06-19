use crate::error::{check, CudaError};
use crate::ffi;
use std::marker::PhantomData;
use std::os::raw::c_void;

/// An owned CUDA device allocation of `len` elements of type `T`.
///
/// `T: Copy` is required even though the README sketch this crate follows
/// shows an unconstrained `CudaBuffer<T>`: [`copy_from_host`](Self::copy_from_host)
/// and [`copy_to_host`](Self::copy_to_host) move bytes via `cudaMemcpy`, a
/// raw byte-for-byte copy that knows nothing about ownership. A type with a
/// custom `Drop` impl copied this way would have its destructor invariants
/// silently violated (e.g. two host values appearing to share one owned
/// resource). `Copy` and `Drop` are mutually exclusive in Rust, so bounding
/// on `Copy` rules out that bug class at the type level — consistent with
/// the rest of this crate's design goal of making GPU misuse a compile error.
pub struct CudaBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> CudaBuffer<T> {
    /// Allocates device memory for `len` elements of `T`.
    #[must_use = "discarding this drops the buffer immediately, freeing the GPU allocation"]
    pub fn alloc(len: usize) -> Result<Self, CudaError> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("buffer size overflow");
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // Safety: `ptr` is a valid `*mut *mut c_void` to a local variable,
        // `bytes` is the exact size requested.
        let code = unsafe { ffi::cudaMalloc(&mut ptr, bytes) };
        check(code, CudaError::AllocationFailed)?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
            _marker: PhantomData,
        })
    }

    /// Number of `T` elements this buffer holds.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies `src` from the host into this device buffer. Fails with
    /// [`CudaError::LengthMismatch`] if `src.len() != self.len()`.
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), CudaError> {
        if src.len() != self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                actual: src.len(),
            });
        }
        let bytes = self.len * std::mem::size_of::<T>();
        // Safety: `self.ptr` is a valid device allocation of `bytes` size
        // (invariant established by `alloc`), `src` is a valid host slice
        // of the same byte length (checked above), and the two regions
        // cannot overlap (one is host memory, one is device memory).
        let code = unsafe {
            ffi::cudaMemcpy(
                self.ptr.cast(),
                src.as_ptr().cast(),
                bytes,
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        check(code, CudaError::MemcpyFailed)
    }

    /// Copies this device buffer back to the host into `dst`. Fails with
    /// [`CudaError::LengthMismatch`] if `dst.len() != self.len()`.
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), CudaError> {
        if dst.len() != self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                actual: dst.len(),
            });
        }
        let bytes = self.len * std::mem::size_of::<T>();
        // Safety: same reasoning as `copy_from_host`, with host and device
        // roles reversed.
        let code = unsafe {
            ffi::cudaMemcpy(
                dst.as_mut_ptr().cast(),
                self.ptr.cast(),
                bytes,
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        check(code, CudaError::MemcpyFailed)
    }

    /// Raw device pointer for passing into FFI kernel launches. Stays
    /// `pub(crate)` — never exposed to callers outside this crate.
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: Copy> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        // Safety: `self.ptr` was allocated by `cudaMalloc` in `alloc` and
        // is freed exactly once because Rust's ownership model guarantees
        // `drop` runs at most once per value.
        unsafe {
            ffi::cudaFree(self.ptr.cast());
        }
    }
}
