//! Verifies `CudaBuffer`'s `Drop` impl actually returns memory to the
//! allocator. No `unsafe` appears in this file — only the public safe API
//! is used, demonstrating that correct cleanup doesn't require the caller
//! to touch `cudaFree` directly.
//!
//! Run under `compute-sanitizer --tool memcheck` (or the older
//! `cuda-memcheck`) on a CUDA-capable machine to confirm no leak or
//! double-free is reported — that tool run is the real verification this
//! test is designed to support; the assertions below are a same-process
//! sanity check that can run under plain `cargo test`.
use cuda_matmul::CudaBuffer;

const LEN: usize = 1 << 20; // ~4MB of f32 — large enough that a leak is not noise

#[test]
fn drop_frees_buffer_for_reuse() {
    {
        let _buf = CudaBuffer::<f32>::alloc(LEN).expect("first alloc should succeed");
    } // _buf dropped here — cudaFree runs

    let second = CudaBuffer::<f32>::alloc(LEN);
    assert!(
        second.is_ok(),
        "second same-size allocation failed after the first was dropped: {:?}",
        second.err()
    );
}

#[test]
fn repeated_alloc_drop_does_not_leak() {
    // A single leaked ~4MB allocation can be hard to distinguish from
    // driver-level slack; 100 leaked allocations is an unmissable signal
    // under a leak-checking tool.
    for _ in 0..100 {
        let buf = CudaBuffer::<f32>::alloc(LEN).expect("alloc should succeed in loop");
        drop(buf);
    }
}
