//! Compares every kernel variant's output against a Rust-native CPU
//! reference implementation, independent of the C++ test harness in
//! `tests/correctness_test.cu` at the repo root (this crate's whole point
//! is demonstrating Rust-level safety, not depending on the C++ side for
//! correctness). Dimension cases are reused from that C++ suite for parity.
use cuda_matmul::{CudaBuffer, CudaError, KernelVariant, MatMulKernel};

const VARIANTS: [KernelVariant; 4] = [
    KernelVariant::Naive,
    KernelVariant::Tiled,
    KernelVariant::Vectorized,
    KernelVariant::Coarsened,
];
// kernels.cuh's verify_result defaults to 1e-3, but tests/correctness_test.cu
// actually calls it with 1e-2 — --use_fast_math (enabled here too, in
// build.rs, to match the existing CMake build) trades precision for speed,
// and 1e-3 is tight enough to risk flaky failures on the larger-K cases.
const TOLERANCE: f32 = 1e-2;

/// Deterministic, dependency-free fill in [-1, 1) — a small linear
/// congruential generator seeded differently per matrix so `a` and `b`
/// don't end up identical.
fn fill(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// Row-major CPU reference: `c (MxN) = a (MxK) * b (KxN)`.
fn matmul_cpu_rs(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for kk in 0..k {
            let a_val = a[row * k + kk];
            for col in 0..n {
                c[row * n + col] += a_val * b[kk * n + col];
            }
        }
    }
    c
}

fn check_case(m: usize, n: usize, k: usize) -> Result<(), CudaError> {
    let a_host = fill(m * k, 123);
    let b_host = fill(k * n, 456);
    let expected = matmul_cpu_rs(&a_host, &b_host, m, n, k);

    let mut a = CudaBuffer::<f32>::alloc(m * k)?;
    let mut b = CudaBuffer::<f32>::alloc(k * n)?;
    a.copy_from_host(&a_host)?;
    b.copy_from_host(&b_host)?;

    for variant in VARIANTS {
        let mut c = CudaBuffer::<f32>::alloc(m * n)?;
        MatMulKernel::launch(&a, &b, &mut c, m, n, k, variant)?;

        let mut got = vec![0.0f32; m * n];
        c.copy_to_host(&mut got)?;

        for (idx, (&exp, &act)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                (exp - act).abs() <= TOLERANCE,
                "M={m} N={n} K={k} variant={variant:?} index={idx}: expected {exp}, got {act}"
            );
        }
    }
    Ok(())
}

#[test]
fn case_1x1x1() -> Result<(), CudaError> {
    check_case(1, 1, 1)
}

#[test]
fn case_16x16x16() -> Result<(), CudaError> {
    check_case(16, 16, 16)
}

#[test]
fn case_31x29x17() -> Result<(), CudaError> {
    check_case(31, 29, 17)
}

#[test]
fn case_64x64x64() -> Result<(), CudaError> {
    check_case(64, 64, 64)
}

#[test]
fn case_100x96x80() -> Result<(), CudaError> {
    check_case(100, 96, 80)
}

#[test]
fn case_127x65x33() -> Result<(), CudaError> {
    check_case(127, 65, 33)
}
