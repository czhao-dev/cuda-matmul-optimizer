//! Times all four kernel variants at a few sizes and prints a table with
//! implied GFLOP/s. Scoped to Rust-internal timing only — comparing against
//! the existing C++ `matmul` binary directly would require that binary to
//! already be built and its stdout to be machine-parseable, neither of
//! which is guaranteed today; that comparison is a follow-up, not v1.
use cuda_matmul::{CudaBuffer, CudaError, KernelVariant, MatMulKernel};
use std::time::Instant;

const SIZES: [usize; 3] = [256, 1024, 1000]; // 1000 exercises non-divisible-by-16 dims
const WARMUP_ITERS: u32 = 2;
const TIMED_ITERS: u32 = 10;

fn variant_name(variant: KernelVariant) -> &'static str {
    match variant {
        KernelVariant::Naive => "Naive",
        KernelVariant::Tiled => "Tiled",
        KernelVariant::Vectorized => "Vectorized",
        KernelVariant::Coarsened => "Coarsened",
    }
}

fn bench_one(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    c: &mut CudaBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    variant: KernelVariant,
) -> Result<f64, CudaError> {
    for _ in 0..WARMUP_ITERS {
        MatMulKernel::launch(a, b, c, m, n, k, variant)?;
    }

    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        MatMulKernel::launch(a, b, c, m, n, k, variant)?;
    }
    let elapsed = start.elapsed();

    Ok(elapsed.as_secs_f64() / f64::from(TIMED_ITERS) * 1000.0)
}

fn main() -> Result<(), CudaError> {
    println!("{:<12} {:>6} {:>6} {:>6} {:>12} {:>10}", "Kernel", "M", "N", "K", "ms", "GFLOP/s");

    for &size in &SIZES {
        let (m, n, k) = (size, size, size);
        let a_host: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();

        let mut a = CudaBuffer::<f32>::alloc(m * k)?;
        let mut b = CudaBuffer::<f32>::alloc(k * n)?;
        let mut c = CudaBuffer::<f32>::alloc(m * n)?;
        a.copy_from_host(&a_host)?;
        b.copy_from_host(&b_host)?;

        for variant in [
            KernelVariant::Naive,
            KernelVariant::Tiled,
            KernelVariant::Vectorized,
            KernelVariant::Coarsened,
        ] {
            let ms = bench_one(&a, &b, &mut c, m, n, k, variant)?;
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            let gflops = flops / (ms / 1000.0) / 1e9;
            println!(
                "{:<12} {:>6} {:>6} {:>6} {:>12.4} {:>10.2}",
                variant_name(variant),
                m,
                n,
                k,
                ms,
                gflops
            );
        }
    }

    Ok(())
}
