use cuda_matmul::{CudaBuffer, KernelVariant, MatMulKernel};

fn main() -> Result<(), cuda_matmul::CudaError> {
    let m = 1024usize;
    let n = 1024usize;
    let k = 1024usize;

    // Fill host matrices.
    let a_host: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
    let b_host: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
    let mut c_host = vec![0.0f32; m * n];

    // Allocate GPU buffers — freed automatically when they go out of scope.
    let mut a = CudaBuffer::<f32>::alloc(m * k)?;
    let mut b = CudaBuffer::<f32>::alloc(k * n)?;
    let mut c = CudaBuffer::<f32>::alloc(m * n)?;

    // Copy host -> device.
    a.copy_from_host(&a_host)?;
    b.copy_from_host(&b_host)?;

    // Launch kernel — caller chooses the variant, no raw pointers required.
    MatMulKernel::launch(&a, &b, &mut c, m, n, k, KernelVariant::Tiled)?;

    // Copy device -> host.
    c.copy_to_host(&mut c_host)?;

    println!("C[0][0] = {:.4}", c_host[0]);

    Ok(())
    // a, b, c go out of scope here — cudaFree called automatically.
}
