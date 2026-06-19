use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Sources compiled into `libcuda_kernels.a`, relative to this crate's
/// manifest directory. The four kernel `.cu` files live in the parent
/// repo's `src/` and are compiled in place — they are not copied into this
/// crate, so this crate has no kernel logic of its own to keep in sync.
const CUDA_SOURCES: &[&str] = &[
    "../src/kernel1_naive.cu",
    "../src/kernel2_tiled.cu",
    "../src/kernel3_vectorized.cu",
    "../src/kernel4_coarsened.cu",
    "cuda/cuda_bridge.cu",
];

fn cuda_root() -> PathBuf {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(path) = env::var(var) {
            return PathBuf::from(path);
        }
    }
    PathBuf::from("/usr/local/cuda")
}

fn find_nvcc(cuda_root: &Path) -> PathBuf {
    let candidate = cuda_root.join("bin").join("nvcc");
    if candidate.exists() {
        candidate
    } else {
        // Fall back to PATH lookup for environments (containers, CI) where
        // nvcc is installed without a conventional CUDA_HOME layout.
        PathBuf::from("nvcc")
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let cuda_root = cuda_root();
    let nvcc = find_nvcc(&cuda_root);
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

    let repo_include = manifest_dir.join("../include");
    let cuda_include = cuda_root.join("include");

    let mut object_files = Vec::with_capacity(CUDA_SOURCES.len());

    for relative_src in CUDA_SOURCES {
        let src = manifest_dir.join(relative_src);
        assert!(
            src.exists(),
            "expected CUDA source at {} (relative to rust/build.rs) but it does not exist \
             — is this crate still located at <repo-root>/rust/?",
            src.display()
        );

        let file_name = src
            .file_stem()
            .expect("CUDA source path has no file stem")
            .to_string_lossy()
            .into_owned();
        let obj = out_dir.join(format!("{file_name}.o"));

        let status = Command::new(&nvcc)
            .args([
                "-c",
                src.to_str().expect("CUDA source path is not valid UTF-8"),
                "-o",
                obj.to_str().expect("object path is not valid UTF-8"),
            ])
            .arg(format!("-I{}", repo_include.display()))
            .arg(format!("-I{}", cuda_include.display()))
            .arg(format!("-arch={cuda_arch}"))
            .args(["-O2", "--use_fast_math"])
            .args(["--compiler-options", "-fPIC"])
            .status()
            .unwrap_or_else(|err| {
                panic!(
                    "failed to run nvcc at {} — install the CUDA Toolkit or set \
                     CUDA_HOME/CUDA_PATH (see scripts/check_cuda_env.sh in the repo root): {err}",
                    nvcc.display()
                )
            });
        assert!(status.success(), "nvcc failed compiling {}", src.display());

        object_files.push(obj);
    }

    let static_lib = out_dir.join("libcuda_kernels.a");
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&static_lib)
        .args(&object_files)
        .status()
        .expect("failed to run `ar` — is binutils installed?");
    assert!(ar_status.success(), "ar failed to archive {}", static_lib.display());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=cuda_kernels");

    let cuda_lib_dir = ["lib64", "lib"]
        .iter()
        .map(|dir| cuda_root.join(dir))
        .find(|path| path.exists())
        .unwrap_or_else(|| cuda_root.join("lib64"));
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");

    for relative_src in CUDA_SOURCES {
        println!("cargo:rerun-if-changed={}", manifest_dir.join(relative_src).display());
    }
    println!("cargo:rerun-if-changed={}", repo_include.join("kernels.cuh").display());
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
