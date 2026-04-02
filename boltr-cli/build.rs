//! Embed a LibTorch `RUNPATH` on the `boltr` binary so `libtorch_cpu.so` / `libc10.so` resolve at
//! run time after `torch-sys` `download-libtorch` (or pip PyTorch via `LIBTORCH_USE_PYTORCH=1`).
//!
//! Cargo does not pass `DEP_TCH_LIBTORCH_LIB` from `torch-sys` into this crate's build script, so we
//! read `cargo:libtorch_lib=…` from `target/<profile>/build/torch-sys-*/output`.
//!
//! After incremental builds, multiple `torch-sys-*` directories may exist (pip vs download). We
//! **prefer** the downloaded tree (`…/out/libtorch/…`) so `RUNPATH` matches the link when both are
//! present.

use std::fs;
use std::path::{Path, PathBuf};

fn lib_dirs_from_torch_sys_outputs(build_root: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut download_layout = Vec::new();
    let mut other = Vec::new();
    let Ok(entries) = fs::read_dir(build_root) else {
        return (download_layout, other);
    };
    let mut dirs: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    dirs.sort_by_key(|e| e.file_name());

    for entry in dirs {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("torch-sys-") || !entry.path().is_dir() {
            continue;
        }
        let out_file = entry.path().join("output");
        let Ok(text) = fs::read_to_string(&out_file) else {
            continue;
        };
        println!("cargo:rerun-if-changed={}", out_file.display());
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("cargo:libtorch_lib=") else {
                continue;
            };
            let p = PathBuf::from(rest);
            if !p.is_dir() {
                continue;
            }
            let s = p.to_string_lossy();
            if s.contains("out/libtorch") {
                download_layout.push(p);
            } else {
                other.push(p);
            }
        }
    }
    (download_layout, other)
}

fn main() {
    if std::env::var_os("CARGO_FEATURE_TCH").is_none() {
        return;
    }
    let manifest_dir =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let build_root = manifest_dir
        .join("..")
        .join("target")
        .join(&profile)
        .join("build");

    let (from_dl, other) = lib_dirs_from_torch_sys_outputs(&build_root);
    let lib = from_dl.into_iter().next().or_else(|| other.into_iter().next());

    if let Some(lib) = lib {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    } else {
        println!("cargo:warning=boltr-cli: could not read LibTorch lib dir from torch-sys build output under {} — run `boltr` with `LD_LIBRARY_PATH` set to the PyTorch `lib/` dir, or use `scripts/with_dev_venv.sh`.", build_root.display());
    }
}
