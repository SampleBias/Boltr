//! Shared path resolution for `boltr` binary (doctor, predict).

use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;

/// `site-packages/torch/lib` for the given interpreter (contains `libtorch_cpu.so` / `libtorch_cuda.so`).
///
/// Matches [`scripts/with_dev_venv.sh`](../../scripts/with_dev_venv.sh): CUDA wheels need this on
/// `LD_LIBRARY_PATH` when running `boltr` outside that wrapper.
#[must_use]
pub fn torch_wheel_lib_dir(py: &Path) -> Option<PathBuf> {
    let out = StdCommand::new(py)
        .args([
            "-c",
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        return None;
    }
    let p = PathBuf::from(s);
    if p.is_dir() {
        Some(p)
    } else {
        None
    }
}

/// Prepend the PyTorch wheel `lib/` directory to `LD_LIBRARY_PATH` so a `tch`/`boltr` binary
/// finds `libtorch_cuda.so` (or CPU libs) when the dev venv uses `LIBTORCH_USE_PYTORCH=1`.
#[cfg(unix)]
pub fn prepend_torch_wheel_lib_to_ld_path(cmd: &mut tokio::process::Command, py: &Path) {
    if let Some(lib) = torch_wheel_lib_dir(py) {
        let key = "LD_LIBRARY_PATH";
        let merged = match std::env::var(key) {
            Ok(existing) if !existing.is_empty() => format!("{}:{}", lib.display(), existing),
            _ => lib.display().to_string(),
        };
        cmd.env(key, merged);
    }
}

#[cfg(not(unix))]
pub fn prepend_torch_wheel_lib_to_ld_path(
    _cmd: &mut tokio::process::Command,
    _py: &Path,
) {
}

/// Resolve the `boltr` executable: `BOLTR` env → walk to `target/release/boltr` → `command -v boltr`.
#[must_use]
pub fn resolve_boltr_exe() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BOLTR") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Ok(mut d) = std::env::current_dir() {
        for _ in 0..8 {
            let cand = d.join("target/release/boltr");
            if cand.is_file() {
                return Some(cand);
            }
            if !d.pop() {
                break;
            }
        }
    }
    let out = std::process::Command::new("sh")
        .args(["-c", "command -v boltr 2>/dev/null"])
        .output()
        .ok()?;
    if out.status.success() {
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !line.is_empty() {
            let p = PathBuf::from(line);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}
