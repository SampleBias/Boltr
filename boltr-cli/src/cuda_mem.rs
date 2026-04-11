//! Free VRAM query via `nvidia-smi` (optional; fail-open for policy).

use std::process::Command;

/// Free memory in MiB for the given CUDA ordinal (matches `nvidia-smi -i`).
///
/// Returns `None` if `nvidia-smi` is missing, fails, or output cannot be parsed.
#[must_use]
pub fn free_memory_mb_for_device_index(device_index: usize) -> Option<u64> {
    let idx = device_index.to_string();
    let o = Command::new("nvidia-smi")
        .args([
            "-i",
            &idx,
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !o.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&o.stdout);
    line.lines().next()?.trim().parse().ok()
}
