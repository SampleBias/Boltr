//! Shared path resolution for `boltr` binary (doctor, predict).

use std::path::PathBuf;

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
