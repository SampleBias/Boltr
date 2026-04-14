//! Free VRAM probing (LibTorch builds only) and single-GPU detection for `--device auto`.
//!
//! With `--features tch`, auto mode can require minimum free VRAM (`BOLTR_AUTO_MIN_FREE_VRAM_MB`)
//! before choosing CUDA; if the probe fails, CUDA is still used when LibTorch reports a GPU (**fail-open**).
//! Single-GPU Boltz defaults live in [`crate::preprocess_cmd::auto_default_boltz_cpu_for_memory`].

use std::process::Command;

/// Core decision for [`auto_vram_allows_cuda`] and unit tests.
#[cfg(any(feature = "tch", test))]
pub(crate) fn auto_vram_decision(min_mb: Option<u64>, free_bytes: Option<u64>) -> bool {
    let Some(min_mb) = min_mb else {
        return true;
    };
    let Some(free) = free_bytes else {
        return true;
    };
    let need = min_mb.saturating_mul(1024 * 1024);
    free >= need
}

/// Parse `BOLTR_AUTO_MIN_FREE_VRAM_MB`: minimum free VRAM in MiB for auto mode to use CUDA.
/// If unset or empty, no VRAM check is applied (only `cuda_is_available()` matters).
#[cfg(feature = "tch")]
pub fn min_free_vram_mb_from_env() -> Option<u64> {
    match std::env::var("BOLTR_AUTO_MIN_FREE_VRAM_MB") {
        Ok(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<u64>().ok()
            }
        }
        Err(_) => None,
    }
}

/// Whether `--device auto` should pick CUDA given LibTorch already reports a GPU.
/// Respects `BOLTR_AUTO_MIN_FREE_VRAM_MB` when set; if the probe fails, returns `true` (fail-open).
#[cfg(feature = "tch")]
pub fn auto_vram_allows_cuda() -> bool {
    let min_mb = min_free_vram_mb_from_env();
    let free = query_free_vram_bytes();
    let ok = auto_vram_decision(min_mb, free);
    if !ok {
        tracing::info!(
            free_mib = free.map(|b| b / (1024 * 1024)),
            min_mib = min_mb,
            "device auto: free VRAM below BOLTR_AUTO_MIN_FREE_VRAM_MB; using cpu"
        );
    } else if min_mb.is_some() && free.is_none() {
        tracing::debug!(
            "BOLTR_AUTO_MIN_FREE_VRAM_MB set but free VRAM probe failed; keeping cuda if available (fail-open)"
        );
    }
    ok
}

/// Best-effort free VRAM for the current process's CUDA device 0 (respects `CUDA_VISIBLE_DEVICES`
/// when using the PyTorch path). Returns `None` if neither PyTorch nor `nvidia-smi` works.
#[cfg(feature = "tch")]
pub fn query_free_vram_bytes() -> Option<u64> {
    query_free_vram_torch().or_else(query_free_vram_nvidia_smi)
}

#[cfg(feature = "tch")]
fn query_free_vram_torch() -> Option<u64> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("BOLTR_PYTHON") {
        let t = p.trim();
        if !t.is_empty() {
            candidates.push(t.to_string());
        }
    }
    candidates.push("python3".to_string());

    let script = r#"import torch
if torch.cuda.is_available():
    free, _total = torch.cuda.mem_get_info(0)
    print(int(free))
else:
    print(0)
"#;

    for py in candidates {
        let out = Command::new(&py).args(["-c", script]).output().ok()?;
        if !out.status.success() {
            continue;
        }
        let s = String::from_utf8_lossy(&out.stdout);
        let n: u64 = s.trim().parse().ok()?;
        return Some(n);
    }
    None
}

#[cfg(feature = "tch")]
fn nvidia_smi_gpu_index_arg() -> Option<String> {
    let s = std::env::var("CUDA_VISIBLE_DEVICES").ok()?;
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let first = t.split(',').next()?.trim();
    if first.is_empty() {
        return None;
    }
    if first.chars().all(|c| c.is_ascii_digit()) {
        Some(first.to_string())
    } else {
        None
    }
}

#[cfg(feature = "tch")]
fn query_free_vram_nvidia_smi() -> Option<u64> {
    let mut cmd = Command::new("nvidia-smi");
    if let Some(ref idx) = nvidia_smi_gpu_index_arg() {
        cmd.arg("-i").arg(idx);
    }
    let out = cmd
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().next()?.trim();
    let mib: u64 = line.parse().ok()?;
    Some(mib.saturating_mul(1024 * 1024))
}

#[cfg(test)]
mod tests {
    use super::auto_vram_decision;

    #[test]
    fn auto_vram_no_threshold_always_cuda() {
        assert!(auto_vram_decision(None, None));
        assert!(auto_vram_decision(None, Some(1)));
    }

    #[test]
    fn auto_vram_threshold_requires_free() {
        assert!(!auto_vram_decision(Some(1024), Some(512 * 1024 * 1024)));
        assert!(auto_vram_decision(Some(512), Some(512 * 1024 * 1024)));
    }

    #[test]
    fn auto_vram_probe_failed_fail_open() {
        assert!(auto_vram_decision(Some(9999), None));
    }
}
