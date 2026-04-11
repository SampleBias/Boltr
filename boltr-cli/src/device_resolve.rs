//! Resolve `--device auto` / `--device gpu` and validate explicit GPU requests before preprocess / LibTorch.
//!
//! **Modes:** `auto` picks CUDA when LibTorch reports a GPU (`tch` / CUDA build), else CPU.
//! After resolution, [`maybe_apply_auto_vram_gate`] may downgrade `auto`→`cuda` to `cpu` when
//! `BOLTR_AUTO_MIN_FREE_VRAM_MB` is set and free VRAM is below threshold.
//! `gpu` requires CUDA and maps to LibTorch `cuda`. `BOLTR_DEVICE` overrides the CLI `--device` value
//! before resolution (see the `predict` command).

use anyhow::Result;

use crate::preprocess_cmd;

#[cfg(feature = "tch")]
use crate::cuda_mem;

#[cfg(feature = "tch")]
use anyhow::Context;
#[cfg(feature = "tch")]
use boltr_backend_tch::device::{cuda_is_available, parse_device_spec};

/// Resolve `auto` / `gpu` and validate explicit `cuda` / `cuda:N`. Returns `(resolved, requested_for_summary)`.
pub fn resolve_predict_device(raw: &str) -> Result<(String, Option<String>)> {
    let t = raw.trim();
    if t.is_empty() {
        anyhow::bail!("empty --device");
    }
    let lower = t.to_lowercase();

    if lower == "auto" {
        #[cfg(feature = "tch")]
        let resolved = if cuda_is_available() {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        };
        #[cfg(not(feature = "tch"))]
        let resolved = "cpu".to_string();
        return Ok((resolved, Some("auto".to_string())));
    }

    if lower == "gpu" {
        #[cfg(feature = "tch")]
        {
            if !cuda_is_available() {
                anyhow::bail!(
                    "CUDA requested (--device gpu) but no GPU is available; use --device cpu or --device auto"
                );
            }
            return Ok(("cuda".to_string(), Some("gpu".to_string())));
        }
        #[cfg(not(feature = "tch"))]
        anyhow::bail!(
            "GPU requested (--device gpu) but boltr was built without --features tch"
        );
    }

    if lower == "cpu" {
        return Ok(("cpu".to_string(), None));
    }

    if preprocess_cmd::predict_device_is_cuda(t) {
        #[cfg(feature = "tch")]
        {
            parse_device_spec(t).context("invalid GPU device for LibTorch")?;
        }
    }

    Ok((t.to_string(), None))
}

/// LibTorch CUDA ordinal from resolved device string (`cuda` / `cuda:N`).
#[cfg(feature = "tch")]
#[must_use]
pub fn libtorch_cuda_index_from_resolved(resolved: &str) -> usize {
    let s = resolved.trim().to_lowercase();
    if s == "cuda" || s == "gpu" {
        return 0;
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        return rest.parse().unwrap_or(0);
    }
    0
}

/// If `auto` resolved to `cuda` but `BOLTR_AUTO_MIN_FREE_VRAM_MB` is set and free VRAM is below threshold, use `cpu` for LibTorch.
#[cfg(feature = "tch")]
#[must_use]
pub fn maybe_apply_auto_vram_gate(resolved: String, device_requested: Option<&str>) -> String {
    if device_requested != Some("auto") || resolved != "cuda" {
        return resolved;
    }
    let Some(min_mb) = std::env::var("BOLTR_AUTO_MIN_FREE_VRAM_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
    else {
        return resolved;
    };
    let idx = libtorch_cuda_index_from_resolved(&resolved);
    if let Some(free) = cuda_mem::free_memory_mb_for_device_index(idx) {
        if free < min_mb {
            tracing::info!(
                free_mb = free,
                min_mb,
                cuda_index = idx,
                "auto: free VRAM below BOLTR_AUTO_MIN_FREE_VRAM_MB; using CPU for LibTorch"
            );
            return "cpu".to_string();
        }
    }
    resolved
}

#[cfg(not(feature = "tch"))]
#[must_use]
pub fn maybe_apply_auto_vram_gate(resolved: String, _device_requested: Option<&str>) -> String {
    resolved
}

#[cfg(test)]
mod tests {
    use super::resolve_predict_device;

    #[test]
    fn cpu_is_stable() {
        let (d, r) = resolve_predict_device("cpu").unwrap();
        assert_eq!(d, "cpu");
        assert!(r.is_none());
    }

    #[test]
    fn auto_always_resolves() {
        let (d, r) = resolve_predict_device("auto").unwrap();
        assert!(d == "cpu" || d == "cuda");
        assert_eq!(r.as_deref(), Some("auto"));
    }

    #[cfg(feature = "tch")]
    #[test]
    fn libtorch_cuda_index_from_resolved_maps() {
        use super::libtorch_cuda_index_from_resolved;
        assert_eq!(libtorch_cuda_index_from_resolved("cuda"), 0);
        assert_eq!(libtorch_cuda_index_from_resolved("cuda:2"), 2);
    }
}
