//! Resolve `--device auto` / `--device gpu` and validate explicit GPU requests before preprocess / LibTorch.
//!
//! **Modes:** `auto` picks CUDA when LibTorch reports a GPU (`tch` / CUDA build), free VRAM is
//! sufficient when `BOLTR_AUTO_MIN_FREE_VRAM_MB` is set (probe fail-open keeps CUDA), else CPU.
//! `gpu` / explicit `cuda:N` require CUDA and are never downgraded. `cpu` is fixed CPU.
//! `BOLTR_DEVICE` overrides the CLI `--device` value before resolution (see the `predict` command).

use anyhow::Result;

use crate::preprocess_cmd;

#[cfg(feature = "tch")]
use anyhow::Context;
#[cfg(feature = "tch")]
use boltr_backend_tch::device::{cuda_is_available, parse_device_spec, probe_cuda_runtime};

/// Resolve `auto` / `gpu` and validate explicit `cuda` / `cuda:N`. Returns `(resolved, requested_for_summary)`.
pub fn resolve_predict_device(raw: &str) -> Result<(String, Option<String>)> {
    let t = raw.trim();
    if t.is_empty() {
        anyhow::bail!("empty --device");
    }
    let lower = t.to_lowercase();

    if lower == "auto" {
        #[cfg(feature = "tch")]
        let resolved = if cuda_is_available()
            && crate::gpu_mem::auto_vram_allows_cuda()
            && probe_cuda_runtime().is_ok()
        {
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
            probe_cuda_runtime()
                .context("CUDA requested (--device gpu) but LibTorch CUDA kernel smoke failed")?;
            return Ok(("cuda".to_string(), Some("gpu".to_string())));
        }
        #[cfg(not(feature = "tch"))]
        anyhow::bail!("GPU requested (--device gpu) but boltr was built without --features tch");
    }

    if lower == "cpu" {
        return Ok(("cpu".to_string(), None));
    }

    if preprocess_cmd::predict_device_is_cuda(t) {
        #[cfg(feature = "tch")]
        {
            parse_device_spec(t).context("invalid GPU device for LibTorch")?;
            probe_cuda_runtime().context("CUDA requested but LibTorch CUDA kernel smoke failed")?;
        }
    }

    Ok((t.to_string(), None))
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
}
