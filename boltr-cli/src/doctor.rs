//! `boltr doctor` — LibTorch / tch runtime probe for CI and boltr-web status.

use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct DoctorJson {
    /// `boltr` was built with `--features tch`.
    pub tch_feature: bool,
    /// Minimal CPU tensor op succeeded (only when `tch_feature`).
    pub libtorch_runtime_ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub libtorch_error: Option<String>,
    /// LibTorch sees at least one CUDA device (only when `libtorch_runtime_ok` is true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_available: Option<bool>,
    /// CUDA tensor smoke succeeded (detects visible-but-incompatible GPUs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_runtime_ok: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda_runtime_error: Option<String>,
    /// What `--device auto` resolves to (`cuda` or `cpu`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_auto_resolves_to: Option<String>,
}

pub fn run(json: bool) -> Result<()> {
    let out = doctor_payload();
    if json {
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("Boltr doctor");
        println!("  tch feature linked: {}", out.tch_feature);
        match out.libtorch_runtime_ok {
            Some(true) => println!("  LibTorch runtime: OK (CPU tensor smoke)"),
            Some(false) => println!(
                "  LibTorch runtime: FAIL — {}",
                out.libtorch_error.as_deref().unwrap_or("unknown")
            ),
            None => println!("  LibTorch runtime: (skipped — rebuild with --features tch)"),
        }
        if let Some(c) = out.cuda_available {
            println!("  CUDA available: {c}");
        }
        if let Some(c) = out.cuda_runtime_ok {
            println!("  CUDA runtime smoke: {c}");
            if !c {
                println!(
                    "  CUDA runtime error: {}",
                    out.cuda_runtime_error.as_deref().unwrap_or("unknown")
                );
            }
        }
        if let Some(ref d) = out.device_auto_resolves_to {
            println!("  --device auto resolves to: {d}");
        }
        #[cfg(feature = "tch")]
        if out.cuda_available == Some(true) {
            let n = crate::preprocess_cmd::parent_visible_cuda_device_count();
            if n == 1 {
                println!(
                    "  GPU memory hint: Boltz preprocess + LibTorch on one visible GPU can OOM; use --preprocess-boltz-cpu, or a second GPU via --preprocess-cuda-visible-devices, or --preprocess-auto-boltz-gpu only if you need Boltz on GPU with --device auto."
                );
            }
        }
    }
    if out.tch_feature && out.libtorch_runtime_ok == Some(false) {
        std::process::exit(1);
    }
    Ok(())
}

fn doctor_payload() -> DoctorJson {
    #[cfg(feature = "tch")]
    {
        match probe_libtorch() {
            Ok(()) => {
                let cuda = boltr_backend_tch::device::cuda_is_available();
                let cuda_runtime = if cuda {
                    match boltr_backend_tch::device::probe_cuda_runtime() {
                        Ok(()) => (Some(true), None),
                        Err(e) => (Some(false), Some(format!("{e:#}"))),
                    }
                } else {
                    (Some(false), None)
                };
                DoctorJson {
                    tch_feature: true,
                    libtorch_runtime_ok: Some(true),
                    libtorch_error: None,
                    cuda_available: Some(cuda),
                    cuda_runtime_ok: cuda_runtime.0,
                    cuda_runtime_error: cuda_runtime.1,
                    device_auto_resolves_to: Some(if cuda && cuda_runtime.0 == Some(true) {
                        "cuda".to_string()
                    } else {
                        "cpu".to_string()
                    }),
                }
            }
            Err(e) => DoctorJson {
                tch_feature: true,
                libtorch_runtime_ok: Some(false),
                libtorch_error: Some(e),
                cuda_available: None,
                cuda_runtime_ok: None,
                cuda_runtime_error: None,
                device_auto_resolves_to: None,
            },
        }
    }
    #[cfg(not(feature = "tch"))]
    {
        DoctorJson {
            tch_feature: false,
            libtorch_runtime_ok: None,
            libtorch_error: None,
            cuda_available: None,
            cuda_runtime_ok: None,
            cuda_runtime_error: None,
            device_auto_resolves_to: None,
        }
    }
}

#[cfg(feature = "tch")]
fn probe_libtorch() -> Result<(), String> {
    use tch::{Device, Kind, Tensor};
    let t = Tensor::zeros(&[2, 2], (Kind::Float, Device::Cpu));
    let _ = t.double_value(&[0, 0]);
    Ok(())
}
