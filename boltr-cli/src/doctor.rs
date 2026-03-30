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
            Ok(()) => DoctorJson {
                tch_feature: true,
                libtorch_runtime_ok: Some(true),
                libtorch_error: None,
            },
            Err(e) => DoctorJson {
                tch_feature: true,
                libtorch_runtime_ok: Some(false),
                libtorch_error: Some(e),
            },
        }
    }
    #[cfg(not(feature = "tch"))]
    {
        DoctorJson {
            tch_feature: false,
            libtorch_runtime_ok: None,
            libtorch_error: None,
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
