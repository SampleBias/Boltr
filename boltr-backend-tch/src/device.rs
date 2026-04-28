//! Resolve compute device strings to LibTorch devices (CUDA when available).

use anyhow::{Context, Result};
use tch::{Device, Kind, Tensor};

/// Parse CLI/device spec: `cpu`, `cuda`, or `cuda:N` (e.g. `cuda:1`).
pub fn parse_device_spec(spec: &str) -> Result<Device> {
    let s = spec.trim().to_lowercase();
    if s == "cpu" {
        return Ok(Device::Cpu);
    }
    if s == "cuda" || s == "gpu" {
        return cuda_device(0);
    }
    if let Some(idx) = s.strip_prefix("cuda:") {
        let n: usize = idx
            .parse()
            .with_context(|| format!("invalid CUDA device index in {spec:?}"))?;
        return cuda_device(n);
    }
    anyhow::bail!(
        "unknown device {spec:?}; expected cpu, cuda, cuda:N (LibTorch must be CUDA-enabled for GPU)"
    );
}

fn cuda_device(index: usize) -> Result<Device> {
    if !tch::Cuda::is_available() {
        anyhow::bail!(
            "CUDA device requested but LibTorch was built without CUDA or no GPU is visible"
        );
    }
    Ok(Device::Cuda(index))
}

/// True if LibTorch reports at least one CUDA device.
pub fn cuda_is_available() -> bool {
    tch::Cuda::is_available()
}

/// Verify that CUDA kernels can actually execute, not only that a GPU is visible.
pub fn probe_cuda_runtime() -> Result<()> {
    if !tch::Cuda::is_available() {
        anyhow::bail!("LibTorch reports no visible CUDA device");
    }
    let t = Tensor::f_ones(&[1], (Kind::Float, Device::Cuda(0))).context("create CUDA tensor")?;
    let cpu = t
        .f_to_device(Device::Cpu)
        .context("copy CUDA tensor to CPU")?;
    let _ = cpu.f_double_value(&[0]).context("read CUDA tensor value")?;
    Ok(())
}
