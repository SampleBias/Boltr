//! Load weights from Safetensors into `tch` tensors (checkpoint export is Python-side).

use std::path::Path;

use anyhow::{Context, Result};
use safetensors::tensor::Dtype;
use safetensors::SafeTensors;
use tch::{Device, Tensor};

fn tensor_from_view(view: safetensors::tensor::TensorView<'_>, device: Device) -> Result<Tensor> {
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let t = match view.dtype() {
        Dtype::F32 => {
            let raw = view.data();
            anyhow::ensure!(
                raw.len() % 4 == 0,
                "corrupt f32 tensor buffer length {}",
                raw.len()
            );
            let floats: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Tensor::from_slice(&floats).view(&shape).to_device(device)
        }
        Dtype::F64 => {
            let raw = view.data();
            anyhow::ensure!(raw.len() % 8 == 0, "corrupt f64 tensor buffer");
            let vals: Vec<f64> = raw
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Tensor::from_slice(&vals).view(&shape).to_device(device)
        }
        Dtype::I64 => {
            let raw = view.data();
            anyhow::ensure!(raw.len() % 8 == 0, "corrupt i64 tensor buffer");
            let vals: Vec<i64> = raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Tensor::from_slice(&vals).view(&shape).to_device(device)
        }
        other => anyhow::bail!(
            "dtype {:?} not yet supported in safetensors loader; export as f32 from Python for now",
            other
        ),
    };
    Ok(t)
}

/// Load one tensor by name from a `.safetensors` file.
pub fn load_tensor_from_safetensors(path: &Path, name: &str, device: Device) -> Result<Tensor> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse safetensors")?;
    let view = st
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?} in {}", path.display()))?;
    tensor_from_view(view, device)
}

/// List tensor names in a safetensors file (for debugging / key alignment).
pub fn list_safetensor_names(path: &Path) -> Result<Vec<String>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse safetensors")?;
    Ok(st.tensors().into_iter().map(|(n, _)| n).collect())
}
