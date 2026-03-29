//! Serialize [`InferenceCollateResult`](crate::collate_pad::InferenceCollateResult) to safetensors for
//! Rust-native trunk collate goldens (same pipeline as tests).

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::path::Path;

use ndarray::ArrayD;
use safetensors::tensor::{serialize_to_file, Dtype, View};

use crate::collate_pad::InferenceCollateResult;
use crate::feature_batch::FeatureTensor;

/// `token_s` / embedder trunk dim (Boltz2 default).
pub const TRUNK_COLLATE_S_INPUT_LAST: usize = 384;

/// Flat buffer for one tensor (dynamic name for safetensors).
pub struct BufTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &BufTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn prepend_batch_dim(ft: &FeatureTensor) -> FeatureTensor {
    fn prepend_f32(a: &ArrayD<f32>) -> ArrayD<f32> {
        let v = a.shape();
        let mut shape = vec![1usize];
        shape.extend_from_slice(v);
        let flat: Vec<f32> = a.iter().cloned().collect();
        ArrayD::from_shape_vec(shape, flat).expect("prepend_batch_dim f32")
    }
    fn prepend_i64(a: &ArrayD<i64>) -> ArrayD<i64> {
        let v = a.shape();
        let mut shape = vec![1usize];
        shape.extend_from_slice(v);
        let flat: Vec<i64> = a.iter().cloned().collect();
        ArrayD::from_shape_vec(shape, flat).expect("prepend_batch_dim i64")
    }
    fn prepend_i32(a: &ArrayD<i32>) -> ArrayD<i32> {
        let v = a.shape();
        let mut shape = vec![1usize];
        shape.extend_from_slice(v);
        let flat: Vec<i32> = a.iter().cloned().collect();
        ArrayD::from_shape_vec(shape, flat).expect("prepend_batch_dim i32")
    }
    match ft {
        FeatureTensor::F32(a) => FeatureTensor::F32(prepend_f32(a)),
        FeatureTensor::I64(a) => FeatureTensor::I64(prepend_i64(a)),
        FeatureTensor::I32(a) => FeatureTensor::I32(prepend_i32(a)),
    }
}

fn f32_array_to_bytes(a: &ArrayD<f32>) -> Vec<u8> {
    a.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn i64_array_to_bytes(a: &ArrayD<i64>) -> Vec<u8> {
    a.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn i32_array_to_bytes(a: &ArrayD<i32>) -> Vec<u8> {
    a.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn feature_tensor_to_buf(name: String, ft: &FeatureTensor) -> BufTensor {
    match ft {
        FeatureTensor::F32(a) => BufTensor {
            name,
            dtype: Dtype::F32,
            shape: a.shape().to_vec(),
            data: f32_array_to_bytes(a),
        },
        FeatureTensor::I64(a) => BufTensor {
            name,
            dtype: Dtype::I64,
            shape: a.shape().to_vec(),
            data: i64_array_to_bytes(a),
        },
        FeatureTensor::I32(a) => BufTensor {
            name,
            dtype: Dtype::I32,
            shape: a.shape().to_vec(),
            data: i32_array_to_bytes(a),
        },
    }
}

/// Build ordered tensor list: stacked `batch` keys, excluded keys with leading batch dim `1`, and
/// `s_inputs` `[B, N, TRUNK_COLLATE_S_INPUT_LAST]` (zeros — embedder output placeholder for backend smoke).
pub fn inference_collate_to_golden_tensors(
    coll: &InferenceCollateResult,
) -> Result<Vec<BufTensor>, String> {
    let mut keys: BTreeSet<String> = coll.batch.tensors.keys().cloned().collect();
    for k in coll.excluded.keys() {
        keys.insert(k.clone());
    }

    let mut out: Vec<BufTensor> = Vec::new();

    for name in keys.iter() {
        if let Some(ft) = coll.batch.tensors.get(name) {
            out.push(feature_tensor_to_buf(name.clone(), ft));
            continue;
        }
        if let Some(v) = coll.excluded.get(name) {
            let first = v.first().ok_or_else(|| format!("excluded {name} empty"))?;
            let stacked = prepend_batch_dim(first);
            out.push(feature_tensor_to_buf(name.clone(), &stacked));
            continue;
        }
    }

    // s_inputs: zeros [B, N, token_s] from token_pad_mask in batch
    let tpm = coll
        .batch
        .get_f32("token_pad_mask")
        .ok_or_else(|| "missing token_pad_mask for s_inputs shape".to_string())?;
    let shape = tpm.shape();
    if shape.len() != 2 {
        return Err("token_pad_mask must be rank-2 [B, N]".to_string());
    }
    let b = shape[0];
    let n = shape[1];
    let s = ndarray::Array3::<f32>::zeros((b, n, TRUNK_COLLATE_S_INPUT_LAST)).into_dyn();
    out.push(BufTensor {
        name: "s_inputs".to_string(),
        dtype: Dtype::F32,
        shape: s.shape().to_vec(),
        data: f32_array_to_bytes(&s),
    });

    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Write golden safetensors (sorted tensor names) to `path`.
pub fn write_inference_collate_golden(
    coll: &InferenceCollateResult,
    path: &Path,
) -> Result<(), String> {
    let tensors = inference_collate_to_golden_tensors(coll)?;
    let iter = tensors
        .iter()
        .map(|t| (t.name.clone(), t))
        .collect::<Vec<_>>();
    serialize_to_file(iter, &None, path).map_err(|e| e.to_string())
}

/// Compare `coll` (+ synthetic `s_inputs`) to a safetensors byte slice. Returns first mismatch description.
pub fn compare_inference_collate_to_safetensors(
    coll: &InferenceCollateResult,
    golden_bytes: &[u8],
    f32_atol: f32,
    f32_rtol: f32,
) -> Result<(), String> {
    use safetensors::SafeTensors;

    let tensors = inference_collate_to_golden_tensors(coll).map_err(|e| e.to_string())?;
    let mut rust_map: std::collections::HashMap<String, BufTensor> =
        std::collections::HashMap::new();
    for t in tensors {
        rust_map.insert(t.name.clone(), t);
    }

    let st = SafeTensors::deserialize(golden_bytes).map_err(|e| e.to_string())?;
    let mut golden_names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n.to_string()).collect();
    golden_names.sort();

    let mut rust_names: Vec<String> = rust_map.keys().cloned().collect();
    rust_names.sort();

    if golden_names != rust_names {
        return Err(format!(
            "key set mismatch\ngolden: {:?}\nrust: {:?}",
            golden_names, rust_names
        ));
    }

    for name in &golden_names {
        let view = st.tensor(name).map_err(|e| e.to_string())?;
        let buf = rust_map.get(name).unwrap();

        if view.dtype() != buf.dtype {
            return Err(format!("{name}: dtype mismatch"));
        }
        if view.shape() != buf.shape.as_slice() {
            return Err(format!(
                "{name}: shape mismatch golden {:?} vs rust {:?}",
                view.shape(),
                buf.shape
            ));
        }

        match view.dtype() {
            Dtype::F32 => {
                let g = read_f32_le(view.data());
                let r = bytes_to_f32(&buf.data);
                if g.len() != r.len() {
                    return Err(format!("{name}: len mismatch"));
                }
                for (i, (&gv, &rv)) in g.iter().zip(r.iter()).enumerate() {
                    let diff = (gv - rv).abs();
                    let scale = f32_rtol * gv.abs().max(rv.abs()).max(1e-8);
                    if diff > f32_atol + scale {
                        return Err(format!(
                            "{name}: mismatch at flat index {i}: golden {gv} vs rust {rv}"
                        ));
                    }
                }
            }
            Dtype::I64 => {
                let g = read_i64_le(view.data());
                let r = bytes_to_i64(&buf.data);
                if g != r {
                    return Err(format!("{name}: i64 data mismatch (first diff)"));
                }
            }
            Dtype::I32 => {
                let g = read_i32_le(view.data());
                let r = bytes_to_i32(&buf.data);
                if g != r {
                    return Err(format!("{name}: i32 data mismatch"));
                }
            }
            _ => return Err(format!("{name}: unsupported dtype")),
        }
    }

    Ok(())
}

fn read_f32_le(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    read_f32_le(b)
}

fn read_i64_le(b: &[u8]) -> Vec<i64> {
    b.chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bytes_to_i64(b: &[u8]) -> Vec<i64> {
    read_i64_le(b)
}

fn read_i32_le(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bytes_to_i32(b: &[u8]) -> Vec<i32> {
    read_i32_le(b)
}
