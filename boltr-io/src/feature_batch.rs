//! Collated model features: `dict[str, Tensor]`-shaped batch for Boltz2-style inference (§4.5 scaffold).
//!
//! Full `collate()` in Python pads per key; here we provide **stack-after-pad** for fixed-shape
//! examples and typed storage. Extend when golden keys/shapes are frozen.

use std::collections::HashMap;

use ndarray::{stack, ArrayD, ArrayViewD, Axis};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CollateError {
    #[error("empty batch")]
    EmptyBatch,
    #[error("missing key {0:?} in one example")]
    MissingKey(String),
    #[error("dtype mismatch for key {key:?}: {a} vs {b}")]
    DtypeMismatch {
        key: String,
        a: &'static str,
        b: &'static str,
    },
    #[error("shape mismatch for key {key:?}: {a:?} vs {b:?}")]
    ShapeMismatch {
        key: String,
        a: Vec<usize>,
        b: Vec<usize>,
    },
    #[error("ndarray stack: {0}")]
    Stack(#[from] ndarray::ShapeError),
}

/// One featurizer tensor (Boltz uses float masks, int indices, etc.).
#[derive(Clone, Debug)]
pub enum FeatureTensor {
    F32(ArrayD<f32>),
    I64(ArrayD<i64>),
    I32(ArrayD<i32>),
}

impl FeatureTensor {
    fn kind(&self) -> &'static str {
        match self {
            FeatureTensor::F32(_) => "f32",
            FeatureTensor::I64(_) => "i64",
            FeatureTensor::I32(_) => "i32",
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            FeatureTensor::F32(a) => a.shape().to_vec(),
            FeatureTensor::I64(a) => a.shape().to_vec(),
            FeatureTensor::I32(a) => a.shape().to_vec(),
        }
    }
}

/// Batch of named feature arrays (one example = one map; collate adds leading batch dimension).
#[derive(Clone, Debug, Default)]
pub struct FeatureBatch {
    pub tensors: HashMap<String, FeatureTensor>,
}

impl FeatureBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_f32(&mut self, key: impl Into<String>, value: ArrayD<f32>) {
        self.tensors.insert(key.into(), FeatureTensor::F32(value));
    }

    pub fn insert_i64(&mut self, key: impl Into<String>, value: ArrayD<i64>) {
        self.tensors.insert(key.into(), FeatureTensor::I64(value));
    }

    pub fn insert_i32(&mut self, key: impl Into<String>, value: ArrayD<i32>) {
        self.tensors.insert(key.into(), FeatureTensor::I32(value));
    }

    pub fn get_f32(&self, key: &str) -> Option<&ArrayD<f32>> {
        match self.tensors.get(key)? {
            FeatureTensor::F32(a) => Some(a),
            _ => None,
        }
    }

    pub fn get_i64(&self, key: &str) -> Option<&ArrayD<i64>> {
        match self.tensors.get(key)? {
            FeatureTensor::I64(a) => Some(a),
            _ => None,
        }
    }

    pub fn get_i32(&self, key: &str) -> Option<&ArrayD<i32>> {
        match self.tensors.get(key)? {
            FeatureTensor::I32(a) => Some(a),
            _ => None,
        }
    }

    pub fn merge(&mut self, other: FeatureBatch) {
        self.tensors.extend(other.tensors);
    }
}

/// Stack a list of **same-shape** `f32` arrays along a new axis 0 (batch dimension).
pub fn stack_f32_views(views: &[ArrayViewD<f32>]) -> Result<ArrayD<f32>, CollateError> {
    if views.is_empty() {
        return Err(CollateError::EmptyBatch);
    }
    let shape0 = views[0].shape();
    for v in views.iter().skip(1) {
        if v.shape() != shape0 {
            return Err(CollateError::ShapeMismatch {
                key: "<stack_f32>".to_string(),
                a: shape0.to_vec(),
                b: v.shape().to_vec(),
            });
        }
    }
    Ok(stack(Axis(0), views)?)
}

/// Collate `FeatureBatch` examples that already have **identical** shapes per key (post-pad).
/// Prepends batch dimension axis 0 to every tensor.
pub fn collate_feature_batches(examples: &[FeatureBatch]) -> Result<FeatureBatch, CollateError> {
    if examples.is_empty() {
        return Err(CollateError::EmptyBatch);
    }
    let mut keys: Vec<String> = examples[0].tensors.keys().cloned().collect();
    keys.sort_unstable();
    for ex in examples.iter().skip(1) {
        for k in &keys {
            if !ex.tensors.contains_key(k) {
                return Err(CollateError::MissingKey(k.clone()));
            }
        }
        if ex.tensors.len() != keys.len() {
            return Err(CollateError::MissingKey(
                "<key set differs from first example>".to_string(),
            ));
        }
    }

    let mut out = FeatureBatch::new();
    for k in keys {
        let first = examples[0].tensors.get(&k).unwrap();
        let kind = first.kind();
        for ex in examples.iter().skip(1) {
            let t = ex.tensors.get(&k).unwrap();
            if t.kind() != kind {
                return Err(CollateError::DtypeMismatch {
                    key: k.clone(),
                    a: kind,
                    b: t.kind(),
                });
            }
            if t.shape() != first.shape() {
                return Err(CollateError::ShapeMismatch {
                    key: k.clone(),
                    a: first.shape(),
                    b: t.shape(),
                });
            }
        }

        match first {
            FeatureTensor::F32(_) => {
                let views: Vec<_> = examples
                    .iter()
                    .map(|e| e.get_f32(&k).unwrap().view())
                    .collect();
                out.insert_f32(k, stack_f32_views(&views)?);
            }
            FeatureTensor::I64(_) => {
                let views: Vec<_> = examples.iter().map(|e| match &e.tensors[&k] {
                    FeatureTensor::I64(a) => a.view(),
                    _ => unreachable!(),
                }).collect();
                out.insert_i64(k, stack(Axis(0), &views)?);
            }
            FeatureTensor::I32(_) => {
                let views: Vec<_> = examples.iter().map(|e| match &e.tensors[&k] {
                    FeatureTensor::I32(a) => a.view(),
                    _ => unreachable!(),
                }).collect();
                out.insert_i32(k, stack(Axis(0), &views)?);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn collate_f32_and_i64() {
        let mut a = FeatureBatch::new();
        a.insert_f32("x", arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn());
        a.insert_i64("y", arr2(&[[1_i64, 2]]).into_dyn());
        let mut b = FeatureBatch::new();
        b.insert_f32("x", arr2(&[[10.0_f32, 20.0], [30.0, 40.0]]).into_dyn());
        b.insert_i64("y", arr2(&[[10_i64, 20]]).into_dyn());

        let batched = collate_feature_batches(&[a, b]).unwrap();
        let x = batched.get_f32("x").unwrap();
        assert_eq!(x.shape(), &[2, 2, 2]);
        assert_eq!(x[[0, 0, 0]], 1.0);
        assert_eq!(x[[1, 0, 0]], 10.0);
        match &batched.tensors["y"] {
            FeatureTensor::I64(y) => {
                assert_eq!(y.shape(), &[2, 1, 2]);
            }
            _ => panic!("expected i64"),
        }
    }

    #[test]
    fn collate_errors_on_shape_mismatch() {
        let mut a = FeatureBatch::new();
        a.insert_f32("x", arr2(&[[1.0_f32]]).into_dyn());
        let mut b = FeatureBatch::new();
        b.insert_f32("x", arr2(&[[1.0_f32, 2.0]]).into_dyn());
        assert!(collate_feature_batches(&[a, b]).is_err());
    }
}
