//! Python-aligned `pad_to_max` and inference `collate` ([`pad.py`](../../boltz-reference/src/boltz/data/pad.py),
//! [`inferencev2.collate`](../../boltz-reference/src/boltz/data/module/inferencev2.py)).

use std::collections::{BTreeSet, HashMap};

use ndarray::{indices, stack, ArrayD, Axis};
use thiserror::Error;

use crate::feature_batch::{
    CollateError, FeatureBatch, FeatureTensor, INFERENCE_COLLATE_EXCLUDED_KEYS,
};

#[derive(Debug, Error)]
pub enum InferenceCollateError {
    #[error("empty batch")]
    EmptyBatch,
    #[error("key set differs between examples")]
    KeySetMismatch,
    #[error("missing key {0:?} in one example")]
    MissingKey(String),
    #[error("dtype mismatch for key {key:?}: {a} vs {b}")]
    DtypeMismatch {
        key: String,
        a: &'static str,
        b: &'static str,
    },
    #[error("ndim mismatch for key {key:?}: {a} vs {b}")]
    NdimMismatch { key: String, a: usize, b: usize },
    #[error("ndarray stack: {0}")]
    Stack(#[from] ndarray::ShapeError),
}

fn collate_err_to_inference(e: CollateError) -> InferenceCollateError {
    match e {
        CollateError::EmptyBatch => InferenceCollateError::EmptyBatch,
        CollateError::ShapeMismatch { key: _, a, b } => InferenceCollateError::NdimMismatch {
            key: "<pad_to_max_f32>".to_string(),
            a: a.len(),
            b: b.len(),
        },
        CollateError::MissingKey(k) => InferenceCollateError::MissingKey(k),
        CollateError::DtypeMismatch { key, a, b } => {
            InferenceCollateError::DtypeMismatch { key, a, b }
        }
        CollateError::Stack(s) => InferenceCollateError::Stack(s),
    }
}

/// Padded batch + per-cell mask (`1` = original cell, `0` = padding), matching Python `pad_to_max`.
#[derive(Clone, Debug)]
pub struct PadToMaxResult {
    pub data: ArrayD<f32>,
    pub padding_mask: ArrayD<f32>,
}

/// Result of [`collate_inference_batches`]: stacked tensor keys plus excluded keys as per-example lists.
#[derive(Clone, Debug, Default)]
pub struct InferenceCollateResult {
    pub batch: FeatureBatch,
    /// Keys in [`INFERENCE_COLLATE_EXCLUDED_KEYS`] present in inputs: one tensor per example (not stacked).
    pub excluded: HashMap<String, Vec<FeatureTensor>>,
}

fn max_shape(shapes: &[Vec<usize>]) -> Vec<usize> {
    if shapes.is_empty() {
        return Vec::new();
    }
    let nd = shapes[0].len();
    let mut m = vec![0usize; nd];
    for s in shapes {
        for i in 0..nd {
            m[i] = m[i].max(s[i]);
        }
    }
    m
}

fn pad_array_to_max_f32(
    a: &ArrayD<f32>,
    max_shape: &[usize],
    value: f32,
) -> Result<(ArrayD<f32>, ArrayD<f32>), CollateError> {
    if a.shape().len() != max_shape.len() {
        return Err(CollateError::ShapeMismatch {
            key: "<pad>".to_string(),
            a: a.shape().to_vec(),
            b: max_shape.to_vec(),
        });
    }
    for (&ms, &as_) in max_shape.iter().zip(a.shape().iter()) {
        if as_ > ms {
            return Err(CollateError::ShapeMismatch {
                key: "<pad>".to_string(),
                a: a.shape().to_vec(),
                b: max_shape.to_vec(),
            });
        }
    }
    let mut out = ArrayD::from_elem(max_shape, value);
    let mut mask = ArrayD::zeros(max_shape);
    for idx in indices(a.raw_dim()) {
        let v = a[&idx];
        out[&idx] = v;
        mask[&idx] = 1.0f32;
    }
    Ok((out, mask))
}

fn pad_array_to_max_i64(
    a: &ArrayD<i64>,
    max_shape: &[usize],
    value: i64,
) -> Result<(ArrayD<i64>, ArrayD<f32>), CollateError> {
    if a.shape().len() != max_shape.len() {
        return Err(CollateError::ShapeMismatch {
            key: "<pad>".to_string(),
            a: a.shape().to_vec(),
            b: max_shape.to_vec(),
        });
    }
    let mut out = ArrayD::from_elem(max_shape, value);
    let mut mask = ArrayD::zeros(max_shape);
    for idx in indices(a.raw_dim()) {
        let v = a[&idx];
        out[&idx] = v;
        mask[&idx] = 1.0f32;
    }
    Ok((out, mask))
}

fn pad_array_to_max_i32(
    a: &ArrayD<i32>,
    max_shape: &[usize],
    value: i32,
) -> Result<(ArrayD<i32>, ArrayD<f32>), CollateError> {
    if a.shape().len() != max_shape.len() {
        return Err(CollateError::ShapeMismatch {
            key: "<pad>".to_string(),
            a: a.shape().to_vec(),
            b: max_shape.to_vec(),
        });
    }
    let mut out = ArrayD::from_elem(max_shape, value);
    let mut mask = ArrayD::zeros(max_shape);
    for idx in indices(a.raw_dim()) {
        let v = a[&idx];
        out[&idx] = v;
        mask[&idx] = 1.0f32;
    }
    Ok((out, mask))
}

/// Pad a list of **same-rank** tensors to the per-dimension maximum, then stack on axis 0.
/// Matches [`pad_to_max`](../../boltz-reference/src/boltz/data/pad.py) (numeric tensors only).
pub fn pad_to_max_f32(data: &[ArrayD<f32>], value: f32) -> Result<PadToMaxResult, CollateError> {
    if data.is_empty() {
        return Err(CollateError::EmptyBatch);
    }
    let nd = data[0].ndim();
    for a in data.iter().skip(1) {
        if a.ndim() != nd {
            return Err(CollateError::ShapeMismatch {
                key: "<pad_to_max_f32>".to_string(),
                a: data[0].shape().to_vec(),
                b: a.shape().to_vec(),
            });
        }
    }
    let shapes: Vec<Vec<usize>> = data.iter().map(|a| a.shape().to_vec()).collect();
    if data.iter().all(|a| a.shape() == data[0].shape()) {
        let views: Vec<_> = data.iter().map(|a| a.view()).collect();
        let stacked = stack(Axis(0), &views)?;
        let mask = ndarray::Array::ones(stacked.raw_dim());
        return Ok(PadToMaxResult {
            data: stacked,
            padding_mask: mask,
        });
    }
    let max_shape = max_shape(&shapes);
    let mut padded = Vec::new();
    let mut masks = Vec::new();
    for a in data {
        let (p, m) = pad_array_to_max_f32(a, &max_shape, value)?;
        padded.push(p);
        masks.push(m);
    }
    Ok(PadToMaxResult {
        data: stack(
            Axis(0),
            &padded.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?,
        padding_mask: stack(Axis(0), &masks.iter().map(|a| a.view()).collect::<Vec<_>>())?,
    })
}

/// Collate like Python `Boltz2InferenceDataModule.collate`: stack when shapes match; else `pad_to_max` with `value`.
/// Excluded keys are **not** stacked; each is collected as `Vec<FeatureTensor>` (one per example).
pub fn collate_inference_batches(
    examples: &[FeatureBatch],
    pad_value_f32: f32,
    pad_value_i64: i64,
    pad_value_i32: i32,
) -> Result<InferenceCollateResult, InferenceCollateError> {
    if examples.is_empty() {
        return Err(InferenceCollateError::EmptyBatch);
    }
    let keys0: BTreeSet<_> = examples[0].tensors.keys().cloned().collect();
    for ex in examples.iter().skip(1) {
        let ks: BTreeSet<_> = ex.tensors.keys().cloned().collect();
        if ks != keys0 {
            return Err(InferenceCollateError::KeySetMismatch);
        }
    }

    let mut batch = FeatureBatch::new();
    let mut excluded: HashMap<String, Vec<FeatureTensor>> = HashMap::new();

    for key in &keys0 {
        if INFERENCE_COLLATE_EXCLUDED_KEYS.contains(&key.as_str()) {
            let mut vec = Vec::with_capacity(examples.len());
            for ex in examples {
                vec.push(
                    ex.tensors
                        .get(key)
                        .cloned()
                        .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))?,
                );
            }
            excluded.insert(key.clone(), vec);
            continue;
        }

        let first = examples[0]
            .tensors
            .get(key)
            .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))?;
        let kind = first.kind();
        for ex in examples.iter().skip(1) {
            let t = ex
                .tensors
                .get(key)
                .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))?;
            if t.kind() != kind {
                return Err(InferenceCollateError::DtypeMismatch {
                    key: key.clone(),
                    a: kind,
                    b: t.kind(),
                });
            }
            if t.shape().len() != first.shape().len() {
                return Err(InferenceCollateError::NdimMismatch {
                    key: key.clone(),
                    a: first.shape().len(),
                    b: t.shape().len(),
                });
            }
        }

        match first {
            FeatureTensor::F32(_) => {
                let arrays: Vec<_> = examples
                    .iter()
                    .map(|e| {
                        e.get_f32(key)
                            .cloned()
                            .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))
                    })
                    .collect::<Result<_, _>>()?;
                let p = pad_to_max_f32(&arrays, pad_value_f32).map_err(collate_err_to_inference)?;
                batch.insert_f32(key.clone(), p.data);
            }
            FeatureTensor::I64(_) => {
                let first_shape = examples[0]
                    .get_i64(key)
                    .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))?
                    .shape()
                    .to_vec();
                let all_same = examples.iter().all(|e| {
                    e.get_i64(key)
                        .map(|a| a.shape() == first_shape.as_slice())
                        .unwrap_or(false)
                });
                if all_same {
                    let views: Vec<_> = examples
                        .iter()
                        .map(|e| e.get_i64(key).unwrap().view())
                        .collect();
                    batch.insert_i64(key.clone(), stack(Axis(0), &views)?);
                } else {
                    let shapes: Vec<Vec<usize>> = examples
                        .iter()
                        .map(|e| {
                            e.get_i64(key)
                                .map(|a| a.shape().to_vec())
                                .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))
                        })
                        .collect::<Result<_, _>>()?;
                    let max_shape = max_shape(&shapes);
                    let mut padded = Vec::new();
                    for ex in examples {
                        let a = ex.get_i64(key).unwrap();
                        let (p, _) = pad_array_to_max_i64(a, &max_shape, pad_value_i64).map_err(
                            |e| match e {
                                CollateError::ShapeMismatch { key: _, a, b } => {
                                    InferenceCollateError::NdimMismatch {
                                        key: key.clone(),
                                        a: a.len(),
                                        b: b.len(),
                                    }
                                }
                                _ => InferenceCollateError::KeySetMismatch,
                            },
                        )?;
                        padded.push(p);
                    }
                    batch.insert_i64(
                        key.clone(),
                        stack(
                            Axis(0),
                            &padded.iter().map(|a| a.view()).collect::<Vec<_>>(),
                        )?,
                    );
                }
            }
            FeatureTensor::I32(_) => {
                let first_shape = examples[0]
                    .get_i32(key)
                    .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))?
                    .shape()
                    .to_vec();
                let all_same = examples.iter().all(|e| {
                    e.get_i32(key)
                        .map(|a| a.shape() == first_shape.as_slice())
                        .unwrap_or(false)
                });
                if all_same {
                    let views: Vec<_> = examples
                        .iter()
                        .map(|e| e.get_i32(key).unwrap().view())
                        .collect();
                    batch.insert_i32(key.clone(), stack(Axis(0), &views)?);
                } else {
                    let shapes: Vec<Vec<usize>> = examples
                        .iter()
                        .map(|e| {
                            e.get_i32(key)
                                .map(|a| a.shape().to_vec())
                                .ok_or_else(|| InferenceCollateError::MissingKey(key.clone()))
                        })
                        .collect::<Result<_, _>>()?;
                    let max_shape = max_shape(&shapes);
                    let mut padded = Vec::new();
                    for ex in examples {
                        let a = ex.get_i32(key).unwrap();
                        let (p, _) = pad_array_to_max_i32(a, &max_shape, pad_value_i32).map_err(
                            |e| match e {
                                CollateError::ShapeMismatch { key: _, a, b } => {
                                    InferenceCollateError::NdimMismatch {
                                        key: key.clone(),
                                        a: a.len(),
                                        b: b.len(),
                                    }
                                }
                                _ => InferenceCollateError::KeySetMismatch,
                            },
                        )?;
                        padded.push(p);
                    }
                    batch.insert_i32(
                        key.clone(),
                        stack(
                            Axis(0),
                            &padded.iter().map(|a| a.view()).collect::<Vec<_>>(),
                        )?,
                    );
                }
            }
        }
    }

    Ok(InferenceCollateResult { batch, excluded })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn pad_to_max_two_different_shapes() {
        let a = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn();
        let b = arr2(&[[10.0_f32, 20.0, 30.0], [40.0, 50.0, 60.0]]).into_dyn();
        let p = pad_to_max_f32(&[a, b], 0.0).unwrap();
        assert_eq!(p.data.shape(), &[2, 2, 3]);
        assert_eq!(p.data[[0, 0, 2]], 0.0);
        assert_eq!(p.data[[0, 1, 2]], 0.0);
        assert_eq!(p.data[[1, 0, 0]], 10.0);
        assert_eq!(p.padding_mask[[0, 0, 2]], 0.0);
        assert_eq!(p.padding_mask[[1, 0, 2]], 1.0);
    }

    #[test]
    fn collate_inference_msa_like_keys() {
        let mut a = FeatureBatch::new();
        a.insert_i64("msa", arr2(&[[1_i64, 2], [3, 4]]).into_dyn());
        let mut b = FeatureBatch::new();
        b.insert_i64("msa", arr2(&[[10_i64, 20, 30], [40, 50, 60]]).into_dyn());

        let out = collate_inference_batches(&[a, b], 0.0, 0, 0).unwrap();
        let m = out.batch.get_i64("msa").unwrap();
        assert_eq!(m.shape(), &[2, 2, 3]);
        assert_eq!(m[[0, 0, 2]], 0);
        assert_eq!(m[[1, 0, 2]], 30);
    }

    #[test]
    fn collate_inference_excluded_record_not_stacked() {
        let mut a = FeatureBatch::new();
        a.insert_f32("x", arr2(&[[1.0_f32]]).into_dyn());
        a.insert_i64("record", arr2(&[[1_i64]]).into_dyn());

        let mut b = FeatureBatch::new();
        b.insert_f32("x", arr2(&[[2.0_f32]]).into_dyn());
        b.insert_i64("record", arr2(&[[2_i64]]).into_dyn());

        let out = collate_inference_batches(&[a, b], 0.0, 0, 0).unwrap();
        assert_eq!(out.batch.get_f32("x").unwrap().shape(), &[2, 1, 1]);
        assert_eq!(out.excluded.len(), 1);
        assert_eq!(out.excluded["record"].len(), 2);
    }
}
