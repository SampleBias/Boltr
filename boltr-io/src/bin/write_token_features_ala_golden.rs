//! Write ALA single-token `process_token_features` golden safetensors + `ala_structure_v2.npz`.
//!
//! ```text
//! cargo run -p boltr-io --bin write_token_features_ala_golden
//! ```
//!
//! Outputs (under `boltr-io/tests/fixtures/collate_golden/`):
//! - `token_features_ala_golden.safetensors` — per-example shapes (no batch axis)
//! - `token_features_ala_collated_golden.safetensors` — leading batch dim `B=1`
//! - `ala_structure_v2.npz` — for [`scripts/dump_token_features_ala_golden.py`](../../../scripts/dump_token_features_ala_golden.py)

use std::borrow::Cow;
use std::env;
use std::path::PathBuf;

use boltr_io::featurizer::{process_token_features, TokenFeatureTensors};
use boltr_io::fixtures::structure_v2_single_ala;
use boltr_io::structure_v2_npz::write_structure_v2_npz_compressed;
use boltr_io::tokenize::boltz2::tokenize_structure;
use safetensors::tensor::{serialize_to_file, Dtype, View};

struct OwnedTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
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

fn f32_flat(name: impl Into<String>, shape: Vec<usize>, slice: &[f32]) -> OwnedTensor {
    let mut data = Vec::with_capacity(slice.len() * 4);
    for v in slice {
        data.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        name: name.into(),
        dtype: Dtype::F32,
        shape,
        data,
    }
}

fn i64_flat(name: impl Into<String>, shape: Vec<usize>, slice: &[i64]) -> OwnedTensor {
    let mut data = Vec::with_capacity(slice.len() * 8);
    for v in slice {
        data.extend_from_slice(&v.to_le_bytes());
    }
    OwnedTensor {
        name: name.into(),
        dtype: Dtype::I64,
        shape,
        data,
    }
}

fn tensors_from_features(t: &TokenFeatureTensors, collated: bool) -> Vec<OwnedTensor> {
    let mut v = vec![
        i64_flat(
            "token_index",
            shape1(collated, t.token_index.len()),
            t.token_index.as_slice().unwrap(),
        ),
        i64_flat(
            "residue_index",
            shape1(collated, t.residue_index.len()),
            t.residue_index.as_slice().unwrap(),
        ),
        i64_flat(
            "asym_id",
            shape1(collated, t.asym_id.len()),
            t.asym_id.as_slice().unwrap(),
        ),
        i64_flat(
            "entity_id",
            shape1(collated, t.entity_id.len()),
            t.entity_id.as_slice().unwrap(),
        ),
        i64_flat(
            "sym_id",
            shape1(collated, t.sym_id.len()),
            t.sym_id.as_slice().unwrap(),
        ),
        i64_flat(
            "mol_type",
            shape1(collated, t.mol_type.len()),
            t.mol_type.as_slice().unwrap(),
        ),
        f32_flat(
            "res_type",
            shape2(collated, t.res_type.shape()),
            t.res_type.as_slice().unwrap(),
        ),
        f32_flat(
            "disto_center",
            shape2(collated, t.disto_center.shape()),
            t.disto_center.as_slice().unwrap(),
        ),
        f32_flat(
            "token_bonds",
            shape3(collated, t.token_bonds.shape()),
            t.token_bonds.as_slice().unwrap(),
        ),
        i64_flat(
            "type_bonds",
            shape2(collated, t.type_bonds.shape()),
            t.type_bonds.as_slice().unwrap(),
        ),
        f32_flat(
            "token_pad_mask",
            shape1(collated, t.token_pad_mask.len()),
            t.token_pad_mask.as_slice().unwrap(),
        ),
        f32_flat(
            "token_resolved_mask",
            shape1(collated, t.token_resolved_mask.len()),
            t.token_resolved_mask.as_slice().unwrap(),
        ),
        f32_flat(
            "token_disto_mask",
            shape1(collated, t.token_disto_mask.len()),
            t.token_disto_mask.as_slice().unwrap(),
        ),
        f32_flat(
            "contact_conditioning",
            shape3(collated, t.contact_conditioning.shape()),
            t.contact_conditioning.as_slice().unwrap(),
        ),
        f32_flat(
            "contact_threshold",
            shape2(collated, t.contact_threshold.shape()),
            t.contact_threshold.as_slice().unwrap(),
        ),
        i64_flat(
            "method_feature",
            shape1(collated, t.method_feature.len()),
            t.method_feature.as_slice().unwrap(),
        ),
        i64_flat(
            "modified",
            shape1(collated, t.modified.len()),
            t.modified.as_slice().unwrap(),
        ),
        f32_flat(
            "cyclic_period",
            shape1(collated, t.cyclic_period.len()),
            t.cyclic_period.as_slice().unwrap(),
        ),
        f32_flat(
            "affinity_token_mask",
            shape1(collated, t.affinity_token_mask.len()),
            t.affinity_token_mask.as_slice().unwrap(),
        ),
    ];
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

fn shape1(collated: bool, n: usize) -> Vec<usize> {
    if collated {
        vec![1, n]
    } else {
        vec![n]
    }
}

fn shape2(collated: bool, sh: &[usize]) -> Vec<usize> {
    if collated {
        let mut s = vec![1];
        s.extend_from_slice(sh);
        s
    } else {
        sh.to_vec()
    }
}

fn shape3(collated: bool, sh: &[usize]) -> Vec<usize> {
    shape2(collated, sh)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir: PathBuf = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("tests/fixtures/collate_golden");

    std::fs::create_dir_all(&dir)?;

    let s = structure_v2_single_ala();
    write_structure_v2_npz_compressed(&dir.join("ala_structure_v2.npz"), &s)?;

    let (tokens, bonds) = tokenize_structure(&s, None);
    let feat = process_token_features(&tokens, &bonds, None);

    let path_per = dir.join("token_features_ala_golden.safetensors");
    let per = tensors_from_features(&feat, false);
    let iter_per: Vec<_> = per.iter().map(|t| (t.name.clone(), t)).collect();
    serialize_to_file(iter_per, &None, &path_per)?;

    let path_col = dir.join("token_features_ala_collated_golden.safetensors");
    let col = tensors_from_features(&feat, true);
    let iter_col: Vec<_> = col.iter().map(|t| (t.name.clone(), t)).collect();
    serialize_to_file(iter_col, &None, &path_col)?;

    eprintln!(
        "Wrote {} and {} ({} tensors each), plus ala_structure_v2.npz",
        path_per.display(),
        path_col.display(),
        per.len()
    );
    Ok(())
}
